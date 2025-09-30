#!/usr/bin/env python3
"""
Fast Nesting Engine - Optimized for Performance
Addresses timeout issues with intelligent algorithms and early termination
"""

import math
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import concurrent.futures
from threading import Lock

@dataclass
class Part:
    id: str
    width: float
    height: float
    quantity: int
    rotation_allowed: bool = True
    area: float = 0.0
    svg_path: str = ""
    
    def __post_init__(self):
        if self.area == 0.0:
            self.area = self.width * self.height

@dataclass
class Board:
    id: str
    width: float
    height: float
    cost: float
    quantity_available: int
    area: float = 0.0
    
    def __post_init__(self):
        if self.area == 0.0:
            self.area = self.width * self.height

class FastNestingEngine:
    """High-performance nesting engine with timeout protection and smart algorithms"""
    
    def __init__(self, margin_mm: float = 10.0, min_gap_mm: float = 5.0, timeout_seconds: int = 30):
        self.margin_mm = margin_mm
        self.min_gap_mm = min_gap_mm
        self.timeout_seconds = timeout_seconds
        self.debug = False
        self.start_time = None
        self._lock = Lock()
        
    def log(self, message: str):
        """Debug logging with time tracking"""
        if self.debug:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"[FAST_NESTING] {elapsed:.1f}s: {message}")
    
    def optimize_nesting(self, parts: List[Part], boards: List[Board], max_boards: int = 5) -> Dict:
        """Main optimization function with timeout protection and smart strategy selection"""
        
        self.start_time = time.time()
        self.log(f"Starting fast nesting with {len(parts)} part types, {sum(p.quantity for p in parts)} total parts")
        
        # Quick validation
        if not parts or not boards:
            return self._create_failure_result("No parts or boards provided")
        
        # Calculate total area needed for quick feasibility check
        total_parts = sum(part.quantity for part in parts)
        total_area_needed = sum(part.area * part.quantity for part in parts)
        total_board_area = sum(board.area * board.quantity_available for board in boards)
        
        # Check if any individual part is too large for any board
        max_board_width = max(board.width for board in boards)
        max_board_height = max(board.height for board in boards)
        
        for part in parts:
            if (part.width > max_board_width - 2 * self.margin_mm or 
                part.height > max_board_height - 2 * self.margin_mm):
                return self._create_failure_result(f"Part {part.id} ({part.width}x{part.height}mm) is too large for any board")
        
        if total_area_needed > total_board_area:
            return self._create_failure_result(f"Insufficient board area: need {total_area_needed:,.0f}mm², have {total_board_area:,.0f}mm²")
        
        # Sort boards by efficiency (area/cost ratio, largest first)
        sorted_boards = sorted(boards, key=lambda b: (b.area, -b.cost), reverse=True)
        
        # Smart strategy selection based on dataset size
        if total_parts <= 50:
            # Small dataset - use comprehensive approach
            strategies = [
                ("fast_single_board", self._try_fast_single_board),
                ("fast_multi_board", self._try_fast_multi_board)
            ]
        elif total_parts <= 200:
            # Medium dataset - use optimized approach
            strategies = [
                ("fast_multi_board", self._try_fast_multi_board),
                ("grid_placement", self._try_grid_placement)
            ]
        else:
            # Large dataset - use fastest approach
            strategies = [
                ("grid_placement", self._try_grid_placement),
                ("fast_multi_board", self._try_fast_multi_board)
            ]
        
        best_result = None
        best_score = 0
        
        for strategy_name, strategy_func in strategies:
            if self._is_timeout():
                self.log(f"Timeout reached, using best result so far")
                break
                
            try:
                self.log(f"Trying strategy: {strategy_name}")
                result = strategy_func(parts, sorted_boards, max_boards)
                
                if result and result.get('success', False):
                    # Calculate score: prioritize solutions that fit all parts, then utilization
                    all_parts_fitted = result.get('all_parts_fitted', False)
                    utilization = result.get('total_utilization', 0)
                    boards_used = result.get('total_boards_used', 0)
                    parts_fitted = result.get('parts_summary', {}).get('fitted_instances', 0)
                    parts_total = result.get('parts_summary', {}).get('total_instances', 0)
                    
                    # Score: all parts fitted gets 10000 points, parts fitted ratio gets 1000 points, utilization gets 100 points, fewer boards is better
                    parts_ratio = parts_fitted / max(parts_total, 1)
                    score = (10000 if all_parts_fitted else 0) + (parts_ratio * 1000) + (utilization * 100) - boards_used
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        self.log(f"New best result: {utilization:.1%} utilization, {parts_fitted}/{parts_total} parts fitted, score: {score:.0f}")
                        
                        # Early termination for perfect solutions
                        if all_parts_fitted and utilization > 0.7:
                            self.log(f"Perfect solution found, stopping early")
                            break
                else:
                    self.log(f"{strategy_name}: Failed")
                    
            except Exception as e:
                self.log(f"{strategy_name}: Error - {str(e)}")
        
        if not best_result:
            # Fallback to simple grid placement
            self.log("All strategies failed, trying simple grid fallback")
            best_result = self._try_simple_grid_fallback(parts, sorted_boards, max_boards)
        
        elapsed = time.time() - self.start_time
        self.log(f"Optimization complete in {elapsed:.1f}s")
        
        return best_result or self._create_failure_result("All strategies failed")
    
    def _is_timeout(self) -> bool:
        """Check if timeout has been reached"""
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            self.log(f"Timeout reached: {elapsed:.1f}s > {self.timeout_seconds}s")
            return True
        return False
    
    def _try_fast_single_board(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Fast single board optimization with early termination"""
        if self._is_timeout():
            return None
            
        # Use largest board
        board = boards[0]
        self.log(f"Trying single board: {board.width}x{board.height}mm")
        
        # Create all part instances
        all_instances = []
        for part in parts:
            for i in range(part.quantity):
                all_instances.append({
                    'id': f"{part.id}_{i+1}",
                    'part_id': part.id,
                    'width': part.width,
                    'height': part.height,
                    'area': part.area,
                    'rotation_allowed': part.rotation_allowed
                })
        
        # Try only 0° and 90° rotations for speed
        rotations = [0, 90] if any(p.rotation_allowed for p in parts) else [0]
        
        best_result = None
        best_utilization = 0
        
        for rotation in rotations:
            if self._is_timeout():
                break
                
            result = self._nest_parts_fast(all_instances, board, rotation)
            if result and result['utilization'] > best_utilization:
                best_utilization = result['utilization']
                best_result = result
                
                # Early termination if we fit all parts
                if result['total_parts_fitted'] == len(all_instances):
                    break
        
        if not best_result:
            return None
            
        return {
            'success': True,
            'total_boards_used': 1,
            'total_utilization': best_result['utilization'],
            'total_scrap_percentage': best_result['scrap_percentage'],
            'all_parts_fitted': best_result['total_parts_fitted'] == len(all_instances),
            'results': [best_result],
            'parts_summary': {
                'total_instances': len(all_instances),
                'fitted_instances': best_result['total_parts_fitted']
            },
            'svg_layout': best_result['svg_layout'],
            'total_cost': board.cost
        }
    
    def _try_fast_multi_board(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Fast multi-board optimization that continues until ALL parts are fitted"""
        if self._is_timeout():
            return None
            
        self.log("Trying fast multi-board optimization - will continue until all parts fitted")
        
        # Create all part instances
        all_instances = []
        for part in parts:
            for i in range(part.quantity):
                all_instances.append({
                    'id': f"{part.id}_{i+1}",
                    'part_id': part.id,
                    'width': part.width,
                    'height': part.height,
                    'area': part.area,
                    'rotation_allowed': part.rotation_allowed
                })
        
        # Sort by area (largest first) for better packing
        all_instances.sort(key=lambda x: x['area'], reverse=True)
        
        results = []
        remaining_instances = all_instances.copy()
        total_cost = 0
        board_instances_used = {}
        
        # Continue adding boards until ALL parts are fitted or we run out of boards/time
        iteration = 0
        max_iterations = max_boards * 10  # Prevent infinite loops
        
        while remaining_instances and iteration < max_iterations:
            if self._is_timeout():
                self.log("Timeout reached during multi-board optimization")
                break
                
            iteration += 1
            board_used_this_round = False
            
            self.log(f"Iteration {iteration}: {len(remaining_instances)} parts remaining")
            
            # Try each board type in order of efficiency
            for board in boards:
                if self._is_timeout() or not remaining_instances:
                    break
                    
                # Check board quantity limits
                if board_instances_used.get(board.id, 0) >= board.quantity_available:
                    continue
                    
                board_instances_used[board.id] = board_instances_used.get(board.id, 0) + 1
                self.log(f"Processing board {board.id} (instance {board_instances_used[board.id]}): {board.width}x{board.height}mm")
                
                # Try to fit parts on this board
                best_board_result = None
                best_utilization = 0
                
                # Use only 0° and 90° rotations for speed
                rotations = [0, 90] if any(p.rotation_allowed for p in parts) else [0]
                
                for rotation in rotations:
                    if self._is_timeout():
                        break
                        
                    board_result = self._nest_parts_fast(remaining_instances, board, rotation)
                    if board_result and board_result['utilization'] > best_utilization:
                        best_utilization = board_result['utilization']
                        best_board_result = board_result
                
                if best_board_result and best_board_result['nested_parts']:
                    # Remove fitted parts from remaining
                    fitted_ids = {p['id'] for p in best_board_result['nested_parts']}
                    remaining_instances = [inst for inst in remaining_instances if inst['id'] not in fitted_ids]
                    
                    results.append(best_board_result)
                    total_cost += board.cost
                    board_used_this_round = True
                    self.log(f"Board {board.id}: {len(best_board_result['nested_parts'])} parts fitted, {len(remaining_instances)} remaining")
                    
                    # If we fitted parts, try to fit more on the same board type
                    if remaining_instances and board_instances_used[board.id] < board.quantity_available:
                        continue  # Try another instance of the same board type
                    else:
                        break  # Move to next board type
                else:
                    self.log(f"Board {board.id}: No parts could be fitted")
            
            # If no boards were used this round, we can't fit more parts
            if not board_used_this_round:
                self.log(f"No more boards can be used, {len(remaining_instances)} parts remain unfitted")
                break
        
        if not results:
            return None
            
        # Calculate summary metrics
        total_parts_fitted = sum(len(r['nested_parts']) for r in results)
        total_utilization = sum(r['utilization'] for r in results) / len(results) if results else 0
        total_scrap = 1.0 - total_utilization
        all_parts_fitted = len(remaining_instances) == 0
        
        self.log(f"Multi-board result: {total_parts_fitted}/{len(all_instances)} parts fitted across {len(results)} boards")
        
        return {
            'success': True,
            'total_boards_used': len(results),
            'total_utilization': total_utilization,
            'total_scrap_percentage': total_scrap,
            'all_parts_fitted': all_parts_fitted,
            'results': results,
            'parts_summary': {
                'total_instances': len(all_instances),
                'fitted_instances': total_parts_fitted
            },
            'svg_layout': results[0]['svg_layout'] if results else '',
            'total_cost': total_cost
        }
    
    def _try_grid_placement(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Multi-board grid placement that continues until ALL parts are fitted"""
        if self._is_timeout():
            return None
            
        self.log("Using multi-board grid placement - will continue until all parts fitted")
        
        # Create all part instances
        all_instances = []
        for part in parts:
            for i in range(part.quantity):
                all_instances.append({
                    'id': f"{part.id}_{i+1}",
                    'part_id': part.id,
                    'width': part.width,
                    'height': part.height,
                    'area': part.area,
                    'rotation_allowed': part.rotation_allowed
                })
        
        # Sort by area (largest first)
        all_instances.sort(key=lambda x: x['area'], reverse=True)
        
        results = []
        remaining_instances = all_instances.copy()
        total_cost = 0
        board_instances_used = {}
        
        # Continue adding boards until ALL parts are fitted
        iteration = 0
        max_iterations = max_boards * 10  # Prevent infinite loops
        
        while remaining_instances and iteration < max_iterations:
            if self._is_timeout():
                self.log("Timeout reached during grid placement")
                break
                
            iteration += 1
            board_used_this_round = False
            
            self.log(f"Grid iteration {iteration}: {len(remaining_instances)} parts remaining")
            
            # Try each board type
            for board in boards:
                if self._is_timeout() or not remaining_instances:
                    break
                    
                # Check board quantity limits
                if board_instances_used.get(board.id, 0) >= board.quantity_available:
                    continue
                    
                board_instances_used[board.id] = board_instances_used.get(board.id, 0) + 1
                self.log(f"Processing board {board.id} (instance {board_instances_used[board.id]}): {board.width}x{board.height}mm")
                
                # Grid placement on this board
                fitted_parts = []
                current_x = self.margin_mm
                current_y = self.margin_mm
                row_height = 0
                
                for instance in remaining_instances[:]:  # Copy to avoid modification during iteration
                    if self._is_timeout():
                        break
                        
                    # Try different orientations for this part
                    orientations = self._get_part_orientations(instance, 0)  # No global rotation for grid
                    best_orientation = None
                    
                    for width, height, rotation in orientations:
                        # Check if part fits in current row with proper boundaries
                        if current_x + width <= board.width - self.margin_mm:
                            if current_y + height <= board.height - self.margin_mm:
                                # Verify part is completely within board boundaries
                                if (current_x >= self.margin_mm and current_y >= self.margin_mm and
                                    current_x + width <= board.width - self.margin_mm and
                                    current_y + height <= board.height - self.margin_mm):
                                    best_orientation = (width, height, rotation)
                                    break
                        
                        # Try next row
                        next_x = self.margin_mm
                        next_y = current_y + row_height + self.min_gap_mm
                        if next_y + height <= board.height - self.margin_mm:
                            if (next_x >= self.margin_mm and next_y >= self.margin_mm and
                                next_x + width <= board.width - self.margin_mm and
                                next_y + height <= board.height - self.margin_mm):
                                best_orientation = (width, height, rotation)
                                break
                    
                    if best_orientation:
                        width, height, rotation = best_orientation
                        
                        # Check if we need to move to next row
                        if current_x + width > board.width - self.margin_mm:
                            current_x = self.margin_mm
                            current_y += row_height + self.min_gap_mm
                            row_height = 0
                        
                        fitted_parts.append({
                            'id': instance['id'],
                            'part_id': instance['part_id'],
                            'x': current_x,
                            'y': current_y,
                            'width': width,
                            'height': height,
                            'rotation': rotation,
                            'area': instance['area'],
                            'original_width': instance['width'],
                            'original_height': instance['height']
                        })
                        
                        current_x += width + self.min_gap_mm
                        row_height = max(row_height, height)
                        
                        # Remove from remaining instances
                        remaining_instances.remove(instance)
                    else:
                        # If no orientation fits, try next part
                        continue
                
                if fitted_parts:
                    # Calculate metrics for this board
                    total_fitted_area = sum(p['area'] for p in fitted_parts)
                    utilization = min(1.0, total_fitted_area / board.area) if board.area > 0 else 0
                    scrap_percentage = max(0, 1.0 - utilization)
                    
                    result = {
                        'board': {
                            'id': board.id,
                            'width_mm': board.width,
                            'height_mm': board.height,
                            'area_sq_mm': board.area,
                            'cost_per_sheet': board.cost
                        },
                        'nested_parts': fitted_parts,
                        'utilization': utilization,
                        'scrap_percentage': scrap_percentage,
                        'total_parts_nested': len(fitted_parts),
                        'total_parts_required': len(all_instances),
                        'fitting_success': len(fitted_parts) > 0,
                        'svg_layout': self._generate_board_svg(board.width, board.height, fitted_parts),
                        'optimization_iterations': 1,
                        'best_rotation_angle': 0,
                        'all_parts_fitted': len(remaining_instances) == 0
                    }
                    
                    results.append(result)
                    total_cost += board.cost
                    board_used_this_round = True
                    self.log(f"Board {board.id}: {len(fitted_parts)} parts fitted, {len(remaining_instances)} remaining")
                    
                    # If we fitted parts, try to fit more on the same board type
                    if remaining_instances and board_instances_used[board.id] < board.quantity_available:
                        continue  # Try another instance of the same board type
                    else:
                        break  # Move to next board type
                else:
                    self.log(f"Board {board.id}: No parts could be fitted")
            
            # If no boards were used this round, we can't fit more parts
            if not board_used_this_round:
                self.log(f"No more boards can be used, {len(remaining_instances)} parts remain unfitted")
                break
        
        if not results:
            return None
            
        # Calculate summary metrics
        total_parts_fitted = sum(len(r['nested_parts']) for r in results)
        total_utilization = sum(r['utilization'] for r in results) / len(results) if results else 0
        total_scrap = 1.0 - total_utilization
        all_parts_fitted = len(remaining_instances) == 0
        
        self.log(f"Grid placement result: {total_parts_fitted}/{len(all_instances)} parts fitted across {len(results)} boards")
        
        return {
            'success': True,
            'total_boards_used': len(results),
            'total_utilization': total_utilization,
            'total_scrap_percentage': total_scrap,
            'all_parts_fitted': all_parts_fitted,
            'results': results,
            'parts_summary': {
                'total_instances': len(all_instances),
                'fitted_instances': total_parts_fitted
            },
            'svg_layout': results[0]['svg_layout'] if results else '',
            'total_cost': total_cost
        }
    
    def _nest_parts_fast(self, instances: List[Dict], board: Board, rotation: float) -> Optional[Dict]:
        """Fast nesting algorithm with intelligent rotation logic"""
        if self._is_timeout():
            return None
            
        fitted_parts = []
        used_positions = []
        
        # Sort by area (largest first) for better packing
        sorted_instances = sorted(instances, key=lambda x: x['area'], reverse=True)
        
        for instance in sorted_instances:
            if self._is_timeout():
                break
                
            # Try multiple orientations for each part
            orientations = self._get_part_orientations(instance, rotation)
            
            best_fit = None
            best_y = float('inf')  # Prefer lower positions (bottom-left rule)
            
            for width, height, part_rotation in orientations:
                # Check if part fits on board with this orientation
                if width <= board.width - 2 * self.margin_mm and height <= board.height - 2 * self.margin_mm:
                    # Find best position for this orientation
                    x, y = self._find_position_fast(width, height, board, used_positions)
                    
                    if x is not None and y is not None and y < best_y:
                        best_fit = (x, y, width, height, part_rotation)
                        best_y = y
            
            if best_fit:
                x, y, width, height, part_rotation = best_fit
                
                fitted_parts.append({
                    'id': instance['id'],
                    'part_id': instance['part_id'],
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'rotation': part_rotation,
                    'area': instance['area'],
                    'original_width': instance['width'],
                    'original_height': instance['height']
                })
                
                used_positions.append({
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                })
            else:
                # If this part doesn't fit in any orientation, continue to next
                continue
        
        if not fitted_parts:
            return None
            
        # Calculate metrics
        total_fitted_area = sum(p['area'] for p in fitted_parts)
        utilization = min(1.0, total_fitted_area / board.area) if board.area > 0 else 0
        scrap_percentage = max(0, 1.0 - utilization)
        
        return {
            'board': {
                'id': board.id,
                'width_mm': board.width,
                'height_mm': board.height,
                'area_sq_mm': board.area,
                'cost_per_sheet': board.cost
            },
            'nested_parts': fitted_parts,
            'utilization': utilization,
            'scrap_percentage': scrap_percentage,
            'total_parts_nested': len(fitted_parts),
            'total_parts_fitted': len(fitted_parts),
            'total_parts_required': len(instances),
            'fitting_success': len(fitted_parts) > 0,
            'svg_layout': self._generate_board_svg(board.width, board.height, fitted_parts),
            'optimization_iterations': 1,
            'best_rotation_angle': rotation,
            'all_parts_fitted': len(fitted_parts) == len(instances)
        }
    
    def _find_position_fast(self, width: float, height: float, board: Board, used_positions: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
        """Enhanced position search with intelligent bottom-left placement and better packing"""
        # Ensure part fits on board
        if width > board.width - 2 * self.margin_mm or height > board.height - 2 * self.margin_mm:
            return None, None
        
        # Calculate valid search area with proper boundaries
        min_x = self.margin_mm
        min_y = self.margin_mm
        max_x = board.width - width - self.margin_mm
        max_y = board.height - height - self.margin_mm
        
        # Ensure we have valid search area
        if max_x < min_x or max_y < min_y:
            return None, None
        
        # Use intelligent step size based on part size and available space
        step_size = max(5, min(width, height) // 3)  # Smaller steps for better packing
        
        # Try bottom-left positions with intelligent search
        best_position = None
        best_y = float('inf')
        
        # First pass: coarse search for quick results
        for y in range(int(min_y), int(max_y) + 1, max(1, int(step_size))):
            for x in range(int(min_x), int(max_x) + 1, max(1, int(step_size))):
                if self._can_place_part_fast(x, y, width, height, board, used_positions):
                    # Prefer lower positions (bottom-left rule)
                    if y < best_y:
                        best_position = (float(x), float(y))
                        best_y = y
                        # If we found a position at the bottom, use it immediately
                        if y <= min_y + step_size:
                            return best_position
        
        # If we found a position, return it
        if best_position:
            return best_position
        
        # Second pass: finer search for better packing
        if not self._is_timeout():
            finer_step = max(2, step_size // 3)
            for y in range(int(min_y), int(max_y) + 1, max(1, int(finer_step))):
                for x in range(int(min_x), int(max_x) + 1, max(1, int(finer_step))):
                    if self._can_place_part_fast(x, y, width, height, board, used_positions):
                        # Prefer lower positions
                        if y < best_y:
                            best_position = (float(x), float(y))
                            best_y = y
                            # Early termination for bottom positions
                            if y <= min_y + finer_step:
                                return best_position
        
        return best_position if best_position else (None, None)
    
    def _can_place_part_fast(self, x: float, y: float, width: float, height: float, board: Board, used_positions: List[Dict]) -> bool:
        """Fast collision detection ensuring parts stay within board boundaries"""
        # Check board boundaries with strict limits
        if (x < self.margin_mm or y < self.margin_mm or 
            x + width > board.width - self.margin_mm or 
            y + height > board.height - self.margin_mm):
            return False
        
        # Additional boundary check to ensure part is completely within board
        if (x + width > board.width or y + height > board.height or
            x < 0 or y < 0):
            return False
        
        # Check collision with other parts - proper gap calculation
        gap = max(self.min_gap_mm, 1.0)  # Minimum gap between parts
        for pos in used_positions:
            # Check if there's sufficient gap between rectangles
            if not (x + width <= pos['x'] - gap or 
                    x >= pos['x'] + pos['width'] + gap or 
                    y + height <= pos['y'] - gap or 
                    y >= pos['y'] + pos['height'] + gap):
                return False
        
        return True
    
    def _get_part_orientations(self, instance: Dict, global_rotation: float) -> List[Tuple[float, float, float]]:
        """Get all valid orientations for a part, prioritizing better packing"""
        orientations = []
        original_width = instance['width']
        original_height = instance['height']
        rotation_allowed = instance.get('rotation_allowed', True)
        
        # Always try original orientation first (0°)
        orientations.append((original_width, original_height, 0.0))
        
        # Prioritize 90° rotation for better packing efficiency
        if rotation_allowed and original_width != original_height:
            orientations.append((original_height, original_width, 90.0))
        
        # Apply global rotation if specified
        if global_rotation != 0 and rotation_allowed:
            for width, height, base_rotation in orientations[:]:  # Copy to avoid modification during iteration
                radians = math.radians(global_rotation)
                cos_angle = abs(math.cos(radians))
                sin_angle = abs(math.sin(radians))
                
                rotated_width = width * cos_angle + height * sin_angle
                rotated_height = width * sin_angle + height * cos_angle
                final_rotation = (base_rotation + global_rotation) % 360
                
                # Only add if the rotated dimensions are different
                if abs(rotated_width - width) > 0.1 or abs(rotated_height - height) > 0.1:
                    orientations.append((rotated_width, rotated_height, final_rotation))
        
        # Try 180° and 270° rotations if allowed (for better packing)
        if rotation_allowed:
            # 180° rotation
            orientations.append((original_width, original_height, 180.0))
            
            # 270° rotation (if different from 90°)
            if original_width != original_height:
                orientations.append((original_height, original_width, 270.0))
        
        # Remove duplicate orientations (same dimensions)
        unique_orientations = []
        seen_dimensions = set()
        
        for width, height, rotation in orientations:
            # Round to avoid floating point precision issues
            dim_key = (round(width, 1), round(height, 1))
            if dim_key not in seen_dimensions:
                unique_orientations.append((width, height, rotation))
                seen_dimensions.add(dim_key)
        
        return unique_orientations
    
    def _try_simple_grid_fallback(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Ultra-simple fallback for when all else fails"""
        if not parts or not boards:
            return None
            
        board = boards[0]  # Use largest board
        
        # Simple grid placement
        fitted_parts = []
        current_x = self.margin_mm
        current_y = self.margin_mm
        row_height = 0
        
        for part in parts:
            for i in range(part.quantity):
                # Create instance for rotation logic
                instance = {
                    'id': f"{part.id}_{i+1}",
                    'part_id': part.id,
                    'width': part.width,
                    'height': part.height,
                    'area': part.area,
                    'rotation_allowed': part.rotation_allowed
                }
                
                # Try different orientations for this part
                orientations = self._get_part_orientations(instance, 0)  # No global rotation for simple grid
                best_orientation = None
                
                for width, height, rotation in orientations:
                    # Check if part fits in current row with proper boundaries
                    if current_x + width <= board.width - self.margin_mm:
                        if current_y + height <= board.height - self.margin_mm:
                            # Verify part is completely within board boundaries
                            if (current_x >= self.margin_mm and current_y >= self.margin_mm and
                                current_x + width <= board.width - self.margin_mm and
                                current_y + height <= board.height - self.margin_mm):
                                best_orientation = (width, height, rotation)
                                break
                    
                    # Try next row
                    next_x = self.margin_mm
                    next_y = current_y + row_height + self.min_gap_mm
                    if next_y + height <= board.height - self.margin_mm:
                        if (next_x >= self.margin_mm and next_y >= self.margin_mm and
                            next_x + width <= board.width - self.margin_mm and
                            next_y + height <= board.height - self.margin_mm):
                            best_orientation = (width, height, rotation)
                            break
                
                if best_orientation:
                    width, height, rotation = best_orientation
                    
                    # Check if we need to move to next row
                    if current_x + width > board.width - self.margin_mm:
                        current_x = self.margin_mm
                        current_y += row_height + self.min_gap_mm
                        row_height = 0
                    
                    fitted_parts.append({
                        'id': instance['id'],
                        'part_id': part.id,
                        'x': current_x,
                        'y': current_y,
                        'width': width,
                        'height': height,
                        'rotation': rotation,
                        'area': part.area,
                        'original_width': part.width,
                        'original_height': part.height
                    })
                    
                    current_x += width + self.min_gap_mm
                    row_height = max(row_height, height)
                else:
                    # If no orientation fits, try next part
                    break
        
        if not fitted_parts:
            return None
            
        # Calculate metrics
        total_fitted_area = sum(p['area'] for p in fitted_parts)
        utilization = min(1.0, total_fitted_area / board.area) if board.area > 0 else 0
        scrap_percentage = max(0, 1.0 - utilization)
        
        result = {
            'board': {
                'id': board.id,
                'width_mm': board.width,
                'height_mm': board.height,
                'area_sq_mm': board.area,
                'cost_per_sheet': board.cost
            },
            'nested_parts': fitted_parts,
            'utilization': utilization,
            'scrap_percentage': scrap_percentage,
            'total_parts_nested': len(fitted_parts),
            'total_parts_required': sum(p.quantity for p in parts),
            'fitting_success': len(fitted_parts) > 0,
            'svg_layout': self._generate_board_svg(board.width, board.height, fitted_parts),
            'optimization_iterations': 1,
            'best_rotation_angle': 0,
            'all_parts_fitted': len(fitted_parts) == sum(p.quantity for p in parts)
        }
        
        return {
            'success': True,
            'total_boards_used': 1,
            'total_utilization': utilization,
            'total_scrap_percentage': scrap_percentage,
            'all_parts_fitted': len(fitted_parts) == sum(p.quantity for p in parts),
            'results': [result],
            'parts_summary': {
                'total_instances': sum(p.quantity for p in parts),
                'fitted_instances': len(fitted_parts)
            },
            'svg_layout': result['svg_layout'],
            'total_cost': board.cost
        }
    
    def _validate_part_positions(self, fitted_parts: List[Dict], board_width: float, board_height: float) -> bool:
        """Validate that all parts are properly positioned within board boundaries"""
        margin = self.margin_mm
        
        for part in fitted_parts:
            x = part['x']
            y = part['y']
            width = part['width']
            height = part['height']
            
            # Check if part is within board boundaries
            if (x < margin or y < margin or 
                x + width > board_width - margin or 
                y + height > board_height - margin):
                self.log(f"Boundary violation: Part {part['id']} at ({x:.1f}, {y:.1f}) size {width}x{height} is outside board boundaries!")
                return False
        
        return True
    
    def _generate_board_svg(self, board_width: float, board_height: float, fitted_parts: List[Dict]) -> str:
        """Generate simplified SVG layout for the board with boundary validation"""
        # Validate part positions before generating SVG
        if not self._validate_part_positions(fitted_parts, board_width, board_height):
            self.log("Warning: Some parts are outside board boundaries!")
        
        # Scale factor for display
        scale = min(800 / board_width, 600 / board_height)
        display_width = board_width * scale
        display_height = board_height * scale
        
        svg_parts = []
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd']
        
        for i, part in enumerate(fitted_parts):
            x = part['x'] * scale
            y = part['y'] * scale
            width = part['width'] * scale
            height = part['height'] * scale
            rotation = part.get('rotation', 0)
            color = colors[i % len(colors)]
            
            # Handle rotation in SVG - show original dimensions when rotated
            if rotation != 0:
                # Use stored original dimensions for proper visual representation
                original_width = part.get('original_width', part['width']) * scale
                original_height = part.get('original_height', part['height']) * scale
                
                # Calculate rotation center (center of the bounding box)
                center_x = x + width / 2
                center_y = y + height / 2
                
                svg_parts.append(f'''
                    <g transform="translate({center_x:.1f}, {center_y:.1f}) rotate({rotation}) translate({-original_width/2:.1f}, {-original_height/2:.1f})">
                        <rect x="0" y="0" width="{original_width:.1f}" height="{original_height:.1f}" 
                              fill="{color}" stroke="#333" stroke-width="1" opacity="0.8">
                            <title>Part {part['part_id']} - {part.get('original_width', part['width']):.0f}x{part.get('original_height', part['height']):.0f}mm (rotated {rotation}°)</title>
                        </rect>
                    </g>
                ''')
            else:
                svg_parts.append(f'''
                    <rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" 
                          fill="{color}" stroke="#333" stroke-width="1" opacity="0.8">
                        <title>Part {part['part_id']} - {part['width']:.0f}x{part['height']:.0f}mm</title>
                    </rect>
                ''')
        
        return f'''
        <svg width="{display_width:.0f}" height="{display_height:.0f}" 
             viewBox="0 0 {display_width:.0f} {display_height:.0f}" 
             style="border: 2px solid #333; background: #f8f9fa;">
            <rect x="0" y="0" width="{display_width:.0f}" height="{display_height:.0f}" 
                  fill="white" stroke="#ddd" stroke-width="1"/>
            {''.join(svg_parts)}
            <text x="10" y="20" font-family="Arial" font-size="14" fill="#333">
                Board: {board_width:.0f} × {board_height:.0f}mm
            </text>
            <text x="10" y="40" font-family="Arial" font-size="12" fill="#666">
                Parts: {len(fitted_parts)} fitted
            </text>
        </svg>
        '''
    
    def _create_failure_result(self, error_message: str) -> Dict:
        """Create failure result"""
        return {
            'success': False,
            'error': error_message,
            'total_boards_used': 0,
            'total_utilization': 0,
            'total_scrap_percentage': 1.0,
            'all_parts_fitted': False,
            'results': [],
            'parts_summary': {'total_instances': 0, 'fitted_instances': 0},
            'svg_layout': '',
            'total_cost': 0
        }

def test_fast_nesting_engine():
    """Test the fast nesting engine"""
    print("Testing Fast Nesting Engine")
    print("=" * 50)
    
    # Create test parts
    parts = [
        Part(id="1", width=200, height=100, area=20000, quantity=10, svg_path="M 0,0 L 200,0 L 200,100 L 0,100 Z"),
        Part(id="2", width=150, height=80, area=12000, quantity=8, svg_path="M 0,0 L 150,0 L 150,80 L 0,80 Z"),
        Part(id="3", width=100, height=50, area=5000, quantity=15, svg_path="M 0,0 L 100,0 L 100,50 L 0,50 Z"),
        Part(id="4", width=80, height=40, area=3200, quantity=20, svg_path="M 0,0 L 80,0 L 80,40 L 0,40 Z")
    ]
    
    # Create test boards
    boards = [
        Board(id="1", width=1000, height=500, area=500000, cost=100.0, quantity_available=10),
        Board(id="2", width=800, height=400, area=320000, cost=80.0, quantity_available=10),
        Board(id="3", width=600, height=300, area=180000, cost=60.0, quantity_available=10)
    ]
    
    print(f"Test Setup:")
    print(f"  - Parts: {len(parts)} types, {sum(p.quantity for p in parts)} total instances")
    print(f"  - Boards: {len(boards)} types available")
    
    # Test fast engine
    engine = FastNestingEngine(margin_mm=10.0, min_gap_mm=5.0, timeout_seconds=10)
    engine.debug = True
    
    print(f"\nRunning fast nesting optimization...")
    start_time = time.time()
    result = engine.optimize_nesting(parts=parts, boards=boards, max_boards=5)
    elapsed = time.time() - start_time
    
    print(f"\nResults (completed in {elapsed:.1f}s):")
    print(f"  - Success: {'YES' if result['success'] else 'NO'}")
    print(f"  - Boards used: {result['total_boards_used']}")
    print(f"  - Total scrap: {result['total_scrap_percentage']:.1%}")
    print(f"  - Total utilization: {result['total_utilization']:.1%}")
    print(f"  - Total cost: ${result['total_cost']:.2f}")
    print(f"  - Parts fitted: {result['parts_summary']['fitted_instances']}/{result['parts_summary']['total_instances']}")
    print(f"  - All parts fitted: {'YES' if result['all_parts_fitted'] else 'NO'}")
    
    return result

if __name__ == "__main__":
    test_fast_nesting_engine()
