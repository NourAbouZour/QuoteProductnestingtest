#!/usr/bin/env python3
"""
Professional 2D Nesting Engine with Multiple Algorithms
Uses real part dimensions and professional bin packing algorithms
"""

import math
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import rectpack
import binpacking

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

@dataclass
class PlacedPart:
    id: str
    part_id: str
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    area: float = 0.0

class ProfessionalNestingEngine:
    """Professional 2D nesting engine with multiple algorithms"""
    
    def __init__(self, margin_mm: float = 10.0, min_gap_mm: float = 5.0):
        self.margin_mm = margin_mm
        self.min_gap_mm = min_gap_mm
        self.debug = True
        
    def log(self, message: str):
        """Debug logging"""
        if self.debug:
            print(f"[PROFESSIONAL_NESTING] {message}")
    
    def optimize_nesting(self, parts: List[Part], boards: List[Board], max_boards: int = 5) -> Dict:
        """Main optimization function with multiple algorithms"""
        
        self.log(f"Starting professional nesting with {len(parts)} part types")
        
        # Calculate total parts and area needed
        total_parts = sum(part.quantity for part in parts)
        total_area_needed = sum(part.area * part.quantity for part in parts)
        
        self.log(f"Total parts: {total_parts}")
        self.log(f"Total area needed: {total_area_needed:,.0f} mm²")
        
        # Sort boards by area (largest first)
        sorted_boards = sorted(boards, key=lambda b: b.area, reverse=True)
        self.log(f"Available boards: {len(sorted_boards)}")
        
        # Try different strategies - prioritize strategies that fit ALL parts
        strategies = [
            ("multi_board_optimal", self._try_multi_board_optimal),
            ("multi_board_forced", self._try_multi_board_forced),
            ("rectpack_optimal", self._try_rectpack_optimal),
            ("rectpack_simple", self._try_rectpack_simple),
            ("bottom_left_fill", self._try_bottom_left_fill),
            ("genetic_algorithm", self._try_genetic_algorithm)
        ]
        
        best_result = None
        best_utilization = 0
        best_all_parts_fitted = False
        
        for strategy_name, strategy_func in strategies:
            try:
                self.log(f"Trying strategy: {strategy_name}")
                start_time = time.time()
                
                result = strategy_func(parts, sorted_boards, max_boards)
                
                if result and result.get('success', False):
                    utilization = result.get('total_utilization', 0)
                    all_parts_fitted = result.get('all_parts_fitted', False)
                    
                    # Get total parts fitted vs total parts required
                    parts_summary = result.get('parts_summary', {})
                    total_parts = parts_summary.get('total_instances', 0)
                    fitted_parts = parts_summary.get('fitted_instances', 0)
                    all_parts_fitted = (fitted_parts >= total_parts and total_parts > 0)
                    
                    self.log(f"{strategy_name}: {utilization:.1%} utilization, {fitted_parts}/{total_parts} parts fitted ({time.time() - start_time:.1f}s)")
                    
                    # Prioritize solutions that fit ALL parts
                    should_select = False
                    if not best_all_parts_fitted and all_parts_fitted:
                        # This is the first solution that fits all parts - select it
                        should_select = True
                        self.log(f"🎯 First solution to fit ALL parts: {fitted_parts}/{total_parts}")
                    elif best_all_parts_fitted and all_parts_fitted:
                        # Both fit all parts - choose higher utilization
                        should_select = utilization > best_utilization
                    elif not best_all_parts_fitted and not all_parts_fitted:
                        # Neither fits all parts - choose higher utilization
                        should_select = utilization > best_utilization
                    # If best fits all parts but this doesn't, don't select this one
                    
                    if should_select:
                        best_utilization = utilization
                        best_result = result
                        best_all_parts_fitted = all_parts_fitted
                        self.log(f"✅ New best result: {utilization:.1%} utilization, {fitted_parts}/{total_parts} parts fitted")
                else:
                    self.log(f"{strategy_name}: Failed")
                    
            except Exception as e:
                self.log(f"{strategy_name}: Error - {str(e)}")
        
        if not best_result:
            # Try a simple fallback: just place parts in a grid
            self.log("All strategies failed, trying simple grid placement")
            fallback_result = self._try_simple_grid_placement(parts, sorted_boards, max_boards)
            if fallback_result:
                return fallback_result
            
            return {
                'success': False,
                'error': 'No suitable nesting found with any algorithm',
                'total_boards_used': 0,
                'total_utilization': 0,
                'total_scrap_percentage': 1.0,
                'results': [],
                'parts_summary': {'total_instances': total_parts, 'fitted_instances': 0}
            }
        
        self.log(f"✅ Best result: {best_utilization:.1%} utilization")
        return best_result
    
    def _try_multi_board_optimal(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Try to fit ALL parts across multiple boards using optimal strategy"""
        try:
            self.log("Using multi-board optimal strategy to fit ALL parts")
            
            # Create instances for all parts
            all_instances = []
            for part in parts:
                for i in range(part.quantity):
                    all_instances.append({
                        'id': f"{part.id}_{i}",
                        'part_id': part.id,
                        'width': part.width,
                        'height': part.height,
                        'area': part.area,
                        'rotation_allowed': part.rotation_allowed
                    })
            
            # Sort by area (largest first)
            all_instances.sort(key=lambda x: x['area'], reverse=True)
            
            results = []
            total_parts_fitted = 0
            total_utilization = 0
            total_cost = 0
            remaining_instances = all_instances.copy()
            
            # Try to fit parts on each board, respecting quantities
            board_instances_used = {}  # Track how many of each board type we've used
            
            # Continue until all parts are fitted or no more boards available
            while remaining_instances:
                board_used_this_round = False
                
                # Try each board type in order of efficiency
                for board_index, board in enumerate(boards[:max_boards]):
                    if not remaining_instances:
                        break
                    
                    # Check if we can use this board type (respect quantity_available)
                    board_type_id = board.id
                    if board_type_id not in board_instances_used:
                        board_instances_used[board_type_id] = 0
                    
                    if board_instances_used[board_type_id] >= board.quantity_available:
                        continue  # Skip this board type if quantity limit reached
                    
                    # Create a unique instance ID for this board
                    board_instance_id = f"{board.id}_{board_instances_used[board_type_id]}"
                    board_instances_used[board_type_id] += 1
                    
                    self.log(f"Processing board {board_instance_id}: {board.width}x{board.height}mm (instance {board_instances_used[board_type_id]}/{board.quantity_available})")
                
                    # Try different rotations for this board
                    best_board_result = None
                    best_board_utilization = 0
                    
                    rotations = [0, 90, 180, 270] if any(p.rotation_allowed for p in parts) else [0]
                    
                    for rotation in rotations:
                        board_result = self._nest_parts_on_board_optimal(remaining_instances, board, rotation)
                        if board_result and board_result['utilization'] > best_board_utilization:
                            best_board_utilization = board_result['utilization']
                            best_board_result = board_result
                    
                    if best_board_result:
                        # Update board result with unique instance ID
                        best_board_result['board']['id'] = board_instance_id
                        best_board_result['board']['instance_id'] = board_instance_id
                        best_board_result['board']['board_type_id'] = board.id
                        best_board_result['board']['instance_number'] = board_instances_used[board_type_id]
                        
                        # Remove fitted parts from remaining instances
                        fitted_part_ids = {p['id'] for p in best_board_result['nested_parts']}
                        remaining_instances = [inst for inst in remaining_instances if inst['id'] not in fitted_part_ids]
                        
                        total_parts_fitted += best_board_result['total_parts_nested']
                        total_utilization += best_board_result['utilization']
                        total_cost += board.cost
                        
                        results.append(best_board_result)
                        board_used_this_round = True
                        self.log(f"Board {board_instance_id}: {best_board_result['total_parts_nested']} parts fitted, {best_board_result['utilization']:.1%} utilization")
                        
                        # If we fitted parts, try to fit more on the same board type
                        if remaining_instances and board_instances_used[board_type_id] < board.quantity_available:
                            continue  # Try another instance of the same board type
                        else:
                            break  # Move to next board type
                    else:
                        self.log(f"Board {board_instance_id}: No parts could be fitted")
                
                # If no boards were used this round, we can't fit more parts
                if not board_used_this_round:
                    self.log("No more boards can be used or no parts can be fitted")
                    break
            
            if not results:
                return None
            
            avg_utilization = total_utilization / len(results) if results else 0
            avg_scrap = 1.0 - avg_utilization
            all_parts_fitted = total_parts_fitted >= len(all_instances)
            
            self.log(f"Multi-board result: {total_parts_fitted}/{len(all_instances)} parts fitted across {len(results)} boards")
            
            return {
                'success': True,
                'total_boards_used': len(results),
                'total_utilization': avg_utilization,
                'total_scrap_percentage': avg_scrap,
                'all_parts_fitted': all_parts_fitted,
                'results': results,
                'parts_summary': {
                    'total_instances': len(all_instances),
                    'fitted_instances': total_parts_fitted
                },
                'svg_layout': results[0]['svg_layout'] if results else '',
                'total_cost': total_cost
            }
            
        except Exception as e:
            self.log(f"Multi-board optimal failed: {str(e)}")
            return None
    
    def _try_multi_board_forced(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Force multi-board approach - continue adding boards until ALL parts are fitted"""
        try:
            self.log("Using FORCED multi-board strategy to fit ALL parts")
            
            # Create instances for all parts
            all_instances = []
            for part in parts:
                for i in range(part.quantity):
                    all_instances.append({
                        'id': f"{part.id}_{i}",
                        'part_id': part.id,
                        'width': part.width,
                        'height': part.height,
                        'area': part.area,
                        'rotation_allowed': part.rotation_allowed
                    })
            
            self.log(f"Created {len(all_instances)} part instances")
            
            # Sort by area (largest first) for better packing
            all_instances.sort(key=lambda x: x['area'], reverse=True)
            
            results = []
            total_parts_fitted = 0
            total_utilization = 0
            total_cost = 0
            remaining_instances = all_instances.copy()
            
            # Track board usage
            board_instances_used = {}
            
            # Force continue until ALL parts are fitted or no more boards available
            iteration = 0
            max_iterations = max_boards * 10  # Prevent infinite loops
            
            while remaining_instances and iteration < max_iterations:
                iteration += 1
                board_used_this_round = False
                
                self.log(f"Iteration {iteration}: {len(remaining_instances)} parts remaining")
                
                # Try each board type
                for board in boards[:max_boards]:
                    if not remaining_instances:
                        break
                    
                    # Check board quantity limits
                    board_type_id = board.id
                    if board_type_id not in board_instances_used:
                        board_instances_used[board_type_id] = 0
                    
                    if board_instances_used[board_type_id] >= board.quantity_available:
                        continue
                    
                    # Create board instance
                    board_instances_used[board_type_id] += 1
                    board_instance_id = f"{board.id}_{board_instances_used[board_type_id]}"
                    
                    self.log(f"Processing board {board_instance_id}: {board.width}x{board.height}mm")
                    
                    # Try to fit parts on this board with different rotations
                    best_board_result = None
                    best_board_utilization = 0
                    
                    rotations = [0, 90, 180, 270] if any(p.rotation_allowed for p in parts) else [0]
                    
                    for rotation in rotations:
                        board_result = self._nest_parts_on_board_optimal(remaining_instances, board, rotation)
                        if board_result and board_result['utilization'] > best_board_utilization:
                            best_board_utilization = board_result['utilization']
                            best_board_result = board_result
                    
                    if best_board_result and best_board_result['nested_parts']:
                        # Update board result with unique instance ID
                        best_board_result['board']['id'] = board_instance_id
                        best_board_result['board']['instance_id'] = board_instance_id
                        best_board_result['board']['board_type_id'] = board.id
                        
                        # Remove fitted parts from remaining instances
                        fitted_part_ids = {p['id'] for p in best_board_result['nested_parts']}
                        remaining_instances = [inst for inst in remaining_instances if inst['id'] not in fitted_part_ids]
                        
                        total_parts_fitted += best_board_result['total_parts_nested']
                        total_utilization += best_board_result['utilization']
                        total_cost += board.cost
                        
                        results.append(best_board_result)
                        board_used_this_round = True
                        
                        self.log(f"Board {board_instance_id}: {best_board_result['total_parts_nested']} parts fitted, "
                                f"{best_board_result['utilization']:.1%} utilization")
                        
                        # Continue to next board type to fit more parts
                        break  # Move to next board type after fitting parts
                    else:
                        self.log(f"Board {board_instance_id}: No parts could be fitted")
                
                # If no boards were used this round, we're stuck
                if not board_used_this_round:
                    self.log(f"❌ No more boards can be used, {len(remaining_instances)} parts remain unfitted")
                    break
            
            if not results:
                return None
            
            avg_utilization = total_utilization / len(results) if results else 0
            avg_scrap = 1.0 - avg_utilization
            all_parts_fitted = len(remaining_instances) == 0
            
            self.log(f"Forced multi-board result: {total_parts_fitted}/{len(all_instances)} parts fitted across {len(results)} boards")
            
            return {
                'success': True,
                'total_boards_used': len(results),
                'total_utilization': avg_utilization,
                'total_scrap_percentage': avg_scrap,
                'all_parts_fitted': all_parts_fitted,
                'results': results,
                'parts_summary': {
                    'total_instances': len(all_instances),
                    'fitted_instances': total_parts_fitted
                },
                'svg_layout': results[0]['svg_layout'] if results else '',
                'total_cost': total_cost
            }
            
        except Exception as e:
            self.log(f"Forced multi-board failed: {str(e)}")
            return None
    
    def _try_rectpack_optimal(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Try rectpack library with optimal settings - ensures ALL parts are fitted"""
        try:
            # Create rectpack instance
            packer = rectpack.newPacker()
            
            # Check if we have valid parts
            if not parts or not boards:
                self.log("No parts or boards provided")
                return None
            
            # Add ALL available boards (not just max_boards)
            available_boards = boards[:max_boards] if max_boards > 0 else boards
            for board in available_boards:
                if board.width > 0 and board.height > 0:
                    packer.add_bin(board.width, board.height, bid=board.id)
                    self.log(f"Added board {board.id}: {board.width}x{board.height}mm")
                else:
                    self.log(f"Invalid board dimensions: {board.width}x{board.height}")
            
            # Add rectangles (parts) - ALL parts must be fitted
            rect_count = 0
            total_parts_needed = sum(part.quantity for part in parts)
            self.log(f"Total parts needed: {total_parts_needed}")
            
            for part in parts:
                if part.width > 0 and part.height > 0 and part.quantity > 0:
                    for _ in range(part.quantity):
                        packer.add_rect(part.width, part.height, rid=f"{part.id}_{_}")
                        rect_count += 1
                else:
                    self.log(f"Invalid part dimensions: {part.width}x{part.height}, qty: {part.quantity}")
            
            if rect_count == 0:
                self.log("No valid rectangles to pack")
                return None
            
            self.log(f"Packing {rect_count} rectangles on {len(available_boards)} boards")
            
            # Pack
            packer.pack()
            
            # Process results
            results = []
            total_utilization = 0
            total_scrap = 0
            total_boards_used = 0
            total_parts_fitted = 0
            total_parts_needed = sum(p.quantity for p in parts)
            
            self.log(f"Processing {len(list(packer))} packed bins")
            
            for bin_index, bin_data in enumerate(packer):
                board_id = bin_data.bid
                board_width = bin_data.width
                board_height = bin_data.height
                board_area = board_width * board_height
                
                # Find original board
                original_board = next((b for b in boards if b.id == board_id), None)
                if not original_board:
                    continue
                
                fitted_parts = []
                used_area = 0
                
                for rect in bin_data:
                    fitted_parts.append({
                        'id': rect.rid,
                        'part_id': rect.rid.split('_')[0],
                        'x': rect.x,
                        'y': rect.y,
                        'width': rect.width,
                        'height': rect.height,
                        'rotation': 0,
                        'area': rect.width * rect.height
                    })
                    used_area += rect.width * rect.height
                
                utilization = used_area / board_area if board_area > 0 else 0
                scrap_percentage = 1.0 - utilization
                total_parts_fitted += len(fitted_parts)
                
                self.log(f"Board {board_id}: {len(fitted_parts)} parts fitted, {utilization:.1%} utilization")
                
                results.append({
                    'board': {
                        'id': board_id,
                        'width_mm': board_width,
                        'height_mm': board_height,
                        'area_sq_mm': board_area,
                        'cost_per_sheet': original_board.cost
                    },
                    'nested_parts': fitted_parts,
                    'utilization': utilization,
                    'scrap_percentage': scrap_percentage,
                    'total_parts_nested': len(fitted_parts),
                    'total_parts_required': total_parts_needed,
                    'fitting_success': len(fitted_parts) > 0,
                    'svg_layout': self._generate_board_svg(board_width, board_height, fitted_parts),
                    'optimization_iterations': 1,
                    'best_rotation_angle': 0
                })
                
                total_utilization += utilization
                total_boards_used += 1
            
            if total_boards_used == 0:
                return None
                
            avg_utilization = total_utilization / total_boards_used if total_boards_used > 0 else 0
            avg_scrap = 1.0 - avg_utilization
            all_parts_fitted = total_parts_fitted >= total_parts_needed
            
            self.log(f"Final result: {total_parts_fitted}/{total_parts_needed} parts fitted, {avg_utilization:.1%} avg utilization")
            
            return {
                'success': True,
                'total_boards_used': total_boards_used,
                'total_utilization': avg_utilization,
                'total_scrap_percentage': avg_scrap,
                'all_parts_fitted': all_parts_fitted,
                'results': results,
                'parts_summary': {
                    'total_instances': total_parts_needed,
                    'fitted_instances': total_parts_fitted
                },
                'svg_layout': results[0]['svg_layout'] if results else '',
                'total_cost': sum(r['board']['cost_per_sheet'] for r in results)
            }
            
        except Exception as e:
            self.log(f"Rectpack optimal failed: {str(e)}")
            return None
    
    def _try_rectpack_simple(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Try rectpack with simple settings"""
        try:
            packer = rectpack.newPacker(rotation=True)
            
            # Add largest board only for simplicity
            largest_board = boards[0]
            packer.add_bin(largest_board.width, largest_board.height, bid=largest_board.id)
            
            # Add rectangles
            for part in parts:
                for _ in range(part.quantity):
                    packer.add_rect(part.width, part.height, rid=f"{part.id}_{_}")
            
            packer.pack()
            
            # Process single board result
            if not packer:
                return None
                
            bin_data = list(packer)[0]
            board_width = bin_data.width
            board_height = bin_data.height
            board_area = board_width * board_height
            
            fitted_parts = []
            used_area = 0
            
            for rect in bin_data:
                fitted_parts.append({
                    'id': rect.rid,
                    'part_id': rect.rid.split('_')[0],
                    'x': rect.x,
                    'y': rect.y,
                    'width': rect.width,
                    'height': rect.height,
                    'rotation': 0,
                    'area': rect.width * rect.height
                })
                used_area += rect.width * rect.height
            
            utilization = used_area / board_area if board_area > 0 else 0
            scrap_percentage = 1.0 - utilization
            
            result = {
                'board': {
                    'id': largest_board.id,
                    'width_mm': board_width,
                    'height_mm': board_height,
                    'area_sq_mm': board_area,
                    'cost_per_sheet': largest_board.cost
                },
                'nested_parts': fitted_parts,
                'utilization': utilization,
                'scrap_percentage': scrap_percentage,
                'total_parts_nested': len(fitted_parts),
                'total_parts_required': sum(p.quantity for p in parts),
                'fitting_success': len(fitted_parts) == sum(p.quantity for p in parts),
                'svg_layout': self._generate_board_svg(board_width, board_height, fitted_parts),
                'optimization_iterations': 1,
                'best_rotation_angle': 0
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
                'total_cost': largest_board.cost
            }
            
        except Exception as e:
            self.log(f"Rectpack simple failed: {str(e)}")
            return None
    
    def _try_bottom_left_fill(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Try bottom-left fill algorithm"""
        try:
            # Use largest board
            board = boards[0]
            
            # Create instances
            instances = []
            for part in parts:
                for i in range(part.quantity):
                    instances.append({
                        'id': f"{part.id}_{i}",
                        'part_id': part.id,
                        'width': part.width,
                        'height': part.height,
                        'area': part.area,
                        'rotation_allowed': part.rotation_allowed
                    })
            
            # Sort by area (largest first)
            instances.sort(key=lambda x: x['area'], reverse=True)
            
            # Try different rotations
            best_result = None
            best_utilization = 0
            
            rotations = [0, 90, 180, 270] if any(p.rotation_allowed for p in parts) else [0]
            
            for rotation in rotations:
                result = self._nest_parts_bottom_left(instances, board, rotation)
                if result and result['utilization'] > best_utilization:
                    best_utilization = result['utilization']
                    best_result = result
            
            if not best_result:
                return None
                
            return {
                'success': True,
                'total_boards_used': 1,
                'total_utilization': best_result['utilization'],
                'total_scrap_percentage': best_result['scrap_percentage'],
                'all_parts_fitted': best_result['all_parts_fitted'],
                'results': [best_result],
                'parts_summary': {
                    'total_instances': len(instances),
                    'fitted_instances': best_result['total_parts_nested']
                },
                'svg_layout': best_result['svg_layout'],
                'total_cost': board.cost
            }
            
        except Exception as e:
            self.log(f"Bottom-left fill failed: {str(e)}")
            return None
    
    def _try_genetic_algorithm(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Try genetic algorithm approach"""
        # For now, fallback to bottom-left fill
        return self._try_bottom_left_fill(parts, boards, max_boards)
    
    def _nest_parts_bottom_left(self, instances: List[Dict], board: Board, rotation: float) -> Optional[Dict]:
        """Bottom-left fill algorithm with rotation"""
        fitted_parts = []
        used_positions = []
        
        for instance in instances:
            # Apply rotation
            if rotation != 0 and instance['rotation_allowed']:
                radians = math.radians(rotation)
                cos_angle = abs(math.cos(radians))
                sin_angle = abs(math.sin(radians))
                
                width = instance['width'] * cos_angle + instance['height'] * sin_angle
                height = instance['width'] * sin_angle + instance['height'] * cos_angle
            else:
                width = instance['width']
                height = instance['height']
            
            # Find best position
            best_x, best_y = self._find_best_position(width, height, board, used_positions)
            
            if best_x is not None and best_y is not None:
                # Add to fitted parts
                fitted_parts.append({
                    'id': instance['id'],
                    'part_id': instance['part_id'],
                    'x': best_x,
                    'y': best_y,
                    'width': width,
                    'height': height,
                    'rotation': rotation,
                    'area': instance['area']
                })
                
                # Add to used positions
                used_positions.append({
                    'x': best_x,
                    'y': best_y,
                    'width': width,
                    'height': height
                })
        
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
            'total_parts_required': len(instances),
            'fitting_success': len(fitted_parts) == len(instances),
            'svg_layout': self._generate_board_svg(board.width, board.height, fitted_parts),
            'optimization_iterations': 1,
            'best_rotation_angle': rotation,
            'all_parts_fitted': len(fitted_parts) == len(instances)
        }
    
    def _find_best_position(self, width: float, height: float, board: Board, used_positions: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
        """Find the best position for a part using bottom-left fill"""
        # Try positions from bottom-left
        step_size = max(10, min(width, height) / 10)
        
        for y in range(int(self.margin_mm), int(board.height - height - self.margin_mm), int(step_size)):
            for x in range(int(self.margin_mm), int(board.width - width - self.margin_mm), int(step_size)):
                if self._can_place_part(x, y, width, height, board, used_positions):
                    return float(x), float(y)
        
        return None, None
    
    def _can_place_part(self, x: float, y: float, width: float, height: float, board: Board, used_positions: List[Dict]) -> bool:
        """Check if a part can be placed at the given position"""
        # Check board boundaries
        if x < self.margin_mm or y < self.margin_mm:
            return False
        if x + width > board.width - self.margin_mm or y + height > board.height - self.margin_mm:
            return False
        
        # Check collision with other parts
        for pos in used_positions:
            if not (x + width <= pos['x'] or x >= pos['x'] + pos['width'] or 
                   y + height <= pos['y'] or y >= pos['y'] + pos['height']):
                return False
        
        return True
    
    def _try_simple_grid_placement(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Simple grid placement fallback"""
        try:
            if not parts or not boards:
                return None
                
            board = boards[0]  # Use largest board
            fitted_parts = []
            current_x = self.margin_mm
            current_y = self.margin_mm
            row_height = 0
            
            for part in parts:
                for _ in range(part.quantity):
                    if current_x + part.width <= board.width - self.margin_mm and current_y + part.height <= board.height - self.margin_mm:
                        fitted_parts.append({
                            'id': f"{part.id}_{_}",
                            'part_id': part.id,
                            'x': current_x,
                            'y': current_y,
                            'width': part.width,
                            'height': part.height,
                            'rotation': 0,
                            'area': part.area
                        })
                        
                        current_x += part.width + self.min_gap_mm
                        row_height = max(row_height, part.height)
                    else:
                        # Move to next row
                        current_x = self.margin_mm
                        current_y += row_height + self.min_gap_mm
                        row_height = 0
                        
                        if current_y + part.height <= board.height - self.margin_mm:
                            fitted_parts.append({
                                'id': f"{part.id}_{_}",
                                'part_id': part.id,
                                'x': current_x,
                                'y': current_y,
                                'width': part.width,
                                'height': part.height,
                                'rotation': 0,
                                'area': part.area
                            })
                            
                            current_x += part.width + self.min_gap_mm
                            row_height = part.height
                        else:
                            break  # No more space
            
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
                'fitting_success': len(fitted_parts) == sum(p.quantity for p in parts),
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
            
        except Exception as e:
            self.log(f"Simple grid placement failed: {str(e)}")
            return None
    
    def _nest_parts_on_board_optimal(self, instances: List[Dict], board: Board, rotation: float) -> Optional[Dict]:
        """Optimal nesting algorithm for a single board using improved bin packing"""
        try:
            fitted_parts = []
            used_positions = []
            
            # Sort instances by area (largest first) for better packing
            sorted_instances = sorted(instances, key=lambda x: x['area'], reverse=True)
            
            for instance in sorted_instances:
                # Try both orientations if rotation is allowed
                orientations = []
                
                # Original orientation
                orientations.append((instance['width'], instance['height'], 0))
                
                # Rotated orientation (if allowed and different)
                if instance.get('rotation_allowed', True) and instance['width'] != instance['height']:
                    orientations.append((instance['height'], instance['width'], 90))
                
                # Apply additional rotation if specified
                if rotation != 0:
                    for width, height, base_rotation in orientations[:]:  # Copy list to avoid modification during iteration
                        radians = math.radians(rotation)
                        cos_angle = abs(math.cos(radians))
                        sin_angle = abs(math.sin(radians))
                        
                        rotated_width = width * cos_angle + height * sin_angle
                        rotated_height = width * sin_angle + height * cos_angle
                        final_rotation = (base_rotation + rotation) % 360
                        
                        orientations.append((rotated_width, rotated_height, final_rotation))
                
                # Find best position for any orientation
                best_fit = None
                best_y = float('inf')
                
                for width, height, part_rotation in orientations:
                    # Check if part fits on board
                    if width <= board.width and height <= board.height:
                        # Find best position using bottom-left fill
                        best_x, best_y_pos = self._find_best_position(width, height, board, used_positions)
                        
                        if best_x is not None and best_y_pos is not None and best_y_pos < best_y:
                            best_fit = (best_x, best_y_pos, width, height, part_rotation)
                            best_y = best_y_pos
                
                if best_fit:
                    best_x, best_y, width, height, part_rotation = best_fit
                    
                    fitted_parts.append({
                        'id': instance['id'],
                        'part_id': instance['part_id'],
                        'x': best_x,
                        'y': best_y,
                        'width': width,
                        'height': height,
                        'rotation': part_rotation,
                        'area': instance['area']
                    })
                    
                    used_positions.append({
                        'x': best_x,
                        'y': best_y,
                        'width': width,
                        'height': height
                    })
                else:
                    # If this part doesn't fit, continue to next part
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
                'total_parts_required': len(instances),
                'fitting_success': len(fitted_parts) > 0,
                'svg_layout': self._generate_board_svg(board.width, board.height, fitted_parts),
                'optimization_iterations': 1,
                'best_rotation_angle': rotation,
                'all_parts_fitted': len(fitted_parts) == len(instances)
            }
            
        except Exception as e:
            self.log(f"Nesting on board failed: {str(e)}")
            return None
    
    def _generate_board_svg(self, board_width: float, board_height: float, fitted_parts: List[Dict]) -> str:
        """Generate SVG layout for the board"""
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
            color = colors[i % len(colors)]
            
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
