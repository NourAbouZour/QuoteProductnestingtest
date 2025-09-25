#!/usr/bin/env python3
"""
Multi-Board Nesting Optimizer
Fits ALL parts using multiple boards with minimal scrap percentage
"""

import sys
import os
import json
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Part:
    """Represents a part to be nested"""
    id: str
    width: float
    height: float
    area: float
    quantity: int
    svg_path: str
    rotation_allowed: bool = True

@dataclass
class Board:
    """Represents a board/sheet"""
    id: str
    width: float
    height: float
    area: float
    cost: float
    quantity_available: int

@dataclass
class NestingResult:
    """Result of nesting parts on a board"""
    board: Board
    parts_fitted: List[Dict]
    utilization: float
    scrap_percentage: float
    total_parts_fitted: int
    total_parts_required: int
    success: bool
    svg_layout: str
    board_used: int = 1

class MultiBoardNestingOptimizer:
    """Optimizes nesting across multiple boards to fit ALL parts with minimal scrap"""
    
    def __init__(self, min_gap_mm: float = 5.0, margin_mm: float = 10.0):
        self.min_gap_mm = min_gap_mm
        self.margin_mm = margin_mm
        self.rotation_angles = [0, 90, 180, 270]  # Standard rotations
        # Full 360-degree rotation optimization (every 5 degrees for fine tuning)
        self.full_rotation_angles = list(range(0, 360, 5))  # 0, 5, 10, 15, ..., 355 degrees
        self.advanced_rotations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    def optimize_multi_board_nesting(self, parts: List[Part], boards: List[Board], 
                                   max_boards: int = 10, use_advanced_rotations: bool = True) -> Dict:
        """
        Optimize nesting across multiple boards to fit ALL parts with minimal scrap
        
        Args:
            parts: List of parts to nest
            boards: List of available boards (sorted by size, largest first)
            max_boards: Maximum number of boards to use
            use_advanced_rotations: Whether to use advanced rotation angles
            
        Returns:
            Dictionary with optimization results
        """
        print(f"[MULTI_BOARD] Starting multi-board optimization")
        print(f"[MULTI_BOARD] Parts: {len(parts)} types, {sum(p.quantity for p in parts)} total instances")
        print(f"[MULTI_BOARD] Boards: {len(boards)} types available")
        
        # Calculate total area needed
        total_parts_area = sum(p.area * p.quantity for p in parts)
        print(f"[MULTI_BOARD] Total parts area needed: {total_parts_area:.0f} sq mm")
        
        # Sort boards by efficiency (area/cost ratio, largest first)
        sorted_boards = sorted(boards, key=lambda b: (b.area, -b.cost), reverse=True)
        
        # Try different board combinations
        best_solution = None
        best_efficiency = float('inf')
        
        # Strategy 1: Try to fit all parts on the largest board first
        print(f"[MULTI_BOARD] Strategy 1: Single largest board")
        largest_board = sorted_boards[0]
        single_board_result = self._try_single_board_nesting(parts, largest_board, use_advanced_rotations)
        
        if single_board_result and single_board_result['success']:
            print(f"[MULTI_BOARD] ✅ All parts fit on single board: {largest_board.width}x{largest_board.height}mm")
            best_solution = single_board_result
            best_efficiency = single_board_result['total_scrap_percentage']
        else:
            print(f"[MULTI_BOARD] ❌ Single board insufficient, trying multi-board approach")
            
            # Strategy 2: Multi-board optimization
            print(f"[MULTI_BOARD] Strategy 2: Multi-board optimization")
            multi_board_result = self._optimize_multi_board_combination(
                parts, sorted_boards, max_boards, use_advanced_rotations
            )
            
            if multi_board_result and multi_board_result['success']:
                print(f"[MULTI_BOARD] ✅ Multi-board solution found")
                best_solution = multi_board_result
                best_efficiency = multi_board_result['total_scrap_percentage']
            else:
                print(f"[MULTI_BOARD] ❌ No solution found with available boards")
                return self._create_failure_result(parts, boards)
        
        # Generate comprehensive results
        result = self._generate_comprehensive_results(best_solution, parts, boards)
        
        print(f"[MULTI_BOARD] Final result: {result['total_boards_used']} boards, "
              f"{result['total_scrap_percentage']:.1%} scrap, "
              f"{result['total_utilization']:.1%} utilization")
        
        return result
    
    def _try_single_board_nesting(self, parts: List[Part], board: Board, use_advanced_rotations: bool) -> Optional[Dict]:
        """Try to fit all parts on a single board"""
        print(f"[SINGLE_BOARD] Trying board {board.width}x{board.height}mm")
        
        # Create all part instances
        all_instances = []
        for part in parts:
            for i in range(part.quantity):
                all_instances.append({
                    'id': f"{part.id}_{i+1}",
                    'part_id': part.id,
                    'instance': i + 1,
                    'width': part.width,
                    'height': part.height,
                    'area': part.area,
                    'svg_path': part.svg_path,
                    'rotation_allowed': part.rotation_allowed
                })
        
        # Try different rotation strategies - use full 360-degree rotation for maximum optimization
        rotation_angles = self.full_rotation_angles if use_advanced_rotations else self.rotation_angles
        
        best_result = None
        best_fitness = 0
        
        for rotation_angle in rotation_angles:
            result = self._nest_parts_on_board(all_instances, board, rotation_angle)
            if result:
                fitness = len(result['fitted_parts']) + (result['utilization'] * 0.1)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_result = result
        
        if best_result and best_result['total_parts_fitted'] == len(all_instances):
            return {
                'success': True,
                'boards_used': [best_result],
                'total_boards_used': 1,
                'total_scrap_percentage': best_result['scrap_percentage'],
                'total_utilization': best_result['utilization'],
                'total_cost': board.cost,
                'all_parts_fitted': True
            }
        
        return None
    
    def _optimize_multi_board_combination(self, parts: List[Part], boards: List[Board], 
                                        max_boards: int, use_advanced_rotations: bool) -> Optional[Dict]:
        """Optimize across multiple boards using different strategies"""
        
        # Create all part instances
        all_instances = []
        for part in parts:
            for i in range(part.quantity):
                all_instances.append({
                    'id': f"{part.id}_{i+1}",
                    'part_id': part.id,
                    'instance': i + 1,
                    'width': part.width,
                    'height': part.height,
                    'area': part.area,
                    'svg_path': part.svg_path,
                    'rotation_allowed': part.rotation_allowed
                })
        
        print(f"[MULTI_BOARD] Optimizing {len(all_instances)} part instances across multiple boards")
        
        # Strategy A: Largest board first, then next largest
        result_a = self._strategy_largest_first(all_instances, boards, max_boards, use_advanced_rotations)
        
        # Strategy B: Most efficient board first (area/cost ratio)
        result_b = self._strategy_most_efficient(all_instances, boards, max_boards, use_advanced_rotations)
        
        # Strategy C: Hybrid approach - try different board combinations
        result_c = self._strategy_hybrid(all_instances, boards, max_boards, use_advanced_rotations)
        
        # Choose best result
        results = [r for r in [result_a, result_b, result_c] if r and r['success']]
        
        if not results:
            return None
        
        # Select result with lowest total scrap percentage
        best_result = min(results, key=lambda r: r['total_scrap_percentage'])
        
        print(f"[MULTI_BOARD] Best strategy found: {best_result['total_boards_used']} boards, "
              f"{best_result['total_scrap_percentage']:.1%} scrap")
        
        return best_result
    
    def _strategy_largest_first(self, instances: List[Dict], boards: List[Board], 
                              max_boards: int, use_advanced_rotations: bool) -> Optional[Dict]:
        """Strategy: Use largest boards first"""
        print(f"[STRATEGY_A] Largest boards first")
        
        used_boards = []
        remaining_instances = instances.copy()
        rotation_angles = self.advanced_rotations if use_advanced_rotations else self.rotation_angles
        
        for board in boards:
            if len(used_boards) >= max_boards or not remaining_instances:
                break
            
            # Try to fit remaining instances on this board
            best_board_result = None
            best_fitness = 0
            
            for rotation_angle in rotation_angles:
                result = self._nest_parts_on_board(remaining_instances, board, rotation_angle)
                if result:
                    fitness = len(result['fitted_parts']) + (result['utilization'] * 0.1)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_board_result = result
            
            if best_board_result and best_board_result['fitted_parts']:
                # Remove fitted parts from remaining
                fitted_ids = {p['id'] for p in best_board_result['fitted_parts']}
                remaining_instances = [i for i in remaining_instances if i['id'] not in fitted_ids]
                
                used_boards.append(best_board_result)
                print(f"[STRATEGY_A] Board {board.width}x{board.height}mm: {len(best_board_result['fitted_parts'])} parts fitted")
        
        if not remaining_instances:
            return self._create_multi_board_result(used_boards, boards)
        
        return None
    
    def _strategy_most_efficient(self, instances: List[Dict], boards: List[Board], 
                               max_boards: int, use_advanced_rotations: bool) -> Optional[Dict]:
        """Strategy: Use most efficient boards first (area/cost ratio)"""
        print(f"[STRATEGY_B] Most efficient boards first")
        
        # Sort boards by efficiency (area/cost ratio)
        efficient_boards = sorted(boards, key=lambda b: b.area / max(b.cost, 1), reverse=True)
        
        used_boards = []
        remaining_instances = instances.copy()
        rotation_angles = self.advanced_rotations if use_advanced_rotations else self.rotation_angles
        
        for board in efficient_boards:
            if len(used_boards) >= max_boards or not remaining_instances:
                break
            
            # Try to fit remaining instances on this board
            best_board_result = None
            best_fitness = 0
            
            for rotation_angle in rotation_angles:
                result = self._nest_parts_on_board(remaining_instances, board, rotation_angle)
                if result:
                    fitness = len(result['fitted_parts']) + (result['utilization'] * 0.1)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_board_result = result
            
            if best_board_result and best_board_result['fitted_parts']:
                # Remove fitted parts from remaining
                fitted_ids = {p['id'] for p in best_board_result['fitted_parts']}
                remaining_instances = [i for i in remaining_instances if i['id'] not in fitted_ids]
                
                used_boards.append(best_board_result)
                print(f"[STRATEGY_B] Board {board.width}x{board.height}mm: {len(best_board_result['fitted_parts'])} parts fitted")
        
        if not remaining_instances:
            return self._create_multi_board_result(used_boards, boards)
        
        return None
    
    def _strategy_hybrid(self, instances: List[Dict], boards: List[Board], 
                       max_boards: int, use_advanced_rotations: bool) -> Optional[Dict]:
        """Strategy: Hybrid approach - try different combinations"""
        print(f"[STRATEGY_C] Hybrid approach")
        
        # Try different board combinations
        best_result = None
        best_efficiency = float('inf')
        
        # Try combinations of 2-3 different board types
        for num_board_types in range(1, min(4, len(boards) + 1)):
            for board_combination in self._get_board_combinations(boards, num_board_types):
                result = self._try_board_combination(instances, board_combination, max_boards, use_advanced_rotations)
                if result and result['success']:
                    efficiency = result['total_scrap_percentage']
                    if efficiency < best_efficiency:
                        best_efficiency = efficiency
                        best_result = result
        
        return best_result
    
    def _get_board_combinations(self, boards: List[Board], num_types: int) -> List[List[Board]]:
        """Get combinations of board types"""
        if num_types == 1:
            return [[board] for board in boards]
        
        combinations = []
        for i in range(len(boards)):
            for j in range(i + 1, len(boards)):
                combinations.append([boards[i], boards[j]])
        
        return combinations
    
    def _try_board_combination(self, instances: List[Dict], board_types: List[Board], 
                             max_boards: int, use_advanced_rotations: bool) -> Optional[Dict]:
        """Try a specific combination of board types"""
        used_boards = []
        remaining_instances = instances.copy()
        rotation_angles = self.advanced_rotations if use_advanced_rotations else self.rotation_angles
        
        # Cycle through board types
        board_index = 0
        while remaining_instances and len(used_boards) < max_boards:
            board = board_types[board_index % len(board_types)]
            
            # Try to fit remaining instances on this board
            best_board_result = None
            best_fitness = 0
            
            for rotation_angle in rotation_angles:
                result = self._nest_parts_on_board(remaining_instances, board, rotation_angle)
                if result:
                    fitness = len(result['fitted_parts']) + (result['utilization'] * 0.1)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_board_result = result
            
            if best_board_result and best_board_result['fitted_parts']:
                # Remove fitted parts from remaining
                fitted_ids = {p['id'] for p in best_board_result['fitted_parts']}
                remaining_instances = [i for i in remaining_instances if i['id'] not in fitted_ids]
                
                used_boards.append(best_board_result)
            
            board_index += 1
        
        if not remaining_instances:
            return self._create_multi_board_result(used_boards, board_types)
        
        return None
    
    def _nest_parts_on_board(self, instances: List[Dict], board: Board, rotation_angle: float) -> Optional[Dict]:
        """Nest parts on a single board using improved bottom-left-fill algorithm"""
        
        available_width = board.width - (2 * self.margin_mm)
        available_height = board.height - (2 * self.margin_mm)
        
        if available_width <= 0 or available_height <= 0:
            return None
        
        # Sort instances by area (largest first) for better packing
        sorted_instances = sorted(instances, key=lambda i: i['area'], reverse=True)
        
        fitted_parts = []
        occupied_rectangles = []  # Track occupied areas to avoid overlaps
        
        def can_place_part(x, y, width, height):
            """Check if a part can be placed at (x, y) without overlapping existing parts"""
            # Check board boundaries
            if x + width > board.width - self.margin_mm or y + height > board.height - self.margin_mm:
                return False
            
            # Check for overlaps with existing parts
            for rect in occupied_rectangles:
                if not (x + width <= rect['x'] or x >= rect['x'] + rect['width'] or 
                       y + height <= rect['y'] or y >= rect['y'] + rect['height']):
                    return False
            return True
        
        def find_best_position(width, height):
            """Find the best position for a part using efficient bottom-left-fill strategy"""
            best_x, best_y = None, None
            min_y = float('inf')
            
            # Use larger step size for efficiency (20mm steps instead of 5mm)
            step_size = max(20, min(width, height) // 2)
            
            # Try placing at strategic positions
            for y in range(int(self.margin_mm), int(board.height - height - self.margin_mm) + 1, int(step_size)):
                for x in range(int(self.margin_mm), int(board.width - width - self.margin_mm) + 1, int(step_size)):
                    if can_place_part(x, y, width, height):
                        # Prefer positions closer to bottom-left
                        if y < min_y or (y == min_y and (best_x is None or x < best_x)):
                            best_x, best_y = x, y
                            min_y = y
                            # If we found a good position, try to optimize it with smaller steps
                            for dy in range(-int(step_size//2), int(step_size//2) + 1, 5):
                                for dx in range(-int(step_size//2), int(step_size//2) + 1, 5):
                                    new_x, new_y = x + dx, y + dy
                                    if (new_x >= self.margin_mm and new_y >= self.margin_mm and
                                        new_x + width <= board.width - self.margin_mm and
                                        new_y + height <= board.height - self.margin_mm and
                                        can_place_part(new_x, new_y, width, height)):
                                        if new_y < min_y or (new_y == min_y and new_x < best_x):
                                            best_x, best_y = new_x, new_y
                                            min_y = new_y
            
            return best_x, best_y
        
        for instance in sorted_instances:
            # Apply rotation - calculate rotated dimensions for any angle
            original_width = instance['width']
            original_height = instance['height']
            
            # For any rotation angle, calculate the bounding box dimensions
            import math
            radians = math.radians(rotation_angle)
            cos_angle = abs(math.cos(radians))
            sin_angle = abs(math.sin(radians))
            
            # Calculate rotated bounding box dimensions
            part_width = original_width * cos_angle + original_height * sin_angle
            part_height = original_width * sin_angle + original_height * cos_angle
            
            # Find best position for this part
            x, y = find_best_position(part_width, part_height)
            
            if x is not None and y is not None:
                # Part can be placed
                fitted_parts.append({
                    'id': instance['id'],
                    'part_id': instance['part_id'],
                    'instance': instance['instance'],
                    'x': x,
                    'y': y,
                    'width': part_width,
                    'height': part_height,
                    'rotation': rotation_angle,
                    'area': instance['area'],
                    'svg_path': instance['svg_path'],
                    'original_width': original_width,
                    'original_height': original_height
                })
                
                # Add to occupied rectangles
                occupied_rectangles.append({
                    'x': x,
                    'y': y,
                    'width': part_width,
                    'height': part_height
                })
            else:
                # No space for this part
                break
        
        if not fitted_parts:
            return None
        
        # Calculate metrics
        total_fitted_area = sum(p['area'] for p in fitted_parts)
        utilization = min(1.0, total_fitted_area / board.area) if board.area > 0 else 0  # Cap at 100%
        scrap_percentage = max(0, 1.0 - utilization)  # Ensure scrap is never negative
        
        # Debug logging (reduced frequency)
        if len(fitted_parts) > 0:  # Only log when parts are actually fitted
            print(f"[NESTING_DEBUG] Board: {board.width}x{board.height}mm (area: {board.area:.0f}mm²)")
            print(f"[NESTING_DEBUG] Fitted parts: {len(fitted_parts)}, total area: {total_fitted_area:.0f}mm²")
            print(f"[NESTING_DEBUG] Utilization: {utilization:.1%}, Scrap: {scrap_percentage:.1%}")
        
        # Generate SVG layout
        svg_layout = self._generate_board_svg(board, fitted_parts)
        
        return {
            'board': board,
            'fitted_parts': fitted_parts,
            'utilization': utilization,
            'scrap_percentage': scrap_percentage,
            'total_parts_fitted': len(fitted_parts),
            'total_parts_required': len(instances),
            'success': len(fitted_parts) == len(instances),
            'svg_layout': svg_layout,
            'board_index': 1  # Add board index for tracking
        }
    
    def _generate_board_svg(self, board: Board, fitted_parts: List[Dict]) -> str:
        """Generate enhanced SVG layout for a board with nested parts"""
        
        # Calculate appropriate scale for display
        max_display_width = 800
        max_display_height = 600
        scale_x = max_display_width / board.width if board.width > 0 else 1
        scale_y = max_display_height / board.height if board.height > 0 else 1
        scale = min(scale_x, scale_y, 1)  # Don't scale up
        
        svg_width = board.width * scale
        svg_height = board.height * scale
        
        # Enhanced color palette with better contrast
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
        ]
        
        # Calculate utilization for color coding
        total_fitted_area = sum(p['area'] for p in fitted_parts)
        utilization = total_fitted_area / board.area if board.area > 0 else 0
        
        svg_content = f'''<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {board.width} {board.height}" xmlns="http://www.w3.org/2000/svg">
    <!-- Board background with utilization indicator -->
    <rect x="0" y="0" width="{board.width}" height="{board.height}" 
          fill="#f8f9fa" stroke="#2c3e50" stroke-width="3"/>
    
    <!-- Board title -->
    <text x="{board.width/2}" y="20" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="16" font-weight="bold" fill="#2c3e50">
        Board: {board.width:.0f} × {board.height:.0f} mm
    </text>
    
    <!-- Utilization indicator -->
    <text x="{board.width/2}" y="40" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="12" fill="#2c3e50">
        Utilization: {utilization:.1%} | Parts: {len(fitted_parts)}
    </text>
    
    <!-- Margin guides -->
    <rect x="{self.margin_mm}" y="{self.margin_mm}" width="{board.width - 2*self.margin_mm}" height="{board.height - 2*self.margin_mm}" 
          fill="none" stroke="#6c757d" stroke-width="1" stroke-dasharray="5,5" opacity="0.7"/>
    
    <!-- Grid lines for better visualization -->
    <defs>
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#e9ecef" stroke-width="0.5"/>
        </pattern>
    </defs>
    <rect width="100%" height="100%" fill="url(#grid)" opacity="0.3"/>
    
    <!-- Nested parts -->'''
        
        for i, part in enumerate(fitted_parts):
            # Use part_id for consistent coloring
            color_index = hash(str(part['part_id'])) % len(colors)
            color = colors[color_index]
            
            x = part['x']
            y = part['y']
            width = part['width']
            height = part['height']
            rotation = part['rotation']
            
            # Get original dimensions for proper rotation
            original_width = part.get('original_width', width)
            original_height = part.get('original_height', height)
            
            # Create transform for rotation - rotate around the part's center
            if rotation != 0:
                # Calculate the center of the original part (before rotation)
                original_center_x = original_width / 2
                original_center_y = original_height / 2
                transform = f"translate({x + original_center_x}, {y + original_center_y}) rotate({rotation}) translate({-original_center_x}, {-original_center_y})"
            else:
                transform = f"translate({x}, {y})"
            
            # Add shadow effect
            shadow_offset = 2
            svg_content += f'''
    <!-- Shadow for part {i+1} -->
    <g transform="{transform}">
        <rect x="{shadow_offset}" y="{shadow_offset}" width="{original_width}" height="{original_height}" 
              fill="#000" opacity="0.2"/>
    </g>
    
    <!-- Part {i+1} -->
    <g transform="{transform}">
        <rect x="0" y="0" width="{original_width}" height="{original_height}" 
              fill="{color}" stroke="#fff" stroke-width="2" opacity="0.8"/>
        
        <!-- Part label -->
        <text x="{original_width/2}" y="{original_height/2 - 5}" text-anchor="middle" dominant-baseline="middle" 
              font-family="Arial, sans-serif" font-size="{max(8, min(original_width, original_height) * 0.15)}" 
              fill="#fff" font-weight="bold">
              {part['part_id']}
        </text>
        
        <!-- Instance number -->
        <text x="{original_width/2}" y="{original_height/2 + 8}" text-anchor="middle" dominant-baseline="middle" 
              font-family="Arial, sans-serif" font-size="{max(6, min(original_width, original_height) * 0.1)}" 
              fill="#fff">
              #{part['instance']}
        </text>
        
        <!-- Dimensions -->
        <text x="{original_width/2}" y="{original_height - 2}" text-anchor="middle" dominant-baseline="middle" 
              font-family="Arial, sans-serif" font-size="{max(6, min(original_width, original_height) * 0.08)}" 
              fill="#fff" opacity="0.8">
              {original_width:.0f}×{original_height:.0f}
        </text>
    </g>'''
        
        svg_content += '''
</svg>'''
        
        return svg_content
    
    def _create_multi_board_result(self, used_boards: List[Dict], boards: List[Board]) -> Dict:
        """Create result for multi-board solution"""
        
        total_boards_used = len(used_boards)
        total_cost = sum(board['board'].cost for board in used_boards)
        
        # Calculate weighted averages
        total_area = sum(board['board'].area for board in used_boards)
        total_used_area = sum(board['utilization'] * board['board'].area for board in used_boards)
        total_utilization = min(1.0, total_used_area / total_area) if total_area > 0 else 0  # Cap at 100%
        total_scrap_percentage = max(0, 1.0 - total_utilization)  # Ensure scrap is never negative
        
        return {
            'success': True,
            'boards_used': used_boards,
            'total_boards_used': total_boards_used,
            'total_scrap_percentage': total_scrap_percentage,
            'total_utilization': total_utilization,
            'total_cost': total_cost,
            'all_parts_fitted': True
        }
    
    def _create_failure_result(self, parts: List[Part], boards: List[Board]) -> Dict:
        """Create result when no solution is found"""
        return {
            'success': False,
            'boards_used': [],
            'total_boards_used': 0,
            'total_scrap_percentage': 1.0,
            'total_utilization': 0.0,
            'total_cost': 0.0,
            'all_parts_fitted': False,
            'error': 'No solution found with available boards'
        }
    
    def _generate_comprehensive_results(self, solution: Dict, parts: List[Part], boards: List[Board]) -> Dict:
        """Generate comprehensive results with all details"""
        
        if not solution or not solution['success']:
            return self._create_failure_result(parts, boards)
        
        # Calculate summary statistics
        total_parts_required = sum(p.quantity for p in parts)
        total_parts_fitted = sum(len(board['fitted_parts']) for board in solution['boards_used'])
        
        # Create detailed board results
        detailed_boards = []
        for i, board_result in enumerate(solution['boards_used']):
            detailed_boards.append({
                'board_index': i + 1,
                'board': {
                    'id': board_result['board'].id,
                    'width_mm': board_result['board'].width,
                    'height_mm': board_result['board'].height,
                    'area_sq_mm': board_result['board'].area,
                    'cost': board_result['board'].cost
                },
                'nested_parts': board_result['fitted_parts'],
                'utilization': board_result['utilization'],
                'scrap_percentage': board_result['scrap_percentage'],
                'total_parts_nested': board_result['total_parts_fitted'],
                'total_parts_required': board_result['total_parts_required'],
                'fitting_success': board_result['success'],
                'svg_layout': board_result['svg_layout']
            })
        
        return {
            'success': True,
            'optimization_type': 'multi_board',
            'total_boards_used': solution['total_boards_used'],
            'total_scrap_percentage': solution['total_scrap_percentage'],
            'total_utilization': solution['total_utilization'],
            'total_cost': solution['total_cost'],
            'all_parts_fitted': solution['all_parts_fitted'],
            'total_parts_required': total_parts_required,
            'total_parts_fitted': total_parts_fitted,
            'boards_used': detailed_boards,
            'best_board': detailed_boards[0] if detailed_boards else None,
            'efficiency_score': (1.0 - solution['total_scrap_percentage']) * 100,
            'cost_per_part': solution['total_cost'] / max(total_parts_fitted, 1)
        }

def test_multi_board_optimizer():
    """Test the multi-board nesting optimizer"""
    print("🧪 Testing Multi-Board Nesting Optimizer")
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
    
    print(f"📋 Test Setup:")
    print(f"  • Parts: {len(parts)} types, {sum(p.quantity for p in parts)} total instances")
    print(f"  • Boards: {len(boards)} types available")
    for board in boards:
        print(f"    - Board {board.id}: {board.width}×{board.height}mm, ${board.cost}")
    
    # Test optimizer
    optimizer = MultiBoardNestingOptimizer(min_gap_mm=5.0, margin_mm=10.0)
    
    print(f"\n🔧 Running multi-board optimization...")
    result = optimizer.optimize_multi_board_nesting(
        parts=parts,
        boards=boards,
        max_boards=5,
        use_advanced_rotations=True
    )
    
    print(f"\n📊 Results:")
    print(f"  • Success: {'✅' if result['success'] else '❌'}")
    print(f"  • Boards used: {result['total_boards_used']}")
    print(f"  • Total scrap: {result['total_scrap_percentage']:.1%}")
    print(f"  • Total utilization: {result['total_utilization']:.1%}")
    print(f"  • Total cost: ${result['total_cost']:.2f}")
    print(f"  • Parts fitted: {result['total_parts_fitted']}/{result['total_parts_required']}")
    print(f"  • All parts fitted: {'✅' if result['all_parts_fitted'] else '❌'}")
    
    if result['success']:
        print(f"\n📋 Board Details:")
        for i, board in enumerate(result['boards_used']):
            print(f"  Board {i+1}: {board['board']['width_mm']}×{board['board']['height_mm']}mm")
            print(f"    • Parts: {board['total_parts_nested']}")
            print(f"    • Utilization: {board['utilization']:.1%}")
            print(f"    • Scrap: {board['scrap_percentage']:.1%}")
            print(f"    • SVG generated: {'✅' if board['svg_layout'] else '❌'}")
    
    return result

if __name__ == "__main__":
    test_multi_board_optimizer()
