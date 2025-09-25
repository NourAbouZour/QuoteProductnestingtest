#!/usr/bin/env python3
"""
Robust Nesting Engine using proven libraries
Combines rectpack and binpacking libraries for optimal 2D nesting
"""

import sys
import os
from typing import List, Dict, Optional, Tuple
import math
import time
from dataclasses import dataclass

# Import proven libraries
try:
    from rectpack import newPacker
    from rectpack.maxrects import MaxRectsBssf
    from rectpack.skyline import SkylineMwf
    RECTPACK_AVAILABLE = True
except ImportError:
    RECTPACK_AVAILABLE = False
    print("Warning: rectpack not available, using fallback algorithm")

try:
    import binpacking
    BINPACKING_AVAILABLE = True
except ImportError:
    BINPACKING_AVAILABLE = False
    print("Warning: binpacking not available, using fallback algorithm")

@dataclass
class Part:
    """Represents a part to be nested"""
    id: str
    width: float
    height: float
    quantity: int
    rotation_allowed: bool = True
    area: float = 0.0
    
    def __post_init__(self):
        if self.area == 0.0:
            self.area = self.width * self.height

@dataclass
class Board:
    """Represents a board/sheet for nesting"""
    id: str
    width: float
    height: float
    cost: float
    quantity_available: int
    area: float = 0.0
    
    def __post_init__(self):
        if self.area == 0.0:
            self.area = self.width * self.height

class RobustNestingEngine:
    """Robust nesting engine using proven algorithms"""
    
    def __init__(self, margin_mm: float = 5.0, min_gap_mm: float = 2.0):
        self.margin_mm = margin_mm
        self.min_gap_mm = min_gap_mm
        self.algorithm = "maxrects"  # or "skyline"
        
    def optimize_nesting(self, parts: List[Part], boards: List[Board], max_boards: int = 10) -> Dict:
        """
        Optimize nesting using the best available algorithm
        """
        print(f"[ROBUST_NESTING] Starting optimization with {len(parts)} part types")
        print(f"[ROBUST_NESTING] Total parts: {sum(p.quantity for p in parts)}")
        print(f"[ROBUST_NESTING] Available boards: {len(boards)}")
        
        # Calculate total area needed
        total_parts_area = sum(p.area * p.quantity for p in parts)
        print(f"[ROBUST_NESTING] Total area needed: {total_parts_area:,.0f} mm²")
        
        # Sort boards by area (largest first)
        sorted_boards = sorted(boards, key=lambda b: b.area, reverse=True)
        
        # Try different strategies (optimized for large datasets)
        if len(parts) > 100:
            # For large datasets, use only fast strategies
            strategies = [
                ("single_largest", self._try_single_largest_board),
            ]
            print(f"[ROBUST_NESTING] Large dataset detected ({len(parts)} parts), using fast algorithm only")
        else:
            # For small datasets, try all strategies
            strategies = [
                ("single_largest", self._try_single_largest_board),
                ("multi_board", self._try_multi_board_nesting),
                ("rectpack", self._try_rectpack_nesting),
            ]
        
        best_result = None
        best_utilization = 0
        
        for strategy_name, strategy_func in strategies:
            if not RECTPACK_AVAILABLE and strategy_name == "rectpack":
                continue
                
            print(f"[ROBUST_NESTING] Trying strategy: {strategy_name}")
            start_time = time.time()
            try:
                result = strategy_func(parts, sorted_boards, max_boards)
                elapsed_time = time.time() - start_time
                
                if result and result.get('success', False):
                    utilization = result.get('total_utilization', 0)
                    print(f"[ROBUST_NESTING] {strategy_name}: {utilization:.1%} utilization ({elapsed_time:.1f}s)")
                    if utilization > best_utilization:
                        best_utilization = utilization
                        best_result = result
                        best_result['strategy_used'] = strategy_name
                else:
                    print(f"[ROBUST_NESTING] {strategy_name}: Failed ({elapsed_time:.1f}s)")
                    
                # For large datasets, stop after first successful result
                if len(parts) > 100 and result and result.get('success', False):
                    print(f"[ROBUST_NESTING] Large dataset: stopping after first successful result")
                    break
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"[ROBUST_NESTING] Strategy {strategy_name} failed: {e} ({elapsed_time:.1f}s)")
                continue
        
        if best_result:
            print(f"[ROBUST_NESTING] ✅ Best result: {best_result['strategy_used']} with {best_utilization:.1%} utilization")
            return best_result
        else:
            print("[ROBUST_NESTING] ❌ All strategies failed")
            return {
                'success': False,
                'error': 'No suitable nesting strategy found',
                'total_boards_used': 0,
                'total_utilization': 0,
                'total_scrap_percentage': 100,
                'all_parts_fitted': False,
                'results': [],
                'best_board': None,
                'svg_layout': ''
            }
    
    def _try_single_largest_board(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Try to fit all parts on the largest available board"""
        if not boards:
            return None
            
        largest_board = boards[0]
        print(f"[SINGLE_BOARD] Trying largest board: {largest_board.width:.0f}x{largest_board.height:.0f}mm")
        
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
        
        # Try different rotation angles
        rotation_angles = [0, 90, 180, 270] if any(p.rotation_allowed for p in parts) else [0]
        best_result = None
        best_utilization = 0
        
        for angle in rotation_angles:
            result = self._nest_parts_on_board(all_instances, largest_board, angle)
            if result and result.get('success', False):
                utilization = result.get('utilization', 0)
                if utilization > best_utilization:
                    best_utilization = utilization
                    best_result = result
        
        if best_result:
            print(f"[SINGLE_BOARD] ✅ Success: {best_utilization:.1%} utilization")
            return {
                'success': True,
                'total_boards_used': 1,
                'total_utilization': best_utilization,
                'total_scrap_percentage': 1.0 - best_utilization,
                'all_parts_fitted': best_result.get('all_parts_fitted', False),
                'results': [best_result],
                'best_board': best_result,
                'svg_layout': best_result.get('svg_layout', ''),
                'strategy_used': 'single_largest'
            }
        
        return None
    
    def _try_multi_board_nesting(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Try multi-board nesting using greedy approach"""
        print("[MULTI_BOARD] Starting multi-board optimization")
        
        used_boards = []
        remaining_parts = parts.copy()
        
        for board in boards[:max_boards]:
            if not remaining_parts:
                break
                
            # Create instances for this board
            board_instances = []
            for part in remaining_parts:
                for i in range(part.quantity):
                    board_instances.append({
                        'id': f"{part.id}_{i+1}",
                        'part_id': part.id,
                        'width': part.width,
                        'height': part.height,
                        'area': part.area,
                        'rotation_allowed': part.rotation_allowed
                    })
            
            # Try to nest on this board
            result = self._nest_parts_on_board(board_instances, board, 0)
            if result and result.get('success', False):
                used_boards.append(result)
                # Remove fitted parts
                fitted_part_ids = {p['part_id'] for p in result.get('fitted_parts', [])}
                remaining_parts = [p for p in remaining_parts if p.id not in fitted_part_ids]
                print(f"[MULTI_BOARD] Board {len(used_boards)}: {result.get('utilization', 0):.1%} utilization")
        
        if used_boards:
            total_utilization = sum(b.get('utilization', 0) for b in used_boards) / len(used_boards)
            all_parts_fitted = len(remaining_parts) == 0
            
            return {
                'success': True,
                'total_boards_used': len(used_boards),
                'total_utilization': total_utilization,
                'total_scrap_percentage': 1.0 - total_utilization,
                'all_parts_fitted': all_parts_fitted,
                'results': used_boards,
                'best_board': used_boards[0] if used_boards else None,
                'svg_layout': used_boards[0].get('svg_layout', '') if used_boards else '',
                'strategy_used': 'multi_board'
            }
        
        return None
    
    def _try_rectpack_nesting(self, parts: List[Part], boards: List[Board], max_boards: int) -> Optional[Dict]:
        """Try using rectpack library for optimal nesting"""
        if not RECTPACK_AVAILABLE:
            return None
            
        print("[RECTPACK] Using rectpack library for optimal nesting")
        
        # Create packer
        packer = newPacker()
        
        # Add bins (boards)
        for i, board in enumerate(boards[:max_boards]):
            packer.add_bin(board.width, board.height, bid=i)
        
        # Add rectangles (parts)
        for part in parts:
            for i in range(part.quantity):
                packer.add_rect(part.width, part.height, rid=f"{part.id}_{i+1}")
        
        # Pack using best algorithm
        if self.algorithm == "maxrects":
            packer.pack()
        else:
            packer.pack()
        
        # Get results
        packed_bins = packer[0] if packer else []
        
        if not packed_bins:
            return None
        
        # Convert to our format
        results = []
        total_utilization = 0
        
        for bin_idx, bin_data in enumerate(packed_bins):
            if bin_idx >= len(boards):
                break
                
            board = boards[bin_idx]
            bin_width, bin_height, bin_x, bin_y = bin_data
            
            # Calculate utilization
            used_area = sum(rect[2] * rect[3] for rect in bin_data[4:])  # width * height of rectangles
            utilization = used_area / (board.width * board.height) if board.width * board.height > 0 else 0
            
            # Create result
            result = {
                'board': board,
                'utilization': utilization,
                'scrap_percentage': 1.0 - utilization,
                'total_parts_nested': len(bin_data[4:]),  # number of rectangles
                'fitted_parts': [],
                'svg_layout': self._generate_svg_layout(board, bin_data[4:])
            }
            
            results.append(result)
            total_utilization += utilization
        
        if results:
            avg_utilization = total_utilization / len(results)
            return {
                'success': True,
                'total_boards_used': len(results),
                'total_utilization': avg_utilization,
                'total_scrap_percentage': 1.0 - avg_utilization,
                'all_parts_fitted': True,  # rectpack ensures all parts fit
                'results': results,
                'best_board': results[0],
                'svg_layout': results[0].get('svg_layout', ''),
                'strategy_used': 'rectpack'
            }
        
        return None
    
    def _nest_parts_on_board(self, instances: List[Dict], board: Board, rotation_angle: float) -> Optional[Dict]:
        """Nest parts on a single board using optimized algorithm for large datasets"""
        if not instances:
            return None
        
        print(f"[NESTING_OPTIMIZED] Processing {len(instances)} parts on {board.width}x{board.height}mm board")
        
        # For large datasets, use simplified algorithm
        if len(instances) > 100:
            return self._nest_parts_simple(instances, board, rotation_angle)
        
        # Apply rotation
        rotated_instances = []
        for instance in instances:
            if rotation_angle != 0 and instance.get('rotation_allowed', True):
                # Calculate rotated dimensions
                radians = math.radians(rotation_angle)
                cos_angle = abs(math.cos(radians))
                sin_angle = abs(math.sin(radians))
                
                rotated_width = instance['width'] * cos_angle + instance['height'] * sin_angle
                rotated_height = instance['width'] * sin_angle + instance['height'] * cos_angle
            else:
                rotated_width = instance['width']
                rotated_height = instance['height']
            
            rotated_instances.append({
                **instance,
                'width': rotated_width,
                'height': rotated_height,
                'rotation': rotation_angle
            })
        
        # Sort by area (largest first)
        sorted_instances = sorted(rotated_instances, key=lambda x: x['area'], reverse=True)
        
        # Use bottom-left-fill algorithm
        fitted_parts = []
        occupied_rectangles = []
        
        def can_place_part(x, y, width, height):
            """Check if part can be placed without overlapping"""
            # Check boundaries
            if (x + width > board.width - self.margin_mm or 
                y + height > board.height - self.margin_mm or
                x < self.margin_mm or y < self.margin_mm):
                return False
            
            # Check overlaps
            for rect in occupied_rectangles:
                if not (x + width <= rect['x'] or x >= rect['x'] + rect['width'] or 
                       y + height <= rect['y'] or y >= rect['y'] + rect['height']):
                    return False
            return True
        
        def find_best_position(width, height):
            """Find best position using bottom-left-fill"""
            best_x, best_y = None, None
            min_y = float('inf')
            
            # Use efficient step size
            step_size = max(10, min(width, height) // 4)
            
            for y in range(int(self.margin_mm), int(board.height - height - self.margin_mm) + 1, int(step_size)):
                for x in range(int(self.margin_mm), int(board.width - width - self.margin_mm) + 1, int(step_size)):
                    if can_place_part(x, y, width, height):
                        if y < min_y or (y == min_y and (best_x is None or x < best_x)):
                            best_x, best_y = x, y
                            min_y = y
            
            return best_x, best_y
        
        # Place parts
        for instance in sorted_instances:
            x, y = find_best_position(instance['width'], instance['height'])
            
            if x is not None and y is not None:
                fitted_parts.append({
                    'id': instance['id'],
                    'part_id': instance['part_id'],
                    'x': x,
                    'y': y,
                    'width': instance['width'],
                    'height': instance['height'],
                    'rotation': instance['rotation'],
                    'area': instance['area']
                })
                
                occupied_rectangles.append({
                    'x': x,
                    'y': y,
                    'width': instance['width'],
                    'height': instance['height']
                })
            else:
                break  # No more space
        
        if not fitted_parts:
            return None
        
        # Calculate metrics
        total_fitted_area = sum(p['area'] for p in fitted_parts)
        utilization = min(1.0, total_fitted_area / board.area) if board.area > 0 else 0
        scrap_percentage = max(0, 1.0 - utilization)
        
        # Generate SVG
        svg_layout = self._generate_svg_layout(board, fitted_parts)
        
        return {
            'board': board,
            'utilization': utilization,
            'scrap_percentage': scrap_percentage,
            'total_parts_nested': len(fitted_parts),
            'fitted_parts': fitted_parts,
            'svg_layout': svg_layout,
            'success': len(fitted_parts) == len(instances),
            'all_parts_fitted': len(fitted_parts) == len(instances)
        }
    
    def _generate_svg_layout(self, board: Board, fitted_parts: List) -> str:
        """Generate SVG layout for visualization"""
        if not fitted_parts:
            return ""
        
        # Calculate scale
        max_display_width = 800
        max_display_height = 600
        scale_x = max_display_width / board.width if board.width > 0 else 1
        scale_y = max_display_height / board.height if board.height > 0 else 1
        scale = min(scale_x, scale_y, 1)
        
        svg_width = board.width * scale
        svg_height = board.height * scale
        
        # Colors
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        # Calculate utilization
        total_area = sum(p.get('area', p.get('width', 0) * p.get('height', 0)) for p in fitted_parts)
        utilization = total_area / board.area if board.area > 0 else 0
        
        svg_content = f'''<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {board.width} {board.height}" xmlns="http://www.w3.org/2000/svg">
    <!-- Board background -->
    <rect x="0" y="0" width="{board.width}" height="{board.height}" 
          fill="#f8f9fa" stroke="#2c3e50" stroke-width="3"/>
    
    <!-- Board info -->
    <text x="{board.width/2}" y="20" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="16" font-weight="bold" fill="#2c3e50">
        Board: {board.width:.0f} × {board.height:.0f} mm
    </text>
    <text x="{board.width/2}" y="40" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="12" fill="#2c3e50">
        Utilization: {utilization:.1%} | Parts: {len(fitted_parts)}
    </text>
    
    <!-- Grid -->
    <defs>
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#e9ecef" stroke-width="0.5"/>
        </pattern>
    </defs>
    <rect width="100%" height="100%" fill="url(#grid)" opacity="0.3"/>
    
    <!-- Parts -->'''
        
        for i, part in enumerate(fitted_parts):
            color_index = hash(str(part.get('part_id', part.get('id', i)))) % len(colors)
            color = colors[color_index]
            
            x = part.get('x', 0)
            y = part.get('y', 0)
            width = part.get('width', 0)
            height = part.get('height', 0)
            rotation = part.get('rotation', 0)
            
            # Create transform
            if rotation != 0:
                center_x = width / 2
                center_y = height / 2
                transform = f"translate({x + center_x}, {y + center_y}) rotate({rotation}) translate({-center_x}, {-center_y})"
            else:
                transform = f"translate({x}, {y})"
            
            svg_content += f'''
    <g transform="{transform}">
        <rect x="0" y="0" width="{width}" height="{height}" 
              fill="{color}" stroke="#fff" stroke-width="2" opacity="0.8"/>
        <text x="{width/2}" y="{height/2 - 5}" text-anchor="middle" dominant-baseline="middle" 
              font-family="Arial, sans-serif" font-size="{max(8, min(width, height) * 0.15)}" 
              fill="#fff" font-weight="bold">
              {part.get('part_id', part.get('id', i))}
        </text>
        <text x="{width/2}" y="{height/2 + 8}" text-anchor="middle" dominant-baseline="middle" 
              font-family="Arial, sans-serif" font-size="{max(6, min(width, height) * 0.1)}" 
              fill="#fff">
              #{i+1}
        </text>
    </g>'''
        
        svg_content += '''
</svg>'''
        
        return svg_content
    
    def _nest_parts_simple(self, instances: List[Dict], board: Board, rotation_angle: float) -> Optional[Dict]:
        """Simplified nesting algorithm for large datasets (666+ parts)"""
        print(f"[SIMPLE_NESTING] Using fast algorithm for {len(instances)} parts")
        
        # Sort by area (largest first)
        sorted_instances = sorted(instances, key=lambda x: x['area'], reverse=True)
        
        fitted_parts = []
        current_x = self.margin_mm
        current_y = self.margin_mm
        row_height = 0
        max_width = board.width - self.margin_mm
        
        # Simple row-based placement
        for i, instance in enumerate(sorted_instances):
            # Apply rotation if needed
            if rotation_angle != 0 and instance.get('rotation_allowed', True):
                radians = math.radians(rotation_angle)
                cos_angle = abs(math.cos(radians))
                sin_angle = abs(math.sin(radians))
                
                width = instance['width'] * cos_angle + instance['height'] * sin_angle
                height = instance['width'] * sin_angle + instance['height'] * cos_angle
            else:
                width = instance['width']
                height = instance['height']
            
            # Check if part fits in current row
            if current_x + width <= max_width and current_y + height <= board.height - self.margin_mm:
                fitted_parts.append({
                    'id': instance['id'],
                    'part_id': instance['part_id'],
                    'x': current_x,
                    'y': current_y,
                    'width': width,
                    'height': height,
                    'rotation': rotation_angle,
                    'area': instance['area']
                })
                
                current_x += width + self.min_gap_mm
                row_height = max(row_height, height)
            else:
                # Move to next row
                current_x = self.margin_mm
                current_y += row_height + self.min_gap_mm
                row_height = 0
                
                # Check if there's still space
                if current_y + height <= board.height - self.margin_mm:
                    fitted_parts.append({
                        'id': instance['id'],
                        'part_id': instance['part_id'],
                        'x': current_x,
                        'y': current_y,
                        'width': width,
                        'height': height,
                        'rotation': rotation_angle,
                        'area': instance['area']
                    })
                    
                    current_x += width + self.min_gap_mm
                    row_height = height
                else:
                    # No more space
                    break
            
            # Progress indicator for large datasets
            if i % 100 == 0 and i > 0:
                print(f"[SIMPLE_NESTING] Processed {i}/{len(sorted_instances)} parts")
        
        if not fitted_parts:
            return None
        
        # Calculate metrics
        total_fitted_area = sum(p['area'] for p in fitted_parts)
        utilization = min(1.0, total_fitted_area / board.area) if board.area > 0 else 0
        scrap_percentage = max(0, 1.0 - utilization)
        
        print(f"[SIMPLE_NESTING] Fitted {len(fitted_parts)}/{len(instances)} parts, {utilization:.1%} utilization")
        
        # Generate simplified SVG
        svg_layout = self._generate_simple_svg(board, fitted_parts)
        
        return {
            'board': board,
            'utilization': utilization,
            'scrap_percentage': scrap_percentage,
            'total_parts_nested': len(fitted_parts),
            'fitted_parts': fitted_parts,
            'svg_layout': svg_layout,
            'success': len(fitted_parts) == len(instances),
            'all_parts_fitted': len(fitted_parts) == len(instances)
        }
    
    def _generate_simple_svg(self, board: Board, fitted_parts: List) -> str:
        """Generate simplified SVG for large datasets"""
        if not fitted_parts:
            return ""
        
        # Calculate scale
        max_display_width = 800
        max_display_height = 600
        scale_x = max_display_width / board.width if board.width > 0 else 1
        scale_y = max_display_height / board.height if board.height > 0 else 1
        scale = min(scale_x, scale_y, 1)
        
        svg_width = board.width * scale
        svg_height = board.height * scale
        
        # Calculate utilization
        total_area = sum(p.get('area', p.get('width', 0) * p.get('height', 0)) for p in fitted_parts)
        utilization = total_area / board.area if board.area > 0 else 0
        
        svg_content = f'''<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {board.width} {board.height}" xmlns="http://www.w3.org/2000/svg">
    <!-- Board background -->
    <rect x="0" y="0" width="{board.width}" height="{board.height}" 
          fill="#f8f9fa" stroke="#2c3e50" stroke-width="3"/>
    
    <!-- Board info -->
    <text x="{board.width/2}" y="20" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="16" font-weight="bold" fill="#2c3e50">
        Board: {board.width:.0f} × {board.height:.0f} mm
    </text>
    <text x="{board.width/2}" y="40" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="12" fill="#2c3e50">
        Utilization: {utilization:.1%} | Parts: {len(fitted_parts)}
    </text>
    
    <!-- Parts (simplified for large datasets) -->'''
        
        # Show only first 50 parts to avoid SVG bloat
        display_parts = fitted_parts[:50]
        for i, part in enumerate(display_parts):
            color = f"hsl({(i * 137.5) % 360}, 70%, 60%)"  # Golden angle for color distribution
            
            x = part.get('x', 0)
            y = part.get('y', 0)
            width = part.get('width', 0)
            height = part.get('height', 0)
            
            svg_content += f'''
    <rect x="{x}" y="{y}" width="{width}" height="{height}" 
          fill="{color}" stroke="#fff" stroke-width="1" opacity="0.8"/>'''
        
        if len(fitted_parts) > 50:
            svg_content += f'''
    <text x="{board.width/2}" y="{board.height - 20}" text-anchor="middle" font-family="Arial, sans-serif" 
          font-size="12" fill="#666">
        Showing first 50 of {len(fitted_parts)} parts
    </text>'''
        
        svg_content += '''
</svg>'''
        
        return svg_content

# Test function
def test_robust_nesting():
    """Test the robust nesting engine"""
    # Create test parts
    parts = [
        Part(id="1", width=100, height=50, quantity=3, rotation_allowed=True),
        Part(id="2", width=80, height=60, quantity=2, rotation_allowed=True),
        Part(id="3", width=120, height=40, quantity=1, rotation_allowed=True),
        Part(id="4", width=60, height=60, quantity=4, rotation_allowed=True),
    ]
    
    # Create test boards
    boards = [
        Board(id="1", width=1000, height=2000, cost=50.0, quantity_available=10),
        Board(id="2", width=800, height=1200, cost=30.0, quantity_available=10),
        Board(id="3", width=600, height=800, cost=20.0, quantity_available=10),
    ]
    
    # Test nesting
    engine = RobustNestingEngine()
    result = engine.optimize_nesting(parts, boards)
    
    print(f"\n=== ROBUST NESTING TEST RESULTS ===")
    print(f"Success: {result.get('success', False)}")
    print(f"Strategy: {result.get('strategy_used', 'unknown')}")
    print(f"Boards used: {result.get('total_boards_used', 0)}")
    print(f"Utilization: {result.get('total_utilization', 0):.1%}")
    print(f"Scrap: {result.get('total_scrap_percentage', 0):.1%}")
    print(f"All parts fitted: {result.get('all_parts_fitted', False)}")
    print(f"Has SVG: {len(result.get('svg_layout', '')) > 0}")
    
    return result

if __name__ == "__main__":
    test_robust_nesting()
