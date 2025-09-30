"""
Leftover Strip Analyzer for Nesting Operations

This module analyzes completed nesting results to identify reusable leftover strips
that can be saved and excluded from scrap calculations.
"""

def analyze_reusable_leftover_strips(fitted_parts, board_specs, nesting_config):
    """
    Analyze nested parts to find continuous reusable leftover strips.
    
    A leftover strip qualifies if:
    1. It's a straight, continuous strip (no interruptions by parts)
    2. It's ≥20% of the board dimension in either height or width 
    3. Only one continuous strip per board should be considered
    
    Args:
        fitted_parts: List of dictionaries containing fitted parts with x, y, width, height, etc.
        board_specs: Board dimensions and specifications
        nesting_config: Nesting configuration including margins
    
    Returns:
        dict: {
            'has_leftover_strip': bool,
            'strip_type': str,  # 'vertical' or 'horizontal'
            'strip_position': float,  # Position of the strip (x for vertical, y for horizontal)
            'strip_size': float,  # Size of the strip 
            'leftover_area_sq_mm': float,  # Area of reusable strip
            'cut_line': dict  # Information about where to cut
        }
    """
    
    board_width = board_specs.get('width_mm', 0)
    board_height = board_specs.get('height_mm', 0)
    margin = nesting_config.get('sheet_margin_mm', {'top': 10, 'right': 10, 'bottom': 10, 'left': 10})
    
    if not fitted_parts or board_width <= 0 or board_height <= 0:
        return {
            'has_leftover_strip': False,
            'strip_type': None,
            'strip_position': None,
            'strip_size': 0,
            'leftover_area_sq_mm': 0,
            'cut_line': None
        }
    
    # Sort parts by position to identify bounds of used space
    parts_by_x = sorted(fitted_parts, key=lambda p: p.get('x', 0))
    parts_by_y = sorted(fitted_parts, key=lambda p: p.get('y', 0))
    
    # Find bounds of used space including part dimensions
    min_x = min(p.get('x', 0) for p in fitted_parts)
    max_x = max((p.get('x', 0) + p.get('width_mm', p.get('width', 0))) for p in fitted_parts)
    min_y = min(p.get('y', 0) for p in fitted_parts)
    max_y = max((p.get('y', 0) + p.get('height_mm', p.get('height', 0))) for p in fitted_parts)
    
    margin_left = margin.get('left', 10)
    margin_right = margin.get('right', 10)
    margin_top = margin.get('top', 10)
    margin_bottom = margin.get('bottom', 10)
    
    print(f"[LEFTOVER_ANALYSIS] Board: {board_width}x{board_height}mm")
    print(f"[LEFTOVER_ANALYSIS] Used area bounds: x={min_x:.1f}-{max_x:.1f}, y={min_y:.1f}-{max_y:.1f}")
    
    # Check for vertical strip on the right side of used parts
    vertical_strip_width = board_width - max_x - margin_right
    if vertical_strip_width >= 0.2 * board_width:
        # Check if there are any parts intersecting this vertical strip
        vertical_strip_x_start = max_x
        vertical_strip_filled = False
        
        print(f"[LEFTOVER_ANALYSIS] Checking vertical strip: width={vertical_strip_width:.1f} (min {0.2 * board_width:.1f}mm)")
        
        # Check if any parts are positioned in this vertical strip
        for part in fitted_parts:
            part_x = part.get('x', 0)
            part_width = part.get('width', 0)
            # Check if part is inside or overlaps the potential vertical strip
            if part_x >= vertical_strip_x_start:
                vertical_strip_filled = True
                break
        
        if not vertical_strip_filled:
            print(f"[LEFTOVER_ANALYSIS] Found vertical strip: width={vertical_strip_width:.1f}mm")
            return {
                'has_leftover_strip': True,
                'strip_type': 'vertical',
                'strip_position': vertical_strip_x_start,
                'strip_size': vertical_strip_width,
                'leftover_area_sq_mm': vertical_strip_width * board_height,
                'cut_line': {
                    'x': vertical_strip_x_start,
                    'orientation': 'vertical',
                    'length': board_height
                }
            }
    
    # Check for horizontal strip below used parts  
    horizontal_strip_height = board_height - max_y - margin_bottom
    if horizontal_strip_height >= 0.2 * board_height:
        # Check if there are any parts intersecting this horizontal strip
        horizontal_strip_y_start = max_y
        horizontal_strip_filled = False
        
        print(f"[LEFTOVER_ANALYSIS] Checking horizontal strip: height={horizontal_strip_height:.1f} (min {0.2 * board_height:.1f}mm)")
        
        # Check if any parts are positioned in this horizontal strip
        for part in fitted_parts:
            part_y = part.get('y', 0)
            part_height = part.get('height', 0)
            # Check if part is inside or overlaps the potential horizontal strip
            if part_y >= horizontal_strip_y_start:
                horizontal_strip_filled = True
                break
        
        if not horizontal_strip_filled:
            print(f"[LEFTOVER_ANALYSIS] Found horizontal strip: height={horizontal_strip_height:.1f}mm")
            return {
                'has_leftover_strip': True,
                'strip_type': 'horizontal',
                'strip_position': horizontal_strip_y_start,
                'strip_size': horizontal_strip_height,
                'leftover_area_sq_mm': horizontal_strip_height * board_width,
                'cut_line': {
                    'y': horizontal_strip_y_start,
                    'orientation': 'horizontal',
                    'length': board_width
                }
            }
    
    print(f"[LEFTOVER_ANALYSIS] No continuous strip found ≥20% of board dimension")
    return {
        'has_leftover_strip': False,
        'strip_type': None,
        'strip_position': None,
        'strip_size': 0,
        'leftover_area_sq_mm': 0,
        'cut_line': None
    }


def update_scrap_calculation_with_leftover(scrap_percentage, board_area, leftover_analysis):
    """
    Update scrap calculation by subtracting reusable leftover area.
    
    Args:
        scrap_percentage: Current scrap percentage (0.0-1.0)
        board_area: Total board area in square mm
        leftover_analysis: Result from analyze_reusable_leftover_strips
        
    Returns:
        dict: Updated scrap information
        {
            'scrap_percentage_original': float,
            'scrap_percentage_updated': float,
            'leftover_area': float,
            'leftover_percentage': float,
            'scrap_reduction': float
        }
    """
    
    if not leftover_analysis['has_leftover_strip']:
        return {
            'scrap_percentage_original': scrap_percentage,
            'scrap_percentage_updated': scrap_percentage,
            'leftover_area': 0,
            'leftover_percentage': 0,
            'scrap_reduction': 0
        }
    
    leftover_area = leftover_analysis['leftover_area_sq_mm']
    leftover_percentage = leftover_area / board_area if board_area > 0 else 0
    
    # Subtract leftover from scrap and ensure non-negative
    updated_scrap_percentage = max(0.0, scrap_percentage - leftover_percentage)
    scrap_reduction = scrap_percentage - updated_scrap_percentage
    
    return {
        'scrap_percentage_original': scrap_percentage,
        'scrap_percentage_updated': updated_scrap_percentage,
        'leftover_area': leftover_area,
        'leftover_percentage': leftover_percentage,
        'scrap_reduction': scrap_reduction
    }


def generate_svg_with_leftover_visualization(fitted_parts, board_specs, leftover_analysis):
    """
    Generate SVG layout showing nesting results with leftover strip highlighting and cut lines.
    
    Args:
        fitted_parts: List of fitted parts
        board_specs: Board specifications
        leftover_analysis: Result from analyze_reusable_leftover_strips
        
    Returns:
        str: SVG content with leftover visualization
    """
    
    board_width = board_specs.get('width_mm', 0)
    board_height = board_specs.get('height_mm', 0)
    
    # Scale factor for display (fit to reasonable viewport)
    scale = min(800 / board_width, 600 / board_height) if board_width > 0 and board_height > 0 else 1
    
    svg_width = board_width * scale
    svg_height = board_height * scale
    
    # Start building SVG
    svg_content = f'''<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {board_width} {board_height}" xmlns="http://www.w3.org/2000/svg">
    <!-- Board background -->
    <rect x="0" y="0" width="{board_width}" height="{board_height}" 
          fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
    
    <!-- Leftover strip highlighting -->
    {_generate_leftover_svg_content(leftover_analysis, board_width, board_height)}
    
    <!-- Margin guides -->
    <rect x="10" y="10" width="{board_width-20}" height="{board_height-20}" 
          fill="none" stroke="#6c757d" stroke-width="1" stroke-dasharray="5,5" opacity="0.5"/>
    
    <!-- Nested parts -->'''
    
    # Color palette for different parts
    colors = ['#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6f42c1', '#e83e8c', '#fd7e14', '#20c997', '#6610f2']
    
    for i, part in enumerate(fitted_parts):
        color = colors[int(part.get('part_id', 0)) % len(colors)]
        x = part.get('x', 0)
        y = part.get('y', 0)
        width = part.get('width', 0)
        height = part.get('height', 0)
        rotation = part.get('rotation', 0)
        
        # Create transform for rotation
        transform = f"translate({x + width/2}, {y + height/2}) rotate({rotation}) translate({-width/2}, {-height/2})" if rotation != 0 else f"translate({x}, {y})"
        
        svg_content += f'''
    <g transform="{transform}">
        <rect x="0" y="0" width="{width}" height="{height}" 
              fill="{color}" stroke="#fff" stroke-width="1" opacity="0.7"/>
        <text x="{width/2}" y="{height/2}" text-anchor="middle" dominant-baseline="middle" 
              font-size="{min(width, height) * 0.1}" fill="#fff" font-weight="bold">
              {part.get('part_id', '')}-{part.get('instance', '')}
        </text>
    </g>'''
    
    # Add cut line if there's a reusable strip
    if leftover_analysis['has_leftover_strip'] and leftover_analysis['cut_line']:
        cut_line = leftover_analysis['cut_line']
        if cut_line['orientation'] == 'vertical':
            x = cut_line['x']
            svg_content += f'''
    <!-- Cut line for vertical strip -->
    <line x1="{x}" y1="0" x2="{x}" y2="{board_height}" 
          stroke="#dc3545" stroke-width="3" stroke-dasharray="5,5" opacity="0.8">
        <title>Cut line for vertical leftover strip</title>
    </line>'''
        else:  # horizontal
            y = cut_line['y']
            svg_content += f'''
    <!-- Cut line for horizontal strip -->
    <line x1="0" y1="{y}" x2="{board_width}" y2="{y}" 
          stroke="#dc3545" stroke-width="3" stroke-dasharray="5,5" opacity="0.8">
        <title>Cut line for horizontal leftover strip</title>
    </line>'''
    
    svg_content += '''
</svg>'''
    
    return svg_content


def _generate_leftover_svg_content(leftover_analysis, board_width, board_height):
    """
    Generate SVG content for highlighting leftover strips.
    
    Args:
        leftover_analysis: Result from analyze_reusable_leftover_strips
        board_width: Board width in mm
        board_height: Board height in mm
        
    Returns:
        str: SVG content for leftover strip highlighting
    """
    
    if not leftover_analysis['has_leftover_strip']:
        return ""
    
    strip_type = leftover_analysis['strip_type']
    strip_position = leftover_analysis['strip_position']
    strip_size = leftover_analysis['strip_size']
    
    if strip_type == 'vertical':
        # Highlight vertical strip on the right
        x = strip_position
        width = strip_size
        height = board_height
        return f'''
    <!-- Leftover vertical strip -->
    <rect x="{x}" y="0" width="{width}" height="{height}" 
          fill="#dc3545" fill-opacity="0.3" stroke="#dc3545" stroke-width="2" stroke-dasharray="3,3">
        <title>Reusable leftover strip (width: {strip_size:.1f}mm)</title>
    </rect>'''
    
    elif strip_type == 'horizontal':
        # Highlight horizontal strip at the bottom
        y = strip_position
        height = strip_size
        width = board_width
        return f'''
    <!-- Leftover horizontal strip -->
    <rect x="0" y="{y}" width="{width}" height="{height}" 
          fill="#dc3545" fill-opacity="0.3" stroke="#dc3545" stroke-width="2" stroke-dasharray="3,3">
        <title>Reusable leftover strip (height: {strip_size:.1f}mm)</title>
    </rect>'''
    
    return ""


# Test/Example function
def test_leftover_analysis():
    """
    Test function to demonstrate leftover strip analysis.
    """
    # Example test data
    fitted_parts = [
        {'x': 10, 'y': 10, 'width': 100, 'height': 50, 'part_id': 1, 'instance': 1},
        {'x': 10, 'y': 70, 'width': 100, 'height': 50, 'part_id': 2, 'instance': 1},
        {'x': 10, 'y': 130, 'width': 100, 'height': 50, 'part_id': 3, 'instance': 1},
    ]
    
    board_specs = {
        'width_mm': 1000,
        'height_mm': 2000,
    }
    
    nesting_config = {
        'sheet_margin_mm': {'top': 10, 'right': 10, 'bottom': 10, 'left': 10}
    }
    
    analysis = analyze_reusable_leftover_strips(fitted_parts, board_specs, nesting_config)
    print("Analysis result:", analysis)
    
    return analysis


if __name__ == "__main__":
    # Run test if script is executed directly
    test_leftover_analysis()
