"""
Nesting Visualizer
Creates SVG visualizations of nesting results with leftover strips and cut lines
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class VisualizationResult:
    """Result of visualization generation"""
    success: bool
    used_area_svg: str = ""
    leftover_scrap_svg: str = ""
    combined_svg: str = ""
    legend_svg: str = ""
    error_message: str = ""


class NestingVisualizer:
    """Creates visualizations for nesting results"""
    
    def __init__(self):
        self.colors = {
            'used_area': '#28a745',
            'leftover_area': '#dc3545',
            'scrap_area': '#6c757d',
            'cut_line': '#ffc107',
            'background': '#f8f9fa'
        }
    
    def create_visualization(self, board_analysis, fitted_parts: List[Dict] = None) -> VisualizationResult:
        """Create visualization for a board analysis"""
        try:
            svg_content = self._generate_board_svg(board_analysis, fitted_parts or [])
            
            return VisualizationResult(
                success=True,
                combined_svg=svg_content,
                used_area_svg="",  # Simplified for now
                leftover_scrap_svg="",
                legend_svg=self._generate_legend()
            )
        except Exception as e:
            return VisualizationResult(
                success=False,
                error_message=str(e)
            )
    
    def _generate_board_svg(self, board_analysis, fitted_parts: List[Dict]) -> str:
        """Generate SVG for the board showing parts, leftovers, and cut lines"""
        
        board_width = board_analysis.board_width_mm
        board_height = board_analysis.board_height_mm
        
        # Scale factor for display
        scale = min(800 / board_width, 600 / board_height) if board_width > 0 and board_height > 0 else 1
        svg_width = board_width * scale
        svg_height = board_height * scale
        
        svg_content = f'''
        <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {board_width} {board_height}" xmlns="http://www.w3.org/2000/svg">
            <!-- Board background -->
            <rect x="0" y="0" width="{board_width}" height="{board_height}" 
                  fill="{self.colors['background']}" stroke="#dee2e6" stroke-width="2" />
            
            <!-- Leftover strip highlighting -->
            {self._generate_leftover_overlay(board_analysis)}
            
            <!-- Cut lines -->
            {self._generate_cut_lines(board_analysis)}
            
            <!-- Parts -->
            {self._generate_parts_overlay(fitted_parts, scale)}
            
            <!-- Legend -->
            {self._generate_inline_legend(board_width, board_height)}
        </svg>'''
        
        return svg_content
    
    def _generate_leftover_overlay(self, board_analysis) -> str:
        """Generate leftover strip overlay"""
        if not board_analysis.meets_threshold or board_analysis.leftover_area_sq_mm <= 0:
            return ""
        
        # This would be enhanced based on the cut_line data from analysis
        cut_line = board_analysis.cut_line
        
        if cut_line and cut_line.get('orientation') == 'vertical':
            # Vertical leftover strip
            return f'''
            <rect x="{cut_line['x']}" y="0" width="{board_analysis.board_width_mm - cut_line['x']}" height="{board_analysis.board_height_mm}" 
                  fill="{self.colors['leftover_area']}" fill-opacity="0.3" stroke="{self.colors['leftover_area']}" stroke-width="2" stroke-dasharray="5,5">
                <title>Reusable Leftover: {board_analysis.leftover_area_sq_mm:.0f} sq mm</title>
            </rect>'''
        
        elif cut_line and cut_line.get('orientation') == 'horizontal':
            # Horizontal leftover strip
            return f'''
            <rect x="0" y="{cut_line['y']}" width="{board_analysis.board_width_mm}" height="{board_analysis.board_height_mm - cut_line['y']}" 
                  fill="{self.colors['leftover_area']}" fill-opacity="0.3" stroke="{self.colors['leftover_area']}" stroke-width="2" stroke-dasharray="5,5">
                <title>Reusable Leftover: {board_analysis.leftover_area_sq_mm:.0f} sq mm</title>
            </rect>'''
        
        return ""
    
    def _generate_cut_lines(self, board_analysis) -> str:
        """Generate cut lines for reusable strips"""
        if not board_analysis.cut_line:
            return ""
        
        cut_line = board_analysis.cut_line
        
        if cut_line.get('orientation') == 'vertical':
            x = cut_line['x']
            return f'''
            <line x1="{x}" y1="0" x2="{x}" y2="{board_analysis.board_height_mm}" 
                  stroke="{self.colors['cut_line']}" stroke-width="4" stroke-dasharray="10,5" opacity="0.8">
                <title>Cut Line - Vertical Leftover</title>
            </line>'''
        
        elif cut_line.get('orientation') == 'horizontal':
            y = cut_line['y']
            return f'''
            <line x1="0" y1="{y}" x2="{board_analysis.board_width_mm}" y2="{y}" 
                  stroke="{self.colors['cut_line']}" stroke-width="4" stroke-dasharray="10,5" opacity="0.8">
                <title>Cut Line - Horizontal Leftover</title>
            </line>'''
        
        return ""
    
    def _generate_parts_overlay(self, fitted_parts: List[Dict], scale: float) -> str:
        """Generate overlay for nested parts"""
        if not fitted_parts:
            return ""
        
        colors = ['#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6f42c1', '#e83e8c', '#fd7e14']
        part_svgs = []
        
        for i, part in enumerate(fitted_parts):
            color = colors[i % len(colors)]
            x = part.get('x', 0)
            y = part.get('y', 0)
            width = part.get('width_mm', part.get('width', 0))
            height = part.get('height_mm', part.get('height', 0))
            
            part_svgs.append(f'''
            <g>
                <rect x="{x}" y="{y}" width="{width}" height="{height}" 
                      fill="{color}" stroke="#fff" stroke-width="1" opacity="0.7"/>
                <text x="{x + width/2}" y="{y + height/2}" text-anchor="middle" dominant-baseline="middle" 
                      font-size="{min(width, height) * 0.1}" fill="#fff" font-weight="bold">
                    {part.get('id', f'P{i+1}')}
                </text>
            </g>''')
        
        return '\n'.join(part_svgs)
    
    def _generate_inline_legend(self, board_width: float, board_height: float) -> str:
        """Generate inline legend"""
        legend_x = board_width - 200
        legend_y = 20
        
        return f'''
        <g id="legend" transform="translate({legend_x}, {legend_y})">
            <rect x="0" y="0" width="180" height="100" fill="white" stroke="#ccc" stroke-width="1" opacity="0.9"/>
            <text x="10" y="20" font-size="14" font-weight="bold" fill="#333">Legend</text>
            
            <rect x="10" y="30" width="15" height="15" fill="{self.colors['used_area']}" opacity="0.7"/>
            <text x="30" y="42" font-size="10" fill="#333">Used Area</text>
            
            <rect x="10" y="50" width="15" height="15" fill="{self.colors['leftover_area']}" stroke="{self.colors['leftover_area']}" stroke-width="1" stroke-dasharray="2,2" fill-opacity="0.3"/>
            <text x="30" y="62" font-size="10" fill="#333">Leftover Strip</text>
            
            <line x1="10" y1="70" x2="25" y2="70" stroke="{self.colors['cut_line']}" stroke-width="3" stroke-dasharray="5,3"/>
            <text x="30" y="75" font-size="10" fill="#333">Cut Line</text>
        </g>'''
    
    def _generate_legend(self) -> str:
        """Generate standalone legend"""
        return """
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #28a745;"></div>
                <span>Used Area</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #dc3545; opacity: 0.3; border: 1px dashed #dc3545;"></div>
                <span>Leftover Strip</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ffc107; height: 3px; border-radius: 2px;"></div>
                <span>Cut Line</span>
            </div>
        </div>"""
