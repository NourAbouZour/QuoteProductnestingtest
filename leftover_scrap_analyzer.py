"""
Leftover Scrap Analyzer for multiple boards
Integrates with the complete_nesting_api architecture
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class NestingResult:
    """Nesting result data structure"""
    board_id: str
    board_width_mm: float
    board_height_mm: float
    placed_parts: List[Dict[str, Any]]


@dataclass 
class BoardAnalysis:
    """Board analysis results"""
    board_id: str
    board_width_mm: float
    board_height_mm: float
    used_area_sq_mm: float
    leftover_area_sq_mm: float
    scrap_area_sq_mm: float
    utilization_percent: float
    leftover_percent: float
    scrap_percent: float
    meets_threshold: bool
    pockets: List[Dict[str, Any]]
    cut_line: Dict[str, Any] or None
    total_parts: int


class LeftoverScrapAnalyzer:
    """Analyzes leftover and scrap areas for multiple boards"""
    
    def __init__(self):
        self.leftover_threshold_percent = 20.0  # 20% minimum to qualify as leftover
        self.min_leftover_width_mm = 100.0
        self.min_leftover_height_mm = 100.0
    
    def analyze_multiple_boards(self, analysis_results: List[NestingResult]) -> List[BoardAnalysis]:
        """Analyze each board for leftover strips"""
        board_analyses = []
        
        for result in analysis_results:
            analysis = self.analyze_single_board(result)
            board_analyses.append(analysis)
            
        return board_analyses
    
    def analyze_single_board(self, result: NestingResult) -> BoardAnalysis:
        """Analyze a single board for leftover strips and scrap areas"""
        
        board_width = result.board_width_mm
        board_height = result.board_height_mm
        board_area = board_width * board_height
        
        if not result.placed_parts:
            # Empty board case
            return BoardAnalysis(
                board_id=result.board_id,
                board_width_mm=board_width,
                board_height_mm=board_height,
                used_area_sq_mm=0.0,
                leftover_area_sq_mm=0.0,
                scrap_area_sq_mm=board_area,
                utilization_percent=0.0,
                leftover_percent=0.0,
                scrap_percent=100.0,
                meets_threshold=False,
                pockets=[],
                cut_line=None,
                total_parts=0
            )
        
        # Find bounds of used space
        min_x = min(p.get('x', 0) for p in result.placed_parts)
        max_x = max((p.get('x', 0) + p.get('width_mm', 0)) for p in result.placed_parts)
        min_y = min(p.get('y', 0) for p in result.placed_parts)
        max_y = max((p.get('y', 0) + p.get('height_mm', 0)) for p in result.placed_parts)
        
        # Calculate used area
        used_area = sum(
            p.get('width_mm', 0) * p.get('height_mm', 0) 
            for p in result.placed_parts
        )
        
        # Analyze for leftover strips using the leftover strip analyzer logic
        leftover_analysis = self._analyze_leftover_strips(result)
        
        scrap_area = board_area - used_area
        
        # If leftover strip found, subtract it from scrap
        if leftover_analysis['has_leftover_strip']:
            leftover_area = leftover_analysis['leftover_area_sq_mm']
            scrap_area = max(0, scrap_area - leftover_area)
        else:
            leftover_area = 0.0
        
        # Calculate percentages
        utilization_percent = (used_area / board_area) * 100 if board_area > 0 else 0
        leftover_percent = (leftover_area / board_area) * 100 if board_area > 0 else 0
        scrap_percent = ((board_area - used_area - leftover_area) / board_area) * 100 if board_area > 0 else 0
        
        # Check if meets threshold
        meets_threshold = (
            leftover_percent >= self.leftover_threshold_percent or
            (leftover_analysis.get('strip_size', 0) >= min(0.2 * result.board_width_mm, 
                                                         0.2 * result.board_height_mm))
        )
        
        # Find pockets (small unused areas)
        pockets = self._find_pockets(result, board_width, board_height)
        
        return BoardAnalysis(
            board_id=result.board_id,
            board_width_mm=board_width,
            board_height_mm=board_height,
            used_area_sq_mm=used_area,
            leftover_area_sq_mm=leftover_area,
            scrap_area_sq_mm=max(0, scrap_area),
            utilization_percent=utilization_percent,
            leftover_percent=leftover_percent,
            scrap_percent=max(0, scrap_percent),
            meets_threshold=meets_threshold,
            pockets=pockets,
            cut_line=leftover_analysis.get('cut_line'),
            total_parts=len(result.placed_parts)
        )
    
    def _analyze_leftover_strips(self, result: NestingResult) -> Dict[str, Any]:
        """Analyze for reusable leftover strips (≥20% of board dimension)"""
        
        board_width = result.board_width_mm
        board_height = result.board_height_mm
        
        if not result.placed_parts:
            return self._empty_result()
        
        # Find bounds of used space
        max_x = max((p.get('x', 0) + p.get('width_mm', 0)) for p in result.placed_parts)
        max_y = max((p.get('y', 0) + p.get('height_mm', 0)) for p in result.placed_parts)
        
        # Check for vertical strip on the right
        vertical_strip_width = board_width - max_x
        if vertical_strip_width >= 0.2 * board_width:
            # Check if any parts intersect this strip
            vertical_strip_filled = any(
                p.get('x', 0) >= max_x for p in result.placed_parts
            )
            
            if not vertical_strip_filled:
                return {
                    'has_leftover_strip': True,
                    'strip_type': 'vertical',
                    'strip_position': max_x,
                    'strip_size': vertical_strip_width,
                    'leftover_area_sq_mm': vertical_strip_width * board_height,
                    'cut_line': {
                        'x': max_x,
                        'orientation': 'vertical',
                        'length': board_height,
                        'position': max_x,
                        'type': 'vertical_cut'
                    }
                }
        
        # Check for horizontal strip at the bottom
        horizontal_strip_height = board_height - max_y
        if horizontal_strip_height >= 0.2 * board_height:
            # Check if any parts intersect this strip
            horizontal_strip_filled = any(
                p.get('y', 0) >= max_y for p in result.placed_parts
            )
            
            if not horizontal_strip_filled:
                return {
                    'has_leftover_strip': True,
                    'strip_type': 'horizontal',
                    'strip_position': max_y,
                    'strip_size': horizontal_strip_height,
                    'leftover_area_sq_mm': horizontal_strip_height * board_width,
                    'cut_line': {
                        'y': max_y,
                        'orientation': 'horizontal',
                        'length': board_width,
                        'position': max_y,
                        'type': 'horizontal_cut'
                    }
                }
        
        return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty analysis result"""
        return {
            'has_leftover_strip': False,
            'strip_type': None,
            'strip_position': None,
            'strip_size': 0,
            'leftover_area_sq_mm': 0,
            'cut_line': None
        }
    
    def _find_pockets(self, result: NestingResult, board_width: float, board_height: float) -> List[Dict[str, Any]]:
        """Find small unused areas (pockets) that can't be reutilized easily"""
        # For now, return empty pockets list
        # This could be enhanced to detect actual pocket areas
        return []
    
    def calculate_totals(self, board_analyses: List[BoardAnalysis]) -> Dict[str, Any]:
        """Calculate totals across all boards"""
        total_boards = len(board_analyses)
        total_boards_with_leftovers = sum(1 for analysis in board_analyses if analysis.leftover_area_sq_mm > 0)
        
        # Calculate weighted totals
        total_board_area = sum(analysis.board_width_mm * analysis.board_height_mm for analysis in board_analyses)
        total_used_area = sum(analysis.used_area_sq_mm for analysis in board_analyses)
        total_leftover_area = sum(analysis.leftover_area_sq_mm for analysis in board_analyses)
        total_scrap_area = sum(analysis.scrap_area_sq_mm for analysis in board_analyses)
        
        # Calculate percentages
        overall_utilization_percent = (total_used_area / total_board_area * 100) if total_board_area > 0 else 0
        total_leftover_percent = (total_leftover_area / total_board_area * 100) if total_board_area > 0 else 0
        total_reported_scrap_percent = (total_scrap_area / total_board_area * 100) if total_board_area > 0 else 0
        
        return {
            'total_boards': total_boards,
            'total_boards_with_leftovers': total_boards_with_leftovers,
            'total_utilization_percent': overall_utilization_percent,
            'total_leftover_percent': total_leftover_percent,
            'total_reported_scrap_percent': total_reported_scrap_percent,
            'total_board_area_sq_mm': total_board_area,
            'total_used_area_sq_mm': total_used_area,
            'total_leftover_area_sq_mm': total_leftover_area,
            'total_scrap_area_sq_mm': total_scrap_area
        }
