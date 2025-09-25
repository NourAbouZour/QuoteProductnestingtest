#!/usr/bin/env python3
"""
Enhanced Nesting Algorithm
Implements improved nesting strategy with proper board and part sorting,
scrap factor calculation, and optimized board selection.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class NestingResult:
    """Result of nesting operation for a single board"""
    board_id: int
    board_dimensions: Dict[str, float]  # length_mm, width_mm, area_sq_mm
    nested_parts: List[Dict[str, Any]]  # Parts successfully nested on this board
    utilization_percentage: float  # Percentage of board area used
    scrap_percentage: float  # Percentage of board area that becomes scrap
    scrap_area_sq_mm: float  # Actual scrap area in square millimeters
    parts_fitted: int  # Total number of individual parts fitted
    efficiency_score: float  # Overall efficiency score (0-1)

@dataclass
class NestingConfiguration:
    """Configuration for nesting algorithm"""
    min_part_gap_mm: float = 5.0
    kerf_mm: float = 0.2
    margin_mm: float = 10.0
    rotation_allowed: bool = True
    rotation_step_deg: float = 90.0
    optimize_for_material_usage: bool = True
    max_scrap_threshold: float = 0.75  # 75% maximum acceptable scrap (increased for large boards)

class EnhancedNestingAlgorithm:
    """
    Enhanced nesting algorithm that prioritizes:
    1. Largest boards first
    2. Largest parts first, scaling down
    3. Accurate scrap factor calculation
    4. Optimal material utilization
    """
    
    def __init__(self, config: NestingConfiguration = None):
        self.config = config or NestingConfiguration()
        self.logger = logging.getLogger(__name__)
    
    def calculate_scrap_factor_for_board(self, board: Dict[str, Any], nested_parts: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calculate scrap factor and scrap area for a specific board with nested parts.
        
        Args:
            board: Board dictionary with dimensions
            nested_parts: List of parts that will be nested on this board
            
        Returns:
            Tuple of (scrap_percentage, scrap_area_sq_mm)
        """
        try:
            board_area = board.get('area_sq_mm', board.get('length_mm', 0) * board.get('width_mm', 0))
            
            # Calculate total area of all parts (including quantities)
            parts_total_area = 0.0
            for part in nested_parts:
                part_area = part.get('area_sq_mm', 0)
                if part_area <= 0:
                    # Fallback to length x width calculation
                    part_area = part.get('length_mm', 0) * part.get('width_mm', 0)
                
                quantity = part.get('quantity', 1)
                parts_total_area += part_area * quantity
            
            # Add kerf losses (cutting width losses)
            kerf_loss_area = self._calculate_kerf_losses(board, nested_parts)
            
            # Calculate effective used area (parts + kerf losses)
            used_area = parts_total_area + kerf_loss_area
            
            # Calculate scrap area and percentage
            scrap_area = max(0, board_area - used_area)
            scrap_percentage = (scrap_area / board_area) if board_area > 0 else 0.0
            
            self.logger.info(f"Board {board.get('id', 'unknown')}: "
                           f"Total area: {board_area:.0f} sq mm, "
                           f"Parts area: {parts_total_area:.0f} sq mm, "
                           f"Kerf losses: {kerf_loss_area:.0f} sq mm, "
                           f"Scrap: {scrap_area:.0f} sq mm ({scrap_percentage:.1%})")
            
            return scrap_percentage, scrap_area
            
        except Exception as e:
            self.logger.error(f"Error calculating scrap factor: {e}")
            return 0.3, 0.0  # Conservative fallback
    
    def _calculate_kerf_losses(self, board: Dict[str, Any], nested_parts: List[Dict[str, Any]]) -> float:
        """
        Calculate area lost due to kerf (cutting width).
        This is an approximation based on part perimeters and cutting patterns.
        """
        try:
            total_kerf_area = 0.0
            kerf_width = self.config.kerf_mm
            
            for part in nested_parts:
                # Approximate perimeter calculation
                length = part.get('length_mm', 0)
                width = part.get('width_mm', 0)
                perimeter = 2 * (length + width)
                quantity = part.get('quantity', 1)
                
                # Kerf area = perimeter * kerf_width * quantity
                part_kerf_area = perimeter * kerf_width * quantity
                total_kerf_area += part_kerf_area
            
            return total_kerf_area
            
        except Exception as e:
            self.logger.error(f"Error calculating kerf losses: {e}")
            return 0.0
    
    def sort_parts_by_priority(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort parts by nesting priority: largest parts first, considering quantity.
        
        Args:
            parts: List of part dictionaries
            
        Returns:
            Sorted list of parts (largest first)
        """
        def part_priority_key(part):
            # Calculate part area
            area = part.get('area_sq_mm', 0)
            if area <= 0:
                area = part.get('length_mm', 0) * part.get('width_mm', 0)
            
            # Calculate total area considering quantity
            quantity = part.get('quantity', 1)
            total_area = area * quantity
            
            # Priority: larger total area first, then larger individual area
            return (-total_area, -area, -max(part.get('length_mm', 0), part.get('width_mm', 0)))
        
        sorted_parts = sorted(parts, key=part_priority_key)
        
        self.logger.info(f"Sorted {len(sorted_parts)} parts by priority (largest first)")
        for i, part in enumerate(sorted_parts[:5]):  # Log first 5 for debugging
            area = part.get('area_sq_mm', part.get('length_mm', 0) * part.get('width_mm', 0))
            qty = part.get('quantity', 1)
            self.logger.debug(f"  {i+1}. Part {part.get('id', 'unknown')}: "
                            f"{area:.0f} sq mm × {qty} = {area * qty:.0f} sq mm total")
        
        return sorted_parts
    
    def sort_boards_by_priority(self, boards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort boards by nesting priority: largest boards first.
        
        Args:
            boards: List of board dictionaries
            
        Returns:
            Sorted list of boards (largest first)
        """
        def board_priority_key(board):
            # Calculate board area
            area = board.get('area_sq_mm', 0)
            if area <= 0:
                area = board.get('length_mm', 0) * board.get('width_mm', 0)
            
            # Priority: larger area first, then larger max dimension
            max_dim = max(board.get('length_mm', 0), board.get('width_mm', 0))
            return (-area, -max_dim)
        
        sorted_boards = sorted(boards, key=board_priority_key)
        
        self.logger.info(f"Sorted {len(sorted_boards)} boards by priority (largest first)")
        for i, board in enumerate(sorted_boards[:3]):  # Log first 3 for debugging
            area = board.get('area_sq_mm', board.get('length_mm', 0) * board.get('width_mm', 0))
            qty = board.get('quantity', 0)
            self.logger.debug(f"  {i+1}. Board {board.get('id', 'unknown')}: "
                            f"{board.get('length_mm', 0):.0f}×{board.get('width_mm', 0):.0f}mm "
                            f"(area: {area:.0f} sq mm) qty: {qty}")
        
        return sorted_boards
    
    def can_part_fit_on_board(self, part: Dict[str, Any], board: Dict[str, Any], 
                             consider_rotation: bool = None) -> bool:
        """
        Check if a part can fit on a board, considering margins and rotation.
        
        Args:
            part: Part dictionary with dimensions
            board: Board dictionary with dimensions
            consider_rotation: Whether to consider rotating the part
            
        Returns:
            True if part can fit, False otherwise
        """
        if consider_rotation is None:
            consider_rotation = self.config.rotation_allowed
        
        # Get dimensions
        part_length = part.get('length_mm', 0)
        part_width = part.get('width_mm', 0)
        board_length = board.get('length_mm', 0)
        board_width = board.get('width_mm', 0)
        
        # Account for margins
        effective_board_length = board_length - (2 * self.config.margin_mm)
        effective_board_width = board_width - (2 * self.config.margin_mm)
        
        # Check if part fits without rotation
        fits_normal = (part_length <= effective_board_length and 
                      part_width <= effective_board_width)
        
        if fits_normal:
            return True
        
        # Check if part fits with rotation (if allowed)
        if consider_rotation:
            fits_rotated = (part_width <= effective_board_length and 
                           part_length <= effective_board_width)
            return fits_rotated
        
        return False
    
    def calculate_board_utilization(self, board: Dict[str, Any], nested_parts: List[Dict[str, Any]]) -> float:
        """
        Calculate how efficiently a board is utilized.
        
        Args:
            board: Board dictionary
            nested_parts: Parts nested on this board
            
        Returns:
            Utilization percentage (0.0 to 1.0)
        """
        board_area = board.get('area_sq_mm', board.get('length_mm', 0) * board.get('width_mm', 0))
        
        if board_area <= 0:
            return 0.0
        
        parts_area = 0.0
        for part in nested_parts:
            part_area = part.get('area_sq_mm', 0)
            if part_area <= 0:
                part_area = part.get('length_mm', 0) * part.get('width_mm', 0)
            
            quantity = part.get('quantity', 1)
            parts_area += part_area * quantity
        
        utilization = parts_area / board_area
        return min(1.0, utilization)  # Cap at 100%
    
    def select_optimal_boards_for_parts(self, boards: List[Dict[str, Any]], 
                                       parts: List[Dict[str, Any]]) -> List[NestingResult]:
        """
        Select optimal boards for nesting parts with minimal scrap.
        
        Args:
            boards: Available boards (should be pre-sorted by size)
            parts: Parts to be nested (will be sorted by priority)
            
        Returns:
            List of NestingResult objects with optimal board selections
        """
        # Sort inputs for optimal nesting
        sorted_boards = self.sort_boards_by_priority(boards)
        sorted_parts = self.sort_parts_by_priority(parts)
        
        nesting_results = []
        remaining_parts = sorted_parts.copy()
        
        self.logger.info(f"Starting nesting optimization for {len(remaining_parts)} parts on {len(sorted_boards)} boards")
        
        # Try to nest parts on boards, starting with largest boards
        for board in sorted_boards:
            if not remaining_parts:
                break
            
            board_qty = board.get('quantity', 0)
            if board_qty <= 0:
                continue
            
            # For each available board of this type
            for _ in range(board_qty):
                if not remaining_parts:
                    break
                
                # Find parts that can fit on this board
                nested_parts = []
                parts_to_remove = []
                
                for part in remaining_parts:
                    if self.can_part_fit_on_board(part, board):
                        nested_parts.append(part.copy())
                        parts_to_remove.append(part)
                
                if nested_parts:
                    # Calculate scrap factor for this board configuration
                    scrap_percentage, scrap_area = self.calculate_scrap_factor_for_board(board, nested_parts)
                    
                    # Only use this board if scrap is within acceptable limits
                    if scrap_percentage <= self.config.max_scrap_threshold:
                        utilization = self.calculate_board_utilization(board, nested_parts)
                        efficiency_score = utilization * (1.0 - scrap_percentage)
                        
                        result = NestingResult(
                            board_id=board.get('id', 0),
                            board_dimensions={
                                'length_mm': board.get('length_mm', 0),
                                'width_mm': board.get('width_mm', 0),
                                'area_sq_mm': board.get('area_sq_mm', 0)
                            },
                            nested_parts=nested_parts,
                            utilization_percentage=utilization,
                            scrap_percentage=scrap_percentage,
                            scrap_area_sq_mm=scrap_area,
                            parts_fitted=sum(part.get('quantity', 1) for part in nested_parts),
                            efficiency_score=efficiency_score
                        )
                        
                        nesting_results.append(result)
                        
                        # Remove nested parts from remaining parts
                        for part in parts_to_remove:
                            if part in remaining_parts:
                                remaining_parts.remove(part)
                        
                        self.logger.info(f"Board {board.get('id')} nesting: "
                                       f"{len(nested_parts)} part types, "
                                       f"{result.parts_fitted} total pieces, "
                                       f"utilization: {utilization:.1%}, "
                                       f"scrap: {scrap_percentage:.1%}")
                    else:
                        self.logger.warning(f"Board {board.get('id')} rejected: "
                                          f"scrap {scrap_percentage:.1%} exceeds threshold {self.config.max_scrap_threshold:.1%}")
        
        # Log summary
        total_boards_used = len(nesting_results)
        total_parts_fitted = sum(result.parts_fitted for result in nesting_results)
        average_utilization = sum(result.utilization_percentage for result in nesting_results) / len(nesting_results) if nesting_results else 0
        average_scrap = sum(result.scrap_percentage for result in nesting_results) / len(nesting_results) if nesting_results else 0
        
        self.logger.info(f"Nesting optimization complete: "
                        f"{total_boards_used} boards used, "
                        f"{total_parts_fitted} parts fitted, "
                        f"avg utilization: {average_utilization:.1%}, "
                        f"avg scrap: {average_scrap:.1%}")
        
        if remaining_parts:
            self.logger.warning(f"{len(remaining_parts)} parts could not be nested due to size or scrap constraints")
        
        return nesting_results

# Example usage and integration functions
def integrate_with_existing_system(parts_data: List[Dict[str, Any]], 
                                  boards_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Integration function for the existing system.
    
    Args:
        parts_data: List of parts from the existing system
        boards_data: List of boards from the existing system
        
    Returns:
        Dictionary with nesting results and scrap information
    """
    # Initialize enhanced nesting algorithm
    config = NestingConfiguration(
        min_part_gap_mm=5.0,
        kerf_mm=0.2,
        margin_mm=10.0,
        rotation_allowed=True,
        max_scrap_threshold=0.75  # 75% max scrap (increased for large boards)
    )
    
    nesting_algo = EnhancedNestingAlgorithm(config)
    
    # Perform nesting optimization
    nesting_results = nesting_algo.select_optimal_boards_for_parts(boards_data, parts_data)
    
    # Format results for existing system
    results = {
        'nesting_results': [],
        'total_boards_used': len(nesting_results),
        'total_scrap_percentage': 0.0,
        'total_scrap_area_sq_mm': 0.0,
        'average_utilization': 0.0,
        'efficiency_score': 0.0
    }
    
    if nesting_results:
        total_scrap_area = sum(result.scrap_area_sq_mm for result in nesting_results)
        total_board_area = sum(result.board_dimensions['area_sq_mm'] for result in nesting_results)
        
        results['total_scrap_percentage'] = (total_scrap_area / total_board_area) if total_board_area > 0 else 0.0
        results['total_scrap_area_sq_mm'] = total_scrap_area
        results['average_utilization'] = sum(result.utilization_percentage for result in nesting_results) / len(nesting_results)
        results['efficiency_score'] = sum(result.efficiency_score for result in nesting_results) / len(nesting_results)
        
        for result in nesting_results:
            results['nesting_results'].append({
                'board_id': result.board_id,
                'board_dimensions': result.board_dimensions,
                'nested_parts_count': len(result.nested_parts),
                'total_parts_fitted': result.parts_fitted,
                'utilization_percentage': result.utilization_percentage,
                'scrap_percentage': result.scrap_percentage,
                'scrap_area_sq_mm': result.scrap_area_sq_mm,
                'efficiency_score': result.efficiency_score
            })
    
    return results
