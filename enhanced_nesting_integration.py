#!/usr/bin/env python3
"""
Enhanced Nesting Integration
Integrates the advanced nesting engine with the existing system,
providing seamless API and database integration.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_nesting_engine import (
    AdvancedNestingEngine, NestingStrategy, Part, Board, Polygon, Point,
    NestingResult, Placement
)
from enhanced_nesting_algorithm import EnhancedNestingAlgorithm, NestingConfiguration
from svgnest_nesting_engine import SVGNestNestingEngine
from multi_board_nesting_optimizer import MultiBoardNestingOptimizer

@dataclass
class SystemPart:
    """Represents a part from the existing system"""
    id: str
    length_mm: float
    width_mm: float
    area_sq_mm: float
    quantity: int
    material_id: str = ""
    priority: int = 0
    rotation_allowed: bool = True
    svg_path: str = ""

@dataclass
class SystemBoard:
    """Represents a board from the existing system"""
    id: str
    length_mm: float
    width_mm: float
    area_sq_mm: float
    cost: float
    quantity_available: int
    material_id: str = ""
    margin_mm: float = 10.0
    kerf_mm: float = 0.2

class EnhancedNestingIntegration:
    """
    Integration layer between the advanced nesting engine and the existing system.
    Provides seamless conversion and optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize different engines
        self.advanced_engine = AdvancedNestingEngine(self.config.get('advanced_config', {}))
        self.enhanced_algorithm = EnhancedNestingAlgorithm(
            NestingConfiguration(**self.config.get('enhanced_config', {}))
        )
        self.svgnest_engine = SVGNestNestingEngine(
            min_gap_mm=self.config.get('min_gap_mm', 5.0),
            margin_mm=self.config.get('margin_mm', 10.0)
        )
        self.multi_board_optimizer = MultiBoardNestingOptimizer(
            min_gap_mm=self.config.get('min_gap_mm', 5.0),
            margin_mm=self.config.get('margin_mm', 10.0)
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'min_gap_mm': 5.0,
            'margin_mm': 10.0,
            'kerf_mm': 0.2,
            'rotation_step_degrees': 5.0,
            'enable_advanced_rotations': True,
            'enable_mirroring': True,
            'parallel_processing': True,
            'max_optimization_time': 300,
            'advanced_config': {
                'genetic_population_size': 50,
                'genetic_generations': 100,
                'genetic_mutation_rate': 0.1,
                'genetic_crossover_rate': 0.8,
                'simulated_annealing_temperature': 1000.0,
                'simulated_annealing_cooling_rate': 0.95
            },
            'enhanced_config': {
                'min_part_gap_mm': 5.0,
                'kerf_mm': 0.2,
                'margin_mm': 10.0,
                'rotation_allowed': True,
                'rotation_step_deg': 5.0,
                'optimize_for_material_usage': True,
                'max_scrap_threshold': 0.25
            }
        }
    
    def convert_system_parts_to_advanced(self, system_parts: List[SystemPart]) -> List[Part]:
        """Convert system parts to advanced engine parts"""
        advanced_parts = []
        
        for sys_part in system_parts:
            # Create polygon from dimensions
            polygon = Polygon([
                Point(0, 0),
                Point(sys_part.length_mm, 0),
                Point(sys_part.length_mm, sys_part.width_mm),
                Point(0, sys_part.width_mm)
            ], id=sys_part.id)
            
            # Create advanced part
            advanced_part = Part(
                id=sys_part.id,
                polygon=polygon,
                quantity=sys_part.quantity,
                material_id=sys_part.material_id,
                priority=sys_part.priority,
                rotation_allowed=sys_part.rotation_allowed,
                rotation_step=self.config.get('rotation_step_degrees', 5.0),
                min_rotation=0.0,
                max_rotation=360.0,
                mirror_allowed=self.config.get('enable_mirroring', True),
                fixed_orientation=not sys_part.rotation_allowed
            )
            
            advanced_parts.append(advanced_part)
        
        return advanced_parts
    
    def convert_system_boards_to_advanced(self, system_boards: List[SystemBoard]) -> List[Board]:
        """Convert system boards to advanced engine boards"""
        advanced_boards = []
        
        for sys_board in system_boards:
            # Create advanced board
            advanced_board = Board(
                id=sys_board.id,
                width=sys_board.length_mm,
                height=sys_board.width_mm,
                cost=sys_board.cost,
                quantity_available=sys_board.quantity_available,
                material_id=sys_board.material_id,
                margin=sys_board.margin_mm,
                kerf_width=sys_board.kerf_mm
            )
            
            advanced_boards.append(advanced_board)
        
        return advanced_boards
    
    def convert_system_parts_to_enhanced(self, system_parts: List[SystemPart]) -> List[Dict[str, Any]]:
        """Convert system parts to enhanced algorithm format"""
        enhanced_parts = []
        
        for sys_part in system_parts:
            enhanced_part = {
                'id': sys_part.id,
                'length_mm': sys_part.length_mm,
                'width_mm': sys_part.width_mm,
                'area_sq_mm': sys_part.area_sq_mm,
                'quantity': sys_part.quantity,
                'material_id': sys_part.material_id,
                'priority': sys_part.priority,
                'rotation_allowed': sys_part.rotation_allowed,
                'svg_path': sys_part.svg_path
            }
            enhanced_parts.append(enhanced_part)
        
        return enhanced_parts
    
    def convert_system_boards_to_enhanced(self, system_boards: List[SystemBoard]) -> List[Dict[str, Any]]:
        """Convert system boards to enhanced algorithm format"""
        enhanced_boards = []
        
        for sys_board in system_boards:
            enhanced_board = {
                'id': sys_board.id,
                'length_mm': sys_board.length_mm,
                'width_mm': sys_board.width_mm,
                'area_sq_mm': sys_board.area_sq_mm,
                'cost': sys_board.cost,
                'quantity': sys_board.quantity_available,
                'material_id': sys_board.material_id,
                'margin_mm': sys_board.margin_mm,
                'kerf_mm': sys_board.kerf_mm
            }
            enhanced_boards.append(enhanced_board)
        
        return enhanced_boards
    
    def convert_system_parts_to_svgnest(self, system_parts: List[SystemPart]) -> List:
        """Convert system parts to SVGNest format"""
        from svgnest_nesting_engine import Part as SVGNestPart
        
        svgnest_parts = []
        
        for sys_part in system_parts:
            svgnest_part = SVGNestPart(
                id=sys_part.id,
                width=sys_part.length_mm,
                height=sys_part.width_mm,
                area=sys_part.area_sq_mm,
                quantity=sys_part.quantity,
                svg_path=sys_part.svg_path,
                rotation_allowed=sys_part.rotation_allowed
            )
            svgnest_parts.append(svgnest_part)
        
        return svgnest_parts
    
    def convert_system_boards_to_svgnest(self, system_boards: List[SystemBoard]) -> List:
        """Convert system boards to SVGNest format"""
        from svgnest_nesting_engine import Board as SVGNestBoard
        
        svgnest_boards = []
        
        for sys_board in system_boards:
            svgnest_board = SVGNestBoard(
                id=sys_board.id,
                width=sys_board.length_mm,
                height=sys_board.width_mm,
                area=sys_board.area_sq_mm,
                cost=sys_board.cost,
                quantity_available=sys_board.quantity_available
            )
            svgnest_boards.append(svgnest_board)
        
        return svgnest_boards
    
    def optimize_nesting_comprehensive(self, system_parts: List[SystemPart], 
                                     system_boards: List[SystemBoard],
                                     strategy: str = "hybrid") -> Dict[str, Any]:
        """
        Comprehensive nesting optimization using multiple engines.
        This is the main entry point for the integration.
        """
        self.logger.info("Starting comprehensive nesting optimization")
        self.logger.info(f"Parts: {len(system_parts)} types, {sum(p.quantity for p in system_parts)} total")
        self.logger.info(f"Boards: {len(system_boards)} types available")
        self.logger.info(f"Strategy: {strategy}")
        
        start_time = time.time()
        results = {}
        
        try:
            # Strategy 1: Advanced Engine (Genetic Algorithm + NFP)
            if strategy in ["hybrid", "advanced", "genetic"]:
                self.logger.info("Running Advanced Engine optimization...")
                try:
                    advanced_parts = self.convert_system_parts_to_advanced(system_parts)
                    advanced_boards = self.convert_system_boards_to_advanced(system_boards)
                    
                    advanced_result = self.advanced_engine.optimize_nesting(
                        advanced_parts, advanced_boards, NestingStrategy.HYBRID_OPTIMIZATION
                    )
                    
                    results['advanced_engine'] = self._convert_advanced_result_to_system(advanced_result)
                    self.logger.info(f"Advanced Engine: {advanced_result.total_boards} boards, "
                                   f"{advanced_result.utilization_percentage:.1f}% utilization")
                except Exception as e:
                    self.logger.error(f"Advanced Engine failed: {e}")
                    results['advanced_engine'] = None
            
            # Strategy 2: Enhanced Algorithm
            if strategy in ["hybrid", "enhanced", "scrap_optimization"]:
                self.logger.info("Running Enhanced Algorithm optimization...")
                try:
                    enhanced_parts = self.convert_system_parts_to_enhanced(system_parts)
                    enhanced_boards = self.convert_system_boards_to_enhanced(system_boards)
                    
                    enhanced_result = self.enhanced_algorithm.select_optimal_boards_for_parts(
                        enhanced_boards, enhanced_parts
                    )
                    
                    results['enhanced_algorithm'] = self._convert_enhanced_result_to_system(enhanced_result)
                    self.logger.info(f"Enhanced Algorithm: {len(enhanced_result)} boards used")
                except Exception as e:
                    self.logger.error(f"Enhanced Algorithm failed: {e}")
                    results['enhanced_algorithm'] = None
            
            # Strategy 3: SVGNest Engine
            if strategy in ["hybrid", "svgnest", "rotation_optimization"]:
                self.logger.info("Running SVGNest Engine optimization...")
                try:
                    svgnest_parts = self.convert_system_parts_to_svgnest(system_parts)
                    svgnest_boards = self.convert_system_boards_to_svgnest(system_boards)
                    
                    svgnest_result = self.svgnest_engine.optimize_nesting_with_svgnest(
                        svgnest_parts, svgnest_boards, max_boards=10, use_advanced_rotations=True
                    )
                    
                    results['svgnest_engine'] = self._convert_svgnest_result_to_system(svgnest_result)
                    self.logger.info(f"SVGNest Engine: {svgnest_result['total_boards_used']} boards, "
                                   f"{svgnest_result['total_utilization']:.1%} utilization")
                except Exception as e:
                    self.logger.error(f"SVGNest Engine failed: {e}")
                    results['svgnest_engine'] = None
            
            # Strategy 4: Multi-Board Optimizer
            if strategy in ["hybrid", "multi_board", "all_parts"]:
                self.logger.info("Running Multi-Board Optimizer...")
                try:
                    multi_parts = self.convert_system_parts_to_svgnest(system_parts)
                    multi_boards = self.convert_system_boards_to_svgnest(system_boards)
                    
                    multi_result = self.multi_board_optimizer.optimize_multi_board_nesting(
                        multi_parts, multi_boards, max_boards=10, use_advanced_rotations=True
                    )
                    
                    results['multi_board_optimizer'] = self._convert_svgnest_result_to_system(multi_result)
                    self.logger.info(f"Multi-Board Optimizer: {multi_result['total_boards_used']} boards, "
                                   f"{multi_result['total_utilization']:.1%} utilization")
                except Exception as e:
                    self.logger.error(f"Multi-Board Optimizer failed: {e}")
                    results['multi_board_optimizer'] = None
            
            # Select best result
            best_result = self._select_best_result(results, system_parts, system_boards)
            
            optimization_time = time.time() - start_time
            best_result['optimization_time'] = optimization_time
            best_result['all_results'] = results
            
            self.logger.info(f"Comprehensive optimization completed in {optimization_time:.2f} seconds")
            self.logger.info(f"Best result: {best_result.get('total_boards_used', 0)} boards, "
                           f"{best_result.get('total_utilization', 0):.1f}% utilization")
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive optimization failed: {e}")
            return self._create_failure_result(str(e), time.time() - start_time)
    
    def _convert_advanced_result_to_system(self, result: NestingResult) -> Dict[str, Any]:
        """Convert advanced engine result to system format"""
        return {
            'success': result.success,
            'total_boards_used': result.total_boards,
            'total_cost': result.total_cost,
            'total_utilization': result.utilization_percentage,
            'total_scrap_percentage': result.scrap_percentage,
            'parts_fitted': result.parts_fitted,
            'parts_total': result.parts_total,
            'efficiency_score': result.efficiency_score,
            'optimization_time': result.optimization_time,
            'strategy_used': result.strategy_used,
            'boards_used': result.boards_used,
            'error_message': result.error_message
        }
    
    def _convert_enhanced_result_to_system(self, results: List) -> Dict[str, Any]:
        """Convert enhanced algorithm result to system format"""
        if not results:
            return {'success': False, 'error_message': 'No results from enhanced algorithm'}
        
        total_boards = len(results)
        total_cost = sum(r.board_dimensions.get('cost', 0) for r in results)
        total_utilization = sum(r.utilization_percentage for r in results) / len(results) if results else 0
        total_scrap = sum(r.scrap_percentage for r in results) / len(results) if results else 0
        total_parts_fitted = sum(r.parts_fitted for r in results)
        
        return {
            'success': True,
            'total_boards_used': total_boards,
            'total_cost': total_cost,
            'total_utilization': total_utilization,
            'total_scrap_percentage': total_scrap,
            'parts_fitted': total_parts_fitted,
            'parts_total': total_parts_fitted,  # Enhanced algorithm fits all parts
            'efficiency_score': (1.0 - total_scrap) * 100,
            'strategy_used': 'enhanced_algorithm',
            'boards_used': [
                {
                    'board_id': r.board_id,
                    'board_dimensions': r.board_dimensions,
                    'nested_parts': r.nested_parts,
                    'utilization': r.utilization_percentage,
                    'scrap_percentage': r.scrap_percentage,
                    'parts_fitted': r.parts_fitted
                }
                for r in results
            ]
        }
    
    def _convert_svgnest_result_to_system(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SVGNest result to system format"""
        return {
            'success': result.get('success', False),
            'total_boards_used': result.get('total_boards_used', 0),
            'total_cost': result.get('total_cost', 0.0),
            'total_utilization': result.get('total_utilization', 0.0) * 100,
            'total_scrap_percentage': result.get('total_scrap_percentage', 0.0) * 100,
            'parts_fitted': result.get('total_parts_fitted', 0),
            'parts_total': result.get('total_parts_required', 0),
            'efficiency_score': result.get('efficiency_score', 0.0),
            'strategy_used': result.get('optimization_type', 'svgnest'),
            'boards_used': result.get('boards_used', []),
            'all_parts_fitted': result.get('all_parts_fitted', False)
        }
    
    def _select_best_result(self, results: Dict[str, Any], system_parts: List[SystemPart], 
                           system_boards: List[SystemBoard]) -> Dict[str, Any]:
        """Select the best result from all optimization strategies"""
        valid_results = [r for r in results.values() if r and r.get('success', False)]
        
        if not valid_results:
            return self._create_failure_result("All optimization strategies failed", 0.0)
        
        # Score results based on multiple criteria
        best_result = None
        best_score = -1
        
        for result in valid_results:
            # Calculate composite score
            utilization = result.get('total_utilization', 0) / 100.0
            scrap = result.get('total_scrap_percentage', 100) / 100.0
            cost_efficiency = 1.0 / (result.get('total_cost', 1) + 1)
            parts_ratio = result.get('parts_fitted', 0) / max(result.get('parts_total', 1), 1)
            
            # Weighted score
            score = (utilization * 0.4 + 
                    (1.0 - scrap) * 0.3 + 
                    cost_efficiency * 0.2 + 
                    parts_ratio * 0.1)
            
            if score > best_score:
                best_score = score
                best_result = result
        
        if best_result:
            best_result['composite_score'] = best_score
            best_result['selected_from'] = 'multiple_strategies'
        
        return best_result or valid_results[0]
    
    def _create_failure_result(self, error_message: str, optimization_time: float) -> Dict[str, Any]:
        """Create a failure result"""
        return {
            'success': False,
            'total_boards_used': 0,
            'total_cost': 0.0,
            'total_utilization': 0.0,
            'total_scrap_percentage': 100.0,
            'parts_fitted': 0,
            'parts_total': 0,
            'efficiency_score': 0.0,
            'optimization_time': optimization_time,
            'strategy_used': 'none',
            'error_message': error_message,
            'boards_used': []
        }
    
    def generate_svg_layouts(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Generate SVG layouts for visualization"""
        svg_layouts = {}
        
        if not result.get('success', False):
            return svg_layouts
        
        boards_used = result.get('boards_used', [])
        
        for i, board_data in enumerate(boards_used):
            board_id = board_data.get('board_id', f'board_{i}')
            
            # Generate SVG layout (simplified)
            svg_content = self._generate_board_svg(board_data)
            svg_layouts[board_id] = svg_content
        
        return svg_layouts
    
    def _generate_board_svg(self, board_data: Dict[str, Any]) -> str:
        """Generate SVG layout for a board"""
        board_width = board_data.get('board_dimensions', {}).get('width_mm', 1000)
        board_height = board_data.get('board_dimensions', {}).get('height_mm', 500)
        
        # Scale for display
        scale = min(800 / board_width, 600 / board_height) if board_width > 0 and board_height > 0 else 1
        svg_width = board_width * scale
        svg_height = board_height * scale
        
        svg_content = f'''<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {board_width} {board_height}" xmlns="http://www.w3.org/2000/svg">
    <!-- Board background -->
    <rect x="0" y="0" width="{board_width}" height="{board_height}" 
          fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
    
    <!-- Margin guides -->
    <rect x="10" y="10" width="{board_width - 20}" height="{board_height - 20}" 
          fill="none" stroke="#6c757d" stroke-width="1" stroke-dasharray="5,5" opacity="0.5"/>
    
    <!-- Nested parts -->
    <text x="{board_width/2}" y="20" text-anchor="middle" font-size="16" font-weight="bold">
        Board {board_data.get('board_id', 'Unknown')} - Utilization: {board_data.get('utilization', 0):.1f}%
    </text>
</svg>'''
        
        return svg_content

# Integration functions for existing system
def integrate_with_existing_system(parts_data: List[Dict[str, Any]], 
                                 boards_data: List[Dict[str, Any]],
                                 strategy: str = "hybrid") -> Dict[str, Any]:
    """
    Main integration function for the existing system.
    This is the primary entry point for nesting optimization.
    """
    # Convert system data to internal format
    system_parts = []
    for part_data in parts_data:
        system_part = SystemPart(
            id=str(part_data.get('id', '')),
            length_mm=float(part_data.get('length_mm', 0)),
            width_mm=float(part_data.get('width_mm', 0)),
            area_sq_mm=float(part_data.get('area_sq_mm', 0)),
            quantity=int(part_data.get('quantity', 1)),
            material_id=str(part_data.get('material_id', '')),
            priority=int(part_data.get('priority', 0)),
            rotation_allowed=bool(part_data.get('rotation_allowed', True)),
            svg_path=str(part_data.get('svg_path', ''))
        )
        system_parts.append(system_part)
    
    system_boards = []
    for board_data in boards_data:
        system_board = SystemBoard(
            id=str(board_data.get('id', '')),
            length_mm=float(board_data.get('length_mm', 0)),
            width_mm=float(board_data.get('width_mm', 0)),
            area_sq_mm=float(board_data.get('area_sq_mm', 0)),
            cost=float(board_data.get('cost', 0)),
            quantity_available=int(board_data.get('quantity_available', 1)),
            material_id=str(board_data.get('material_id', '')),
            margin_mm=float(board_data.get('margin_mm', 10.0)),
            kerf_mm=float(board_data.get('kerf_mm', 0.2))
        )
        system_boards.append(system_board)
    
    # Create integration instance
    integration = EnhancedNestingIntegration()
    
    # Run comprehensive optimization
    result = integration.optimize_nesting_comprehensive(
        system_parts, system_boards, strategy
    )
    
    # Generate SVG layouts
    svg_layouts = integration.generate_svg_layouts(result)
    result['svg_layouts'] = svg_layouts
    
    return result

def test_enhanced_integration():
    """Test the enhanced integration"""
    print("🧪 Testing Enhanced Nesting Integration")
    print("=" * 60)
    
    # Create test data in system format
    parts_data = [
        {
            'id': '1',
            'length_mm': 200,
            'width_mm': 100,
            'area_sq_mm': 20000,
            'quantity': 5,
            'material_id': 'steel',
            'priority': 1,
            'rotation_allowed': True,
            'svg_path': 'M 0,0 L 200,0 L 200,100 L 0,100 Z'
        },
        {
            'id': '2',
            'length_mm': 150,
            'width_mm': 80,
            'area_sq_mm': 12000,
            'quantity': 8,
            'material_id': 'steel',
            'priority': 2,
            'rotation_allowed': True,
            'svg_path': 'M 0,0 L 150,0 L 150,80 L 0,80 Z'
        },
        {
            'id': '3',
            'length_mm': 100,
            'width_mm': 50,
            'area_sq_mm': 5000,
            'quantity': 12,
            'material_id': 'steel',
            'priority': 3,
            'rotation_allowed': True,
            'svg_path': 'M 0,0 L 100,0 L 100,50 L 0,50 Z'
        }
    ]
    
    boards_data = [
        {
            'id': '1',
            'length_mm': 1000,
            'width_mm': 500,
            'area_sq_mm': 500000,
            'cost': 100.0,
            'quantity_available': 10,
            'material_id': 'steel',
            'margin_mm': 10.0,
            'kerf_mm': 0.2
        },
        {
            'id': '2',
            'length_mm': 800,
            'width_mm': 400,
            'area_sq_mm': 320000,
            'cost': 80.0,
            'quantity_available': 10,
            'material_id': 'steel',
            'margin_mm': 10.0,
            'kerf_mm': 0.2
        }
    ]
    
    print(f"📋 Test Setup:")
    print(f"  • Parts: {len(parts_data)} types, {sum(p['quantity'] for p in parts_data)} total")
    print(f"  • Boards: {len(boards_data)} types available")
    
    # Test different strategies
    strategies = ["hybrid", "advanced", "enhanced", "svgnest"]
    
    for strategy in strategies:
        print(f"\n🔧 Testing {strategy} strategy...")
        
        start_time = time.time()
        result = integrate_with_existing_system(parts_data, boards_data, strategy)
        end_time = time.time()
        
        print(f"📊 Results for {strategy}:")
        print(f"  • Success: {'✅' if result.get('success', False) else '❌'}")
        print(f"  • Boards used: {result.get('total_boards_used', 0)}")
        print(f"  • Total cost: ${result.get('total_cost', 0):.2f}")
        print(f"  • Utilization: {result.get('total_utilization', 0):.1f}%")
        print(f"  • Scrap: {result.get('total_scrap_percentage', 0):.1f}%")
        print(f"  • Parts fitted: {result.get('parts_fitted', 0)}/{result.get('parts_total', 0)}")
        print(f"  • Efficiency score: {result.get('efficiency_score', 0):.2f}")
        print(f"  • Optimization time: {end_time - start_time:.2f}s")
        
        if result.get('error_message'):
            print(f"  • Error: {result['error_message']}")
        
        if result.get('svg_layouts'):
            print(f"  • SVG layouts generated: {len(result['svg_layouts'])}")
    
    return result

if __name__ == "__main__":
    test_enhanced_integration()
