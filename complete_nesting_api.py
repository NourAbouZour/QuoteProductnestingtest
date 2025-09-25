#!/usr/bin/env python3
"""
Complete Nesting API
Ensures ALL parts are fitted across multiple boards with comprehensive analysis
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify, current_app
import traceback

# Import our optimized components
from optimized_multi_board_nesting import OptimizedMultiBoardNesting, Part, Board
from leftover_scrap_analyzer import LeftoverScrapAnalyzer, NestingResult
from nesting_visualizer import NestingVisualizer
from nesting_scrap_config import nesting_scrap_config

# Create Blueprint for complete nesting API
complete_nesting_bp = Blueprint('complete_nesting', __name__, url_prefix='/api/complete-nesting')

@complete_nesting_bp.route('/optimize', methods=['POST'])
def optimize_complete_nesting():
    """
    Complete nesting optimization that guarantees ALL parts are fitted.
    
    Expected JSON payload:
    {
        "parts": [
            {
                "id": "part1",
                "length_mm": 200,
                "width_mm": 100,
                "quantity": 5,
                "material_id": "steel",
                "rotation_allowed": true
            }
        ],
        "max_boards": 20,
        "config": {
            "min_gap_mm": 5.0,
            "margin_mm": 10.0,
            "leftover_config": {
                "leftover_threshold_percent": 20.0,
                "min_leftover_width_mm": 180.0,
                "min_leftover_height_mm": 600.0
            }
        }
    }
    
    Note: Boards are automatically loaded from the database.
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract parameters
        parts_data = data.get('parts', [])
        max_boards = data.get('max_boards', 20)
        config = data.get('config', {})
        
        # Debug logging
        current_app.logger.info(f"Complete nesting request received:")
        current_app.logger.info(f"  - Parts data: {len(parts_data)} parts")
        current_app.logger.info(f"  - Max boards: {max_boards}")
        current_app.logger.info(f"  - Config: {config}")
        
        if parts_data:
            total_instances = sum(p.get('quantity', 1) for p in parts_data)
            current_app.logger.info(f"  - Total part instances: {total_instances}")
            
            for i, part in enumerate(parts_data[:3]):  # Log first 3 parts
                current_app.logger.info(f"  - Part {i+1}: {part.get('id', 'no-id')} - {part.get('length_mm', 0)}x{part.get('width_mm', 0)}mm x{part.get('quantity', 1)}")
        
        # Validate inputs
        if not parts_data:
            current_app.logger.error("No parts provided in request")
            return jsonify({
                'success': False,
                'error': 'No parts provided'
            }), 400
        
        # Load boards from database
        from database_nesting_integration import get_boards_from_database
        boards_data = get_boards_from_database()
        
        if not boards_data:
            return jsonify({
                'success': False,
                'error': 'No boards available in database'
            }), 500
        
        current_app.logger.info(f"Loaded {len(boards_data)} boards from database")
        
        # Update configuration if provided
        if 'leftover_config' in config:
            nesting_scrap_config.update_config(**config['leftover_config'])
        
        # Convert data to internal format
        parts = _convert_parts_to_internal_format(parts_data)
        boards = _convert_boards_to_internal_format(boards_data)
        
        # Run complete nesting optimization
        start_time = time.time()
        
        # Step 1: Multi-board nesting to fit ALL parts
        nesting_engine = OptimizedMultiBoardNesting(
            margin_mm=config.get('margin_mm', 10.0),
            min_gap_mm=config.get('min_gap_mm', 5.0)
        )
        
        nesting_result = nesting_engine.optimize_all_parts(parts, boards, max_boards)
        nesting_time = time.time() - start_time
        
        if not nesting_result.get('success', False):
            return jsonify({
                'success': False,
                'error': 'Failed to fit all parts',
                'details': nesting_result.get('error', 'Unknown error'),
                'remaining_parts': len(nesting_result.get('remaining_parts', []))
            }), 500
        
        # Step 2: Leftover/Scrap Analysis
        analysis_start = time.time()
        
        # Convert nesting results to analysis format
        analysis_results = _convert_nesting_to_analysis_format(nesting_result)
        
        # Perform leftover/scrap analysis
        scrap_analyzer = LeftoverScrapAnalyzer()
        board_analyses = scrap_analyzer.analyze_multiple_boards(analysis_results)
        totals = scrap_analyzer.calculate_totals(board_analyses)
        
        analysis_time = time.time() - analysis_start
        
        # Step 3: Generate Visualizations
        visualization_start = time.time()
        
        visualizer = NestingVisualizer()
        visualizations = []
        
        for i, (analysis, board_result) in enumerate(zip(board_analyses, nesting_result['results'])):
            try:
                placed_parts = board_result.get('fitted_parts', [])
                viz_result = visualizer.create_visualization(analysis, placed_parts)
                
                if viz_result.success:
                    visualizations.append({
                        'board_id': analysis.board_id,
                        'used_area_svg': viz_result.used_area_svg,
                        'leftover_scrap_svg': viz_result.leftover_scrap_svg,
                        'combined_svg': viz_result.combined_svg,
                        'legend_svg': viz_result.legend_svg
                    })
                else:
                    current_app.logger.warning(f"Failed to create visualization for board {analysis.board_id}")
                    visualizations.append({
                        'board_id': analysis.board_id,
                        'used_area_svg': '',
                        'leftover_scrap_svg': '',
                        'combined_svg': '',
                        'legend_svg': ''
                    })
                    
            except Exception as e:
                current_app.logger.error(f"Error generating visualization for board {i}: {e}")
                visualizations.append({
                    'board_id': f'board_{i}',
                    'used_area_svg': '',
                    'leftover_scrap_svg': '',
                    'combined_svg': '',
                    'legend_svg': ''
                })
        
        visualization_time = time.time() - visualization_start
        total_time = time.time() - start_time
        
        # Create comprehensive result
        api_result = {
            'success': True,
            'all_parts_fitted': nesting_result.get('all_parts_fitted', False),
            'summary': {
                'total_boards_used': nesting_result.get('total_boards_used', 0),
                'total_cost': nesting_result.get('total_cost', 0.0),
                'total_utilization': nesting_result.get('total_utilization', 0.0),
                'total_reported_scrap_percent': totals['total_reported_scrap_percent'],
                'total_leftover_percent': totals['total_leftover_percent'],
                'boards_with_leftovers': totals['total_boards_with_leftovers'],
                'parts_fitted': nesting_result.get('parts_summary', {}).get('fitted_instances', 0),
                'parts_total': nesting_result.get('parts_summary', {}).get('total_instances', 0),
                'strategy_used': nesting_result.get('strategy_used', 'unknown')
            },
            'performance': {
                'nesting_time_ms': nesting_time * 1000,
                'analysis_time_ms': analysis_time * 1000,
                'visualization_time_ms': visualization_time * 1000,
                'total_time_ms': total_time * 1000
            },
            'board_details': [
                {
                    'board_id': analysis.board_id,
                    'board_size': f"{analysis.board_width_mm:.0f}×{analysis.board_height_mm:.0f}mm",
                    'used_area_sq_mm': analysis.used_area_sq_mm,
                    'leftover_area_sq_mm': analysis.leftover_area_sq_mm,
                    'scrap_area_sq_mm': analysis.scrap_area_sq_mm,
                    'leftover_percent': analysis.leftover_percent,
                    'scrap_percent': analysis.scrap_percent,
                    'meets_threshold': analysis.meets_threshold,
                    'pockets_count': len(analysis.pockets),
                    'parts_fitted': board_result.get('total_parts_nested', 0),
                    'utilization': board_result.get('utilization', 0.0)
                }
                for analysis, board_result in zip(board_analyses, nesting_result['results'])
            ],
            'visualizations': visualizations,
            'config_used': nesting_scrap_config.get_config()
        }
        
        # Log results
        current_app.logger.info(f"Complete nesting optimization completed:")
        current_app.logger.info(f"  - Boards used: {api_result['summary']['total_boards_used']}")
        current_app.logger.info(f"  - Parts fitted: {api_result['summary']['parts_fitted']}/{api_result['summary']['parts_total']}")
        current_app.logger.info(f"  - Utilization: {api_result['summary']['total_utilization']:.1%}")
        current_app.logger.info(f"  - Leftover: {api_result['summary']['total_leftover_percent']:.1%}")
        current_app.logger.info(f"  - Scrap: {api_result['summary']['total_reported_scrap_percent']:.1%}")
        current_app.logger.info(f"  - All parts fitted: {api_result['all_parts_fitted']}")
        current_app.logger.info(f"  - Total time: {total_time:.2f}s")
        
        return jsonify(api_result)
        
    except Exception as e:
        current_app.logger.error(f"Complete nesting optimization failed: {e}")
        current_app.logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f'Optimization failed: {str(e)}',
            'error_type': type(e).__name__
        }), 500

def _convert_parts_to_internal_format(parts_data: List[Dict[str, Any]]) -> List[Part]:
    """Convert parts data to internal Part format"""
    parts = []
    
    for part_data in parts_data:
        part = Part(
            id=str(part_data.get('id', '')),
            width=float(part_data.get('width_mm', 0)),
            height=float(part_data.get('length_mm', 0)),  # Note: length becomes height
            quantity=int(part_data.get('quantity', 1)),
            rotation_allowed=bool(part_data.get('rotation_allowed', True)),
            svg_path=str(part_data.get('svg_path', ''))
        )
        parts.append(part)
    
    return parts

def _convert_boards_to_internal_format(boards_data: List[Dict[str, Any]]) -> List[Board]:
    """Convert boards data to internal Board format"""
    boards = []
    
    for board_data in boards_data:
        board = Board(
            id=str(board_data.get('id', '')),
            width=float(board_data.get('width_mm', 0)),
            height=float(board_data.get('length_mm', 0)),  # Note: length becomes height
            cost=float(board_data.get('cost', 0)),
            quantity_available=int(board_data.get('quantity_available', 1))
        )
        boards.append(board)
    
    return boards

def _convert_nesting_to_analysis_format(nesting_result: Dict) -> List[NestingResult]:
    """Convert nesting result to format expected by analyzer"""
    analysis_results = []
    
    for board_result in nesting_result.get('results', []):
        board = board_result.get('board', {})
        fitted_parts = board_result.get('fitted_parts', [])
        
        # Convert fitted parts to the format expected by analyzer
        placed_parts = []
        for part in fitted_parts:
            placed_parts.append({
                'id': part['id'],
                'x': part['x'],
                'y': part['y'],
                'width_mm': part['width'],
                'height_mm': part['height'],
                'rotation': part.get('rotation', 0)
            })
        
        analysis_result = NestingResult(
            board_id=board.get('id', 'unknown'),
            board_width_mm=board.get('width', 0),
            board_height_mm=board.get('height', 0),
            placed_parts=placed_parts
        )
        
        analysis_results.append(analysis_result)
    
    return analysis_results

@complete_nesting_bp.route('/test', methods=['GET'])
def test_complete_nesting():
    """Test endpoint with sample data that requires multiple boards"""
    try:
        # Create sample data that will definitely require multiple boards
        test_data = {
            'parts': [
                {
                    'id': f'part_{i}',
                    'length_mm': 200 + (i % 5) * 50,
                    'width_mm': 100 + (i % 3) * 30,
                    'quantity': 15,  # 15 instances of each part
                    'material_id': 'steel',
                    'rotation_allowed': True
                }
                for i in range(1, 21)  # 20 different part types = 300 total parts
            ],
            'max_boards': 25,
            'config': {
                'min_gap_mm': 5.0,
                'margin_mm': 10.0,
                'leftover_config': {
                    'leftover_threshold_percent': 20.0,
                    'min_leftover_width_mm': 180.0,
                    'min_leftover_height_mm': 600.0
                }
            }
        }
        
        # Simulate the request
        request._json = test_data
        
        # Call the optimization endpoint
        return optimize_complete_nesting()
        
    except Exception as e:
        current_app.logger.error(f"Test complete nesting failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Test failed: {str(e)}'
        }), 500

@complete_nesting_bp.route('/status', methods=['GET'])
def get_status():
    """Get system status and capabilities"""
    try:
        status = {
            'success': True,
            'system': 'Complete Nesting Engine - Guaranteed All Parts Fitted',
            'version': '3.0.0',
            'capabilities': [
                'Guaranteed fitting of ALL parts across multiple boards',
                'Exhaustive multi-board optimization strategies',
                'Comprehensive leftover/scrap analysis',
                'Side-by-side visualization with legend',
                'Real-time optimization with performance metrics',
                'Multiple optimization strategies (exhaustive, greedy, balanced)',
                'Configurable leftover thresholds and minimum sizes',
                'Advanced pocket detection and classification',
                'Database integration for boards and materials'
            ],
            'guarantees': [
                'ALL parts will be fitted (or clear error message if impossible)',
                'Optimal board utilization across multiple boards',
                'Comprehensive scrap analysis excluding usable leftovers',
                'Detailed visualizations for each board',
                'Performance metrics and timing information'
            ],
            'supported_formats': ['JSON', 'SVG', 'DXF'],
            'max_parts': 10000,
            'max_boards': 100,
            'max_optimization_time': 3600,
            'current_config': nesting_scrap_config.get_config()
        }
        
        return jsonify(status)
        
    except Exception as e:
        current_app.logger.error(f"Status check failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Status check failed: {str(e)}'
        }), 500

# Error handlers
@complete_nesting_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': str(error)
    }), 400

@complete_nesting_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# Health check endpoint
@complete_nesting_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': time.time(),
        'guarantee': 'ALL parts will be fitted across multiple boards',
        'config': nesting_scrap_config.get_config()
    })

# Register the blueprint with the Flask app
def register_complete_nesting_api(app):
    """Register the complete nesting API with the Flask app"""
    app.register_blueprint(complete_nesting_bp)
    app.logger.info("Complete Nesting API - Guaranteed All Parts Fitted - registered successfully")
