#!/usr/bin/env python3
"""
Enhanced Nesting API with Leftover/Scrap Analysis
Provides REST API endpoints for the enhanced nesting functionality with comprehensive scrap analysis.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify, current_app
import traceback

# Import our enhanced nesting components
from enhanced_nesting_with_scrap import EnhancedNestingWithScrap, optimize_with_scrap_analysis
from nesting_scrap_config import nesting_scrap_config, get_leftover_config, get_visualization_config

# Create Blueprint for enhanced nesting API
enhanced_nesting_bp = Blueprint('enhanced_nesting', __name__, url_prefix='/api/enhanced-nesting')

@enhanced_nesting_bp.route('/optimize', methods=['POST'])
def optimize_nesting_with_scrap():
    """
    Main endpoint for enhanced nesting optimization with leftover/scrap analysis.
    
    Expected JSON payload:
    {
        "parts": [
            {
                "id": "part1",
                "length_mm": 200,
                "width_mm": 100,
                "quantity": 5,
                "material_id": "steel",
                "priority": 1,
                "rotation_allowed": true
            }
        ],
        "strategy": "hybrid",
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
        strategy = data.get('strategy', 'hybrid')
        config = data.get('config', {})
        
        # Debug logging
        current_app.logger.info(f"Enhanced nesting request received:")
        current_app.logger.info(f"  - Parts data: {len(parts_data)} parts")
        current_app.logger.info(f"  - Strategy: {strategy}")
        current_app.logger.info(f"  - Config: {config}")
        
        if parts_data:
            for i, part in enumerate(parts_data[:3]):  # Log first 3 parts
                current_app.logger.info(f"  - Part {i+1}: {part.get('id', 'no-id')} - {part.get('length_mm', 0)}x{part.get('width_mm', 0)}mm x{part.get('quantity', 1)}")
        
        # Validate inputs
        if not parts_data:
            current_app.logger.error("No parts provided in request")
            return jsonify({
                'success': False,
                'error': 'No parts provided',
                'debug_info': {
                    'received_data_keys': list(data.keys()),
                    'parts_data_type': type(parts_data).__name__,
                    'parts_data_length': len(parts_data) if parts_data else 0
                }
            }), 400
        
        # Update configuration if provided
        if 'leftover_config' in config:
            nesting_scrap_config.update_config(**config['leftover_config'])
        
        # Run enhanced optimization
        start_time = time.time()
        result = optimize_with_scrap_analysis(parts_data, strategy=strategy, config=config)
        optimization_time = time.time() - start_time
        
        # Convert result to API format
        api_result = _convert_result_to_api_format(result)
        api_result['api_processing_time'] = optimization_time
        
        # Log results
        current_app.logger.info(f"Enhanced nesting optimization completed: {result.total_boards_used} boards, "
                              f"{result.total_leftover_percent:.1f}% leftovers, "
                              f"{result.total_reported_scrap_percent:.1f}% scrap, "
                              f"{optimization_time:.2f}s")
        
        return jsonify(api_result)
        
    except Exception as e:
        current_app.logger.error(f"Enhanced nesting optimization failed: {e}")
        current_app.logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f'Optimization failed: {str(e)}',
            'error_type': type(e).__name__
        }), 500

@enhanced_nesting_bp.route('/config', methods=['GET'])
def get_config():
    """Get current configuration for leftover/scrap analysis"""
    try:
        config = nesting_scrap_config.get_config()
        
        return jsonify({
            'success': True,
            'config': config,
            'descriptions': {
                'leftover_threshold_percent': 'Only show leftovers if board scrap > this percentage',
                'min_leftover_width_mm': 'Minimum width for a leftover pocket',
                'min_leftover_height_mm': 'Minimum height for a leftover pocket',
                'visualization': 'Visualization settings for colors and styles',
                'analysis': 'Analysis algorithm settings',
                'performance': 'Performance and timeout settings'
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting config: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get config: {str(e)}'
        }), 500

@enhanced_nesting_bp.route('/config', methods=['POST'])
def update_config():
    """Update configuration for leftover/scrap analysis"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Update configuration
        nesting_scrap_config.update_config(**data)
        
        # Validate configuration
        errors = nesting_scrap_config.validate_config()
        if errors:
            return jsonify({
                'success': False,
                'error': 'Configuration validation failed',
                'validation_errors': errors
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully',
            'config': nesting_scrap_config.get_config()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error updating config: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to update config: {str(e)}'
        }), 500

@enhanced_nesting_bp.route('/strategies', methods=['GET'])
def get_available_strategies():
    """Get list of available optimization strategies"""
    strategies = [
        {
            'id': 'hybrid',
            'name': 'Hybrid Optimization',
            'description': 'Uses multiple algorithms and selects the best result with scrap analysis',
            'features': ['Genetic Algorithm', 'No-Fit Polygon', 'Bottom-Left Fill', 'Simulated Annealing', 'Leftover/Scrap Analysis']
        },
        {
            'id': 'advanced',
            'name': 'Advanced Engine',
            'description': 'State-of-the-art genetic algorithm with full rotation support and scrap analysis',
            'features': ['360-degree rotation', 'Mirroring', 'Genetic optimization', 'NFP algorithms', 'Advanced scrap analysis']
        },
        {
            'id': 'enhanced',
            'name': 'Enhanced Algorithm',
            'description': 'Optimized for material usage and comprehensive scrap reduction',
            'features': ['Scrap optimization', 'Board prioritization', 'Part sorting', 'Leftover detection']
        },
        {
            'id': 'svgnest',
            'name': 'SVGNest Engine',
            'description': 'Based on SVGNest library with advanced rotation support and scrap analysis',
            'features': ['SVG support', 'Advanced rotations', 'Multi-board optimization', 'Visual scrap analysis']
        },
        {
            'id': 'multi_board',
            'name': 'Multi-Board Optimizer',
            'description': 'Ensures all parts are fitted using multiple boards with scrap analysis',
            'features': ['Guaranteed fitting', 'Multi-board strategy', 'Cost optimization', 'Comprehensive scrap analysis']
        }
    ]
    
    return jsonify({
        'success': True,
        'strategies': strategies
    })

@enhanced_nesting_bp.route('/test', methods=['GET'])
def test_optimization():
    """Test endpoint with sample data"""
    try:
        # Create sample data
        test_data = {
            'parts': [
                {
                    'id': '1',
                    'length_mm': 200,
                    'width_mm': 100,
                    'quantity': 5,
                    'material_id': 'steel',
                    'priority': 1,
                    'rotation_allowed': True
                },
                {
                    'id': '2',
                    'length_mm': 150,
                    'width_mm': 80,
                    'quantity': 8,
                    'material_id': 'steel',
                    'priority': 2,
                    'rotation_allowed': True
                },
                {
                    'id': '3',
                    'length_mm': 100,
                    'width_mm': 50,
                    'quantity': 12,
                    'material_id': 'steel',
                    'priority': 3,
                    'rotation_allowed': True
                }
            ],
            'strategy': 'hybrid',
            'config': {
                'leftover_config': {
                    'leftover_threshold_percent': 20.0,
                    'min_leftover_width_mm': 180.0,
                    'min_leftover_height_mm': 600.0
                }
            }
        }
        
        # Run optimization
        start_time = time.time()
        result = optimize_with_scrap_analysis(test_data['parts'], strategy='hybrid', config=test_data['config'])
        optimization_time = time.time() - start_time
        
        # Convert result to API format
        api_result = _convert_result_to_api_format(result)
        api_result['api_processing_time'] = optimization_time
        api_result['test_data'] = test_data
        
        return jsonify(api_result)
        
    except Exception as e:
        current_app.logger.error(f"Test optimization failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Test failed: {str(e)}'
        }), 500

@enhanced_nesting_bp.route('/validate', methods=['POST'])
def validate_inputs():
    """Validate input data before optimization"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        parts_data = data.get('parts', [])
        config = data.get('config', {})
        
        validation_result = {
            'success': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Validate parts
        if not parts_data:
            validation_result['errors'].append('No parts provided')
        else:
            total_parts_area = 0
            for i, part in enumerate(parts_data):
                if not part.get('id'):
                    validation_result['errors'].append(f'Part {i} missing ID')
                
                length = part.get('length_mm', 0)
                width = part.get('width_mm', 0)
                quantity = part.get('quantity', 1)
                
                if length <= 0 or width <= 0:
                    validation_result['errors'].append(f'Part {part.get("id", i)} has invalid dimensions')
                
                area = length * width
                total_parts_area += area * quantity
            
            validation_result['statistics']['total_parts'] = len(parts_data)
            validation_result['statistics']['total_instances'] = sum(p.get('quantity', 1) for p in parts_data)
            validation_result['statistics']['total_parts_area'] = total_parts_area
        
        # Validate leftover configuration
        leftover_config = config.get('leftover_config', {})
        threshold = leftover_config.get('leftover_threshold_percent', 20.0)
        min_width = leftover_config.get('min_leftover_width_mm', 180.0)
        min_height = leftover_config.get('min_leftover_height_mm', 600.0)
        
        if not (0 <= threshold <= 100):
            validation_result['errors'].append('leftover_threshold_percent must be between 0 and 100')
        
        if min_width <= 0 or min_height <= 0:
            validation_result['errors'].append('min_leftover_width_mm and min_leftover_height_mm must be positive')
        
        validation_result['success'] = len(validation_result['errors']) == 0
        
        return jsonify(validation_result)
        
    except Exception as e:
        current_app.logger.error(f"Input validation failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Validation failed: {str(e)}'
        }), 500

@enhanced_nesting_bp.route('/status', methods=['GET'])
def get_status():
    """Get system status and capabilities"""
    try:
        status = {
            'success': True,
            'system': 'Enhanced Nesting Engine with Scrap Analysis',
            'version': '2.0.0',
            'capabilities': [
                '360-degree rotation optimization',
                'Genetic algorithm optimization',
                'No-Fit Polygon algorithms',
                'Multi-board optimization',
                'Comprehensive leftover/scrap analysis',
                'Side-by-side visualization',
                'Real-time optimization',
                'Multiple optimization strategies',
                'Configurable leftover thresholds',
                'Pocket detection and classification'
            ],
            'supported_formats': ['JSON', 'SVG', 'DXF'],
            'max_parts': 1000,
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

def _convert_result_to_api_format(result) -> Dict[str, Any]:
    """Convert EnhancedNestingResult to API response format"""
    if not result.success:
        return {
            'success': False,
            'error': 'Nesting optimization failed',
            'config_used': result.config_used
        }
    
    return {
        'success': True,
        'summary': {
            'total_boards_used': result.total_boards_used,
            'total_cost': result.total_cost,
            'total_utilization': result.total_utilization,
            'total_reported_scrap_percent': result.total_reported_scrap_percent,
            'total_leftover_percent': result.total_leftover_percent,
            'boards_with_leftovers': result.boards_with_leftovers,
            'parts_fitted': result.parts_fitted,
            'parts_total': result.parts_total,
            'efficiency_score': result.efficiency_score
        },
        'performance': {
            'nesting_time_ms': result.nesting_time_ms,
            'analysis_time_ms': result.analysis_time_ms,
            'visualization_time_ms': result.visualization_time_ms,
            'total_time_ms': result.total_time_ms
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
                'analysis_time_ms': analysis.analysis_time_ms
            }
            for analysis in result.board_analyses
        ],
        'visualizations': result.board_visualizations,
        'config_used': result.config_used
    }

# Error handlers
@enhanced_nesting_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': str(error)
    }), 400

@enhanced_nesting_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# Health check endpoint
@enhanced_nesting_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': time.time(),
        'config': nesting_scrap_config.get_config()
    })

# Register the blueprint with the Flask app
def register_enhanced_nesting_api(app):
    """Register the enhanced nesting API with the Flask app"""
    app.register_blueprint(enhanced_nesting_bp)
    app.logger.info("Enhanced Nesting API with Scrap Analysis registered successfully")
