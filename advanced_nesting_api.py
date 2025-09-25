#!/usr/bin/env python3
"""
Advanced Nesting API
Provides REST API endpoints for the advanced nesting functionality.
Integrates with the existing Flask application.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify, current_app
import traceback

# Import our advanced nesting modules
from enhanced_nesting_integration import (
    EnhancedNestingIntegration, integrate_with_existing_system,
    SystemPart, SystemBoard
)
from advanced_nesting_engine import (
    AdvancedNestingEngine, NestingStrategy, Part, Board, Polygon, Point
)

# Create Blueprint for advanced nesting API
advanced_nesting_bp = Blueprint('advanced_nesting', __name__, url_prefix='/api/advanced-nesting')

@advanced_nesting_bp.route('/optimize', methods=['POST'])
def optimize_nesting():
    """
    Main endpoint for advanced nesting optimization with database integration.
    
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
            "rotation_step_degrees": 5.0,
            "enable_advanced_rotations": true,
            "enable_mirroring": true
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
        current_app.logger.info(f"Advanced nesting request received:")
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
        
        # Log request
        current_app.logger.info(f"Advanced nesting optimization request: "
                               f"{len(parts_data)} parts, strategy: {strategy}")
        
        # Use database integration for optimization
        from database_nesting_integration import run_database_nesting_optimization
        
        # Run optimization with database integration
        start_time = time.time()
        result = run_database_nesting_optimization(data, strategy)
        optimization_time = time.time() - start_time
        
        # Add timing information
        result['api_processing_time'] = optimization_time
        
        # Log results
        current_app.logger.info(f"Database nesting optimization completed: {result.get('total_boards_used', 0)} boards, "
                              f"{result.get('total_utilization', 0):.1f}% utilization, "
                              f"{optimization_time:.2f}s")
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Advanced nesting optimization failed: {e}")
        current_app.logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f'Optimization failed: {str(e)}',
            'error_type': type(e).__name__
        }), 500

@advanced_nesting_bp.route('/strategies', methods=['GET'])
def get_available_strategies():
    """Get list of available optimization strategies"""
    strategies = [
        {
            'id': 'hybrid',
            'name': 'Hybrid Optimization',
            'description': 'Uses multiple algorithms and selects the best result',
            'features': ['Genetic Algorithm', 'No-Fit Polygon', 'Bottom-Left Fill', 'Simulated Annealing']
        },
        {
            'id': 'advanced',
            'name': 'Advanced Engine',
            'description': 'State-of-the-art genetic algorithm with full rotation support',
            'features': ['360-degree rotation', 'Mirroring', 'Genetic optimization', 'NFP algorithms']
        },
        {
            'id': 'enhanced',
            'name': 'Enhanced Algorithm',
            'description': 'Optimized for material usage and scrap reduction',
            'features': ['Scrap optimization', 'Board prioritization', 'Part sorting']
        },
        {
            'id': 'svgnest',
            'name': 'SVGNest Engine',
            'description': 'Based on SVGNest library with advanced rotation support',
            'features': ['SVG support', 'Advanced rotations', 'Multi-board optimization']
        },
        {
            'id': 'multi_board',
            'name': 'Multi-Board Optimizer',
            'description': 'Ensures all parts are fitted using multiple boards',
            'features': ['Guaranteed fitting', 'Multi-board strategy', 'Cost optimization']
        }
    ]
    
    return jsonify({
        'success': True,
        'strategies': strategies
    })

@advanced_nesting_bp.route('/config', methods=['GET'])
def get_default_config():
    """Get default configuration options"""
    config = {
        'min_gap_mm': {
            'default': 5.0,
            'min': 0.0,
            'max': 50.0,
            'description': 'Minimum gap between parts in millimeters'
        },
        'margin_mm': {
            'default': 10.0,
            'min': 0.0,
            'max': 100.0,
            'description': 'Margin from board edges in millimeters'
        },
        'rotation_step_degrees': {
            'default': 5.0,
            'min': 1.0,
            'max': 90.0,
            'description': 'Step size for rotation angles in degrees'
        },
        'enable_advanced_rotations': {
            'default': True,
            'type': 'boolean',
            'description': 'Enable 360-degree rotation optimization'
        },
        'enable_mirroring': {
            'default': True,
            'type': 'boolean',
            'description': 'Enable part mirroring for better fitting'
        },
        'genetic_population_size': {
            'default': 50,
            'min': 10,
            'max': 200,
            'description': 'Population size for genetic algorithm'
        },
        'genetic_generations': {
            'default': 100,
            'min': 10,
            'max': 1000,
            'description': 'Number of generations for genetic algorithm'
        },
        'max_optimization_time': {
            'default': 300,
            'min': 10,
            'max': 3600,
            'description': 'Maximum optimization time in seconds'
        }
    }
    
    return jsonify({
        'success': True,
        'config': config
    })

@advanced_nesting_bp.route('/validate', methods=['POST'])
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
        boards_data = data.get('boards', [])
        
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
        
        # Validate boards
        if not boards_data:
            validation_result['errors'].append('No boards provided')
        else:
            total_board_area = 0
            for i, board in enumerate(boards_data):
                if not board.get('id'):
                    validation_result['errors'].append(f'Board {i} missing ID')
                
                length = board.get('length_mm', 0)
                width = board.get('width_mm', 0)
                quantity = board.get('quantity_available', 1)
                
                if length <= 0 or width <= 0:
                    validation_result['errors'].append(f'Board {board.get("id", i)} has invalid dimensions')
                
                area = length * width
                total_board_area += area * quantity
            
            validation_result['statistics']['total_boards'] = len(boards_data)
            validation_result['statistics']['total_board_area'] = total_board_area
        
        # Check if parts can fit
        if validation_result['statistics'].get('total_parts_area', 0) > validation_result['statistics'].get('total_board_area', 0):
            validation_result['warnings'].append('Total parts area exceeds total board area')
        
        validation_result['success'] = len(validation_result['errors']) == 0
        
        return jsonify(validation_result)
        
    except Exception as e:
        current_app.logger.error(f"Input validation failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Validation failed: {str(e)}'
        }), 500

@advanced_nesting_bp.route('/test', methods=['GET'])
def test_optimization():
    """Test endpoint with sample data using database integration"""
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
            ]
        }
        
        # Use database integration for testing
        from database_nesting_integration import run_database_nesting_optimization
        
        # Run optimization with database integration
        start_time = time.time()
        result = run_database_nesting_optimization(test_data, 'hybrid')
        optimization_time = time.time() - start_time
        
        result['api_processing_time'] = optimization_time
        result['test_data'] = test_data
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Test optimization failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Test failed: {str(e)}'
        }), 500

@advanced_nesting_bp.route('/test-database', methods=['GET'])
def test_database_integration():
    """Test database integration specifically"""
    try:
        from database_nesting_integration import test_database_nesting
        
        success = test_database_nesting()
        
        return jsonify({
            'success': success,
            'message': 'Database integration test completed',
            'database_connected': success
        })
        
    except Exception as e:
        current_app.logger.error(f"Database integration test failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Database integration test failed: {str(e)}'
        }), 500

@advanced_nesting_bp.route('/status', methods=['GET'])
def get_status():
    """Get system status and capabilities"""
    try:
        status = {
            'success': True,
            'system': 'Advanced Nesting Engine',
            'version': '1.0.0',
            'capabilities': [
                '360-degree rotation optimization',
                'Genetic algorithm optimization',
                'No-Fit Polygon algorithms',
                'Multi-board optimization',
                'SVG layout generation',
                'Real-time optimization',
                'Multiple optimization strategies'
            ],
            'supported_formats': ['JSON', 'SVG', 'DXF'],
            'max_parts': 1000,
            'max_boards': 100,
            'max_optimization_time': 3600
        }
        
        return jsonify(status)
        
    except Exception as e:
        current_app.logger.error(f"Status check failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Status check failed: {str(e)}'
        }), 500

# Error handlers
@advanced_nesting_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': str(error)
    }), 400

@advanced_nesting_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# Health check endpoint
@advanced_nesting_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': time.time()
    })

# Register the blueprint with the Flask app
def register_advanced_nesting_api(app):
    """Register the advanced nesting API with the Flask app"""
    app.register_blueprint(advanced_nesting_bp)
    app.logger.info("Advanced Nesting API registered successfully")
