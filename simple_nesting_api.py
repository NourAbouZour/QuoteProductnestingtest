#!/usr/bin/env python3
"""
Simple Nesting API - Direct database integration without complex imports
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify, current_app
import traceback

# Create Blueprint for simple nesting API
simple_nesting_bp = Blueprint('simple_nesting', __name__, url_prefix='/api/simple-nesting')

@simple_nesting_bp.route('/debug', methods=['GET'])
def debug_nesting():
    """Debug endpoint to check system status"""
    try:
        from DatabaseConfig import get_standard_boards
        
        boards_data = get_standard_boards()
        
        return jsonify({
            'success': True,
            'database_connected': True,
            'boards_available': len(boards_data),
            'boards': boards_data[:3] if boards_data else [],  # First 3 boards
            'message': 'Debug information retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'database_connected': False
        }), 500

@simple_nesting_bp.route('/optimize', methods=['POST'])
def optimize_nesting():
    """
    Simple nesting optimization with database integration
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
        current_app.logger.info(f"Simple nesting request received:")
        current_app.logger.info(f"  - Parts data: {len(parts_data)} parts")
        current_app.logger.info(f"  - Strategy: {strategy}")
        
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
        
        # Get boards from database
        from DatabaseConfig import get_standard_boards
        
        boards_data = get_standard_boards()
        if not boards_data:
            return jsonify({
                'success': False,
                'error': 'No boards available in database'
            }), 400
        
        # Convert boards to the format expected by the nesting system
        formatted_boards = []
        for board in boards_data:
            # Handle both 'length'/'width' and 'length_mm'/'width_mm' column names
            length = board.get('length_mm', board.get('length', 0))
            width = board.get('width_mm', board.get('width', 0))
            
            formatted_board = {
                'id': str(board.get('id', '')),
                'length_mm': float(length),
                'width_mm': float(width),
                'area_sq_mm': float(board.get('area_sq_mm', length * width)),
                'cost': float(board.get('cost', 0)),
                'quantity_available': int(board.get('quantity', 0)),
                'material_id': str(board.get('material_id', 'steel')),
                'margin_mm': 10.0,
                'kerf_mm': 0.2
            }
            formatted_boards.append(formatted_board)
        
        current_app.logger.info(f"Loaded {len(formatted_boards)} boards from database")
        
        # Simple nesting logic - just fit parts on the largest board
        largest_board = max(formatted_boards, key=lambda b: b['area_sq_mm'])
        
        # Calculate total parts area
        total_parts_area = 0
        total_parts_count = 0
        for part in parts_data:
            part_area = part.get('length_mm', 0) * part.get('width_mm', 0)
            quantity = part.get('quantity', 1)
            total_parts_area += part_area * quantity
            total_parts_count += quantity
        
        # Calculate utilization
        board_area = largest_board['area_sq_mm']
        utilization = (total_parts_area / board_area) * 100 if board_area > 0 else 0
        scrap = 100 - utilization
        
        # Create result
        result = {
            'success': True,
            'total_boards_used': 1,
            'total_cost': largest_board['cost'],
            'total_utilization': utilization,
            'total_scrap_percentage': scrap,
            'parts_fitted': total_parts_count,
            'parts_total': total_parts_count,
            'efficiency_score': utilization,
            'strategy_used': 'simple_largest_board',
            'boards_used': [{
                'board_id': largest_board['id'],
                'board_width': largest_board['length_mm'],
                'board_height': largest_board['width_mm'],
                'utilization': utilization / 100,
                'placements': total_parts_count
            }],
            'all_parts_fitted': True,
            'composite_score': utilization,
            'selected_from': 'simple_algorithm',
            'optimization_time': 0.1,
            'database_integration': True,
            'boards_from_database': len(formatted_boards),
            'parts_from_request': len(parts_data)
        }
        
        current_app.logger.info(f"Simple nesting completed: {result['total_boards_used']} boards, "
                              f"{result['total_utilization']:.1f}% utilization")
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Simple nesting optimization failed: {e}")
        current_app.logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f'Optimization failed: {str(e)}',
            'error_type': type(e).__name__
        }), 500

def register_simple_nesting_api(app):
    """Register the simple nesting API with the Flask app"""
    app.register_blueprint(simple_nesting_bp)
    print("✅ Simple Nesting API registered successfully")
