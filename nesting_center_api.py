#!/usr/bin/env python3
"""
Nesting Center API Integration
Provides REST API endpoints for the Nesting Center cloud nesting service.
Uses the professional nesting algorithms from nestingcenter.com
"""

import asyncio
import json
import logging
import time
import sys
import os
import ssl
import socket
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify, current_app
import traceback

# Add the .cursor/nesting folder to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.cursor'))

# Import the Nesting Center modules
try:
    from nesting.Nesting import Nesting
    from nesting.NestingConverters import NestingConverters
    from nesting.SvgCreator import SvgCreator
    import aiohttp
    NESTING_CENTER_AVAILABLE = True
except ImportError as e:
    NESTING_CENTER_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Create Blueprint for Nesting Center API
nesting_center_bp = Blueprint('nesting_center', __name__, url_prefix='/api/nesting-center')

# Configure logging
logger = logging.getLogger(__name__)

# #region agent log
def _debug_log(location, message, data, hypothesis_id):
    try:
        log_path = r"c:\Users\User\Desktop\QuoteProduct-132d0ac7cb48648e235b5458dc590025a7b7d3c0\.cursor\debug.log"
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            f.flush()
    except Exception as e:
        print(f"[DEBUG LOG ERROR] {e}")
# #endregion


def convert_parts_to_nesting_format(parts: List[Dict], gap: float = 5.0) -> List[Dict]:
    """
    Convert application parts format to Nesting Center format.
    
    Args:
        parts: List of parts with length_mm, width_mm, quantity
        gap: Gap between parts in mm
        
    Returns:
        List of parts in Nesting Center format
    """
    nesting_parts = []
    total_quantity = 0
    for part in parts:
        # Handle rectangular parts
        length = float(part.get('length_mm', 0))
        width = float(part.get('width_mm', 0))
        quantity = int(part.get('quantity', 1))
        total_quantity += quantity
        
        if length > 0 and width > 0:
            nesting_part = {
                "RectangularShape": {
                    "Length": length,
                    "Width": width
                },
                "Quantity": quantity
            }
            
            # Add part ID as label if available
            if 'id' in part:
                nesting_part["Label"] = str(part['id'])
            
            nesting_parts.append(nesting_part)
            logger.info(f"Converted part: {length}x{width}mm, Quantity: {quantity}")
    
    logger.info(f"Total parts converted: {len(nesting_parts)}, Total quantity: {total_quantity}")
    return nesting_parts


def convert_boards_to_nesting_format(boards: List[Dict]) -> List[Dict]:
    """
    Convert application boards format to Nesting Center raw plates format.
    
    Args:
        boards: List of boards with width_mm, height_mm, quantity
        
    Returns:
        List of raw plates in Nesting Center format
    """
    raw_plates = []
    for board in boards:
        length = float(board.get('width_mm', board.get('length_mm', 0)))
        width = float(board.get('height_mm', board.get('width_mm', 0)))
        quantity = int(board.get('quantity', board.get('available_quantity', 1)))
        
        if length > 0 and width > 0:
            raw_plate = {
                "RectangularShape": {
                    "Length": length,
                    "Width": width
                },
                "Quantity": quantity
            }
            
            # Add board ID as label if available
            if 'id' in board:
                raw_plate["Label"] = str(board['id'])
            
            raw_plates.append(raw_plate)
    
    return raw_plates


def build_nesting_request(
    parts: List[Dict],
    raw_plates: List[Dict],
    settings: Optional[Dict] = None,
    stop_conditions: Optional[Dict] = None
) -> Dict:
    """
    Build a complete nesting request for the Nesting Center API.
    
    Args:
        parts: List of parts in Nesting Center format
        raw_plates: List of raw plates in Nesting Center format
        settings: Nesting settings (distances, rotation, etc.)
        stop_conditions: When to stop the nesting computation
        
    Returns:
        Complete nesting request dictionary
    """
    # Default settings
    default_settings = {
        "DistancePartPart": 5.0,        # Gap between parts
        "DistancePartRawPlate": 5.0,    # Margin from plate edges
        "RotationControl": "Fixed90"     # Allow 0°, 90°, 180°, 270° rotations
    }
    
    if settings:
        default_settings.update(settings)
    
    # Default stop conditions
    default_stop = {
        "AllPartsNested": True,  # Stop when all parts are nested
        "Timeout": 60            # Maximum 60 seconds
    }
    
    if stop_conditions:
        default_stop.update(stop_conditions)
    
    return {
        "Context": {
            "Settings": default_settings,
            "Problem": {
                "Parts": parts,
                "RawPlates": raw_plates
            }
        },
        "StopJson": default_stop
    }


async def run_nesting_computation(
    nesting_data: Dict,
    poll_interval: float = 2.0,
    max_wait: float = 120.0
) -> Dict:
    """
    Run a nesting computation on the Nesting Center cloud service.
    
    Args:
        nesting_data: The nesting request data
        poll_interval: How often to check status (seconds)
        max_wait: Maximum time to wait for results (seconds)
        
    Returns:
        Nesting result dictionary
    """
    # #region agent log
    hostname = "api-nesting.nestingcenter.com"
    _debug_log("nesting_center_api.py:179", "run_nesting_computation entry", {"hostname": hostname}, "C")
    # #endregion
    
    # #region agent log
    # Test general internet connectivity with a well-known DNS server
    try:
        _debug_log("nesting_center_api.py:183", "testing general DNS connectivity", {"test_host": "8.8.8.8"}, "C")
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        _debug_log("nesting_center_api.py:185", "general network connectivity OK", {}, "C")
    except Exception as net_ex:
        _debug_log("nesting_center_api.py:187", "general network connectivity failed", {"error": str(net_ex), "error_type": type(net_ex).__name__}, "C")
    # #endregion
    
    # #region agent log
    try:
        _debug_log("nesting_center_api.py:191", "testing DNS resolution before SSL setup", {"hostname": hostname}, "C")
        # Try IPv4 first
        try:
            addr_info_v4 = socket.getaddrinfo(hostname, 443, socket.AF_INET, socket.SOCK_STREAM)
            _debug_log("nesting_center_api.py:194", "DNS resolution IPv4 successful", {"addr_count": len(addr_info_v4) if addr_info_v4 else 0, "first_addr": str(addr_info_v4[0]) if addr_info_v4 else None}, "C")
        except Exception as v4_ex:
            _debug_log("nesting_center_api.py:196", "DNS resolution IPv4 failed", {"error": str(v4_ex), "error_type": type(v4_ex).__name__}, "C")
        # Try IPv6
        try:
            addr_info_v6 = socket.getaddrinfo(hostname, 443, socket.AF_INET6, socket.SOCK_STREAM)
            _debug_log("nesting_center_api.py:199", "DNS resolution IPv6 successful", {"addr_count": len(addr_info_v6) if addr_info_v6 else 0}, "C")
        except Exception as v6_ex:
            _debug_log("nesting_center_api.py:201", "DNS resolution IPv6 failed", {"error": str(v6_ex), "error_type": type(v6_ex).__name__}, "C")
        # Try UNSPEC (original)
        addr_info = socket.getaddrinfo(hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
        _debug_log("nesting_center_api.py:204", "DNS resolution UNSPEC successful", {"addr_count": len(addr_info) if addr_info else 0, "first_addr": str(addr_info[0]) if addr_info else None}, "C")
    except Exception as dns_ex:
        _debug_log("nesting_center_api.py:206", "DNS resolution failed before connection", {"hostname": hostname, "error": str(dns_ex), "error_type": type(dns_ex).__name__}, "C")
    # #endregion
    
    # Create SSL context that doesn't verify certificates (for testing)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # #region agent log
    _debug_log("nesting_center_api.py:192", "SSL context created", {"check_hostname": ssl_context.check_hostname, "verify_mode": ssl_context.verify_mode}, "D")
    # #endregion
    
    # Configure custom DNS resolver to fix Windows DNS resolution issues
    # Use AsyncResolver with Google DNS servers for reliable DNS resolution
    resolver = None
    try:
        from aiohttp.resolver import AsyncResolver
        resolver = AsyncResolver(nameservers=["8.8.8.8", "8.8.4.4"])
        # #region agent log
        _debug_log("nesting_center_api.py:250", "AsyncResolver created with custom nameservers", {"nameservers": ["8.8.8.8", "8.8.4.4"]}, "D")
        # #endregion
    except ImportError:
        # aiodns not available, fall back to default resolver
        # #region agent log
        _debug_log("nesting_center_api.py:254", "AsyncResolver not available (aiodns missing), using default resolver", {}, "D")
        # #endregion
        pass
    except Exception as resolver_ex:
        # #region agent log
        _debug_log("nesting_center_api.py:258", "Failed to create AsyncResolver, using default", {"error": str(resolver_ex)}, "D")
        # #endregion
        pass
    
    connector = aiohttp.TCPConnector(ssl=ssl_context, resolver=resolver)
    # #region agent log
    _debug_log("nesting_center_api.py:262", "TCPConnector created", {"connector_type": type(connector).__name__, "has_custom_resolver": resolver is not None}, "D")
    # #endregion
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # #region agent log
        _debug_log("nesting_center_api.py:198", "ClientSession created, about to start computation", {"session_type": type(session).__name__}, "E")
        # #endregion
        # Start the nesting job
        try:
            job_url = await Nesting.start_computation(session, nesting_data)
        except Exception as e:
            # #region agent log
            _debug_log("nesting_center_api.py:203", "exception in start_computation", {"error": str(e), "error_type": type(e).__name__, "error_class": str(type(e)), "error_args": str(e.args) if hasattr(e, 'args') else None}, "E")
            # #endregion
            # Re-raise with more context if it's an authentication error
            error_msg = str(e)
            if "authentication token" in error_msg or "access_denied" in error_msg or "AADB2C" in error_msg:
                raise Exception(f"Authentication failed. Please check your credentials in NestingCredentials.py. Original error: {error_msg}")
            raise Exception(f"Failed to start nesting computation: {error_msg}")
        
        if not job_url:
            raise Exception("Failed to start nesting computation: No job URL returned")
        
        logger.info(f"Nesting job started: {job_url}")
        
        # Wait for completion
        start_time = time.time()
        result = None
        
        while (time.time() - start_time) < max_wait:
            await asyncio.sleep(poll_interval)
            
            status = await Nesting.computation_status(session, job_url)
            
            if status is None:
                logger.warning("Failed to get computation status")
                continue
            
            state = status.get('StateString', '')
            logger.info(f"Nesting status: {state}")
            
            if state == 'Stopped':
                # Get the result
                result_version = status.get('ResultVersion')
                result = await Nesting.computation_result(session, job_url, result_version)
                break
            elif state == 'Failed':
                stop_details = await Nesting.computation_stop_details(session, job_url)
                raise Exception(f"Nesting computation failed: {stop_details}")
        
        # Clean up the job
        await Nesting.delete_computation(session, job_url)
        
        if result is None:
            raise Exception("Nesting computation timed out")
        
        return result


def parse_nesting_result(result: Dict, original_request_data: Optional[Dict] = None) -> Dict:
    """
    Parse the Nesting Center result into a standardized format.
    
    Args:
        result: Raw result from Nesting Center
        
    Returns:
        Parsed result with layouts, statistics, etc.
    """
    parsed = {
        "success": True,
        "total_parts_nested": 0,
        "total_plates_used": 0,
        "layouts": [],
        "statistics": {},
        "svg_layouts": []
    }
    
    if not result:
        parsed["success"] = False
        logger.warning("parse_nesting_result: result is None or empty")
        return parsed
    
    # Debug: log result structure
    logger.info(f"parse_nesting_result: result keys = {list(result.keys())}")
    logger.debug(f"parse_nesting_result: Full result structure (first 2000 chars): {json.dumps(result, indent=2)[:2000]}")
    
    # Extract the result data - try multiple possible structures
    nesting_result = result.get('Result', {})
    if not nesting_result:
        # Try direct access if Result key doesn't exist
        nesting_result = result
    
    logger.info(f"parse_nesting_result: nesting_result keys = {list(nesting_result.keys())}")
    
    # Get total parts nested - try multiple field names
    parsed["total_parts_nested"] = (
        nesting_result.get('NP', 0) or 
        nesting_result.get('NestedParts', 0) or
        nesting_result.get('TotalParts', 0) or
        0
    )
    
    logger.info(f"parse_nesting_result: total_parts_nested = {parsed['total_parts_nested']}")
    
    # Get context for SVG generation - store full context for later use
    # SvgCreator needs the original Context with Parts and RawPlates from the request
    # The API result doesn't include this, so we need to use the original request
    context = None
    if original_request_data and 'Context' in original_request_data:
        context = original_request_data['Context']
        logger.info("Using original request Context for SVG generation")
    elif result.get('Context'):
        context = result.get('Context')
        logger.info("Using Context from API result for SVG generation")
    else:
        logger.warning("No Context available - SVG generation may fail")
        context = result  # Fallback
    
    # Store both for SVG generation
    parsed['_context'] = context
    parsed['_full_result'] = result
    parsed['_original_request'] = original_request_data  # Store original request
    
    # First, try to get RawPlatesNested (the actual API response format)
    raw_plates_nested = nesting_result.get('RawPlatesNested', [])
    if not raw_plates_nested:
        raw_plates_nested = nesting_result.get('raw_plates_nested', [])
    
    # Process layouts - try multiple possible locations
    layouts = nesting_result.get('Layouts', [])
    if not layouts:
        layouts = nesting_result.get('layouts', [])
    if not layouts:
        layouts = result.get('Layouts', [])
    if not layouts:
        layouts = result.get('layouts', [])
    
    # If we have RawPlatesNested but no Layouts, use RawPlatesNested
    if raw_plates_nested and not layouts:
        logger.info(f"parse_nesting_result: Using RawPlatesNested (found {len(raw_plates_nested)} plates)")
        parsed["total_plates_used"] = len(raw_plates_nested)
        
        # Create layout entries from RawPlatesNested
        for i, raw_plate in enumerate(raw_plates_nested):
            # Log the structure of the raw plate for debugging
            logger.info(f"RawPlate {i} keys: {list(raw_plate.keys()) if isinstance(raw_plate, dict) else 'Not a dict'}")
            
            # Extract parts from RawPlate - try multiple field names
            parts_on_plate = raw_plate.get('PartsNested', [])
            if not parts_on_plate:
                parts_on_plate = raw_plate.get('parts_nested', [])
            if not parts_on_plate:
                parts_on_plate = raw_plate.get('Parts', [])
            if not parts_on_plate:
                parts_on_plate = raw_plate.get('parts', [])
            if not parts_on_plate:
                # Try nested structures - sometimes parts are in a nested object
                if 'PartsNested' in str(raw_plate):
                    logger.warning(f"Plate {i}: 'PartsNested' key exists but is empty or not a list")
            
            # Count parts if available
            parts_count = len(parts_on_plate) if parts_on_plate else 0
            
            # If we still have no parts, log the full raw_plate structure
            if parts_count == 0:
                area_parts = raw_plate.get('AreaPartsNested', 0)
                logger.warning(f"Plate {i}: No parts found in PartsNested. RawPlate structure: {json.dumps(raw_plate, indent=2)[:1000]}")
                if area_parts > 0:
                    logger.warning(f"Plate {i}: Has area ({area_parts} mm²) but no parts list - parts may be in different location")
            else:
                logger.info(f"Plate {i}: Found {parts_count} parts in PartsNested")
            
            # Calculate utilization and scrap
            area_parts = raw_plate.get('AreaPartsNested', 0)
            area_board_used = raw_plate.get('AreaRawPlateUsed', 0)
            area_board_total = raw_plate.get('AreaRawPlateTotal', 0)
            area_board = area_board_used if area_board_used > 0 else area_board_total
            
            utilization = (area_parts / area_board * 100) if area_board > 0 else 0
            scrap_percentage = max(0, 100 - utilization)
            
            # Get board dimensions from RawPlate or Context
            board_width = 0
            board_height = 0
            board_label = f'Board {i+1}'
            
            # Try to get from RawPlate first
            if 'RectangularShape' in raw_plate:
                shape = raw_plate['RectangularShape']
                board_width = shape.get('Width', 0)
                board_height = shape.get('Length', 0)
            elif 'Width' in raw_plate and 'Length' in raw_plate:
                board_width = raw_plate.get('Width', 0)
                board_height = raw_plate.get('Length', 0)
            
            # If not found, try to get from Context
            if (board_width == 0 or board_height == 0) and parsed.get('_context'):
                context = parsed.get('_context')
                if 'Problem' in context:
                    problem = context['Problem']
                    raw_plates = problem.get('RawPlates', [])
                    if raw_plates and len(raw_plates) > 0:
                        # Use the first raw plate for dimensions (all plates of same type)
                        first_plate = raw_plates[0]
                        if 'RectangularShape' in first_plate:
                            shape = first_plate['RectangularShape']
                            board_width = shape.get('Width', 0)
                            board_height = shape.get('Length', 0)
                        board_label = first_plate.get('Label', f'Board {i+1}')
            
            # Fallback: calculate from area if we have area but no dimensions
            if (board_width == 0 or board_height == 0) and area_board > 0:
                # Assume square board as fallback
                board_width = board_height = (area_board ** 0.5)
            
            layout_info = {
                "plate_index": i,
                "parts_nested": parts_count,
                "parts": parts_on_plate if parts_on_plate else [],  # Include actual parts data
                "area_parts_nested": area_parts,
                "area_raw_plate_total": area_board_total,
                "area_raw_plate_used": area_board_used,
                "utilization": utilization,  # Already a percentage (0-100)
                "scrap_percentage": scrap_percentage,
                "board": {
                    "width_mm": board_width,
                    "height_mm": board_height,
                    "label": board_label
                }
            }
            
            logger.info(f"Plate {i}: {parts_count} parts nested, area: {area_parts:,.0f} mm², utilization: {utilization:.2f}%")
            parsed["layouts"].append(layout_info)
            
            # Try to generate SVG for this plate
            # SvgCreator expects: context with Context.Problem.Parts and Context.Problem.RawPlates, 
            # and layout with PartsNested and RawPlateIndex
            if parts_on_plate:
                try:
                    # SvgCreator needs Context with Problem.Parts and Problem.RawPlates
                    context_for_svg = parsed.get('_context')
                    if not context_for_svg:
                        logger.warning(f"Missing Context - cannot generate SVG for plate {i}")
                    else:
                        # Verify Context has the required structure
                        if 'Problem' not in context_for_svg:
                            logger.warning(f"Context missing Problem - cannot generate SVG for plate {i}")
                        else:
                            problem = context_for_svg['Problem']
                            raw_plates = problem.get('RawPlates', [])
                            parts = problem.get('Parts', [])
                            
                            if not raw_plates:
                                logger.warning(f"No RawPlates in Context - cannot generate SVG for plate {i}")
                            elif not parts:
                                logger.warning(f"No Parts in Context - cannot generate SVG for plate {i}")
                            else:
                                # For single board type scenarios, all plates use RawPlateIndex 0
                                # The RawPlates array contains the board types, and all nested plates use the first one
                                raw_plate_idx = 0
                                
                                # Create layout structure for SvgCreator
                                layout_for_svg = {
                                    'RawPlateIndex': raw_plate_idx,
                                    'PartsNested': parts_on_plate
                                }
                                
                                # Create context structure for SvgCreator (it expects {"Context": {...}})
                                result_for_svg = {'Context': context_for_svg}
                                
                                try:
                                    svg = SvgCreator.createSvgNestingLayout(result_for_svg, layout_for_svg)
                                    parsed["svg_layouts"].append({
                                        "plate_index": i,
                                        "svg": svg,
                                        "board": layout_info.get("board", {}),
                                        "parts_nested": parts_count,
                                        "utilization": utilization,  # Already a percentage (0-100)
                                        "scrap_percentage": scrap_percentage,
                                        "nested_parts": parts_on_plate if parts_on_plate else []
                                    })
                                    logger.info(f"✅ Generated SVG for plate {i} (RawPlateIndex: {raw_plate_idx}, Parts: {len(parts_on_plate)})")
                                except Exception as svg_error:
                                    logger.warning(f"❌ SVG generation failed for plate {i}: {svg_error}")
                                    import traceback
                                    logger.debug(f"Traceback:\n{traceback.format_exc()}")
                except Exception as e:
                    logger.warning(f"Failed to generate SVG for plate {i}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            else:
                logger.warning(f"Skipping SVG generation for plate {i} - no parts data available")
    else:
        logger.info(f"parse_nesting_result: found {len(layouts)} layouts")
        parsed["total_plates_used"] = len(layouts) if layouts else len(raw_plates_nested)
        
        for i, layout in enumerate(layouts):
            # Handle different layout structures
            parts_nested = layout.get('PartsNested', [])
            if not parts_nested:
                parts_nested = layout.get('parts_nested', [])
            if not parts_nested:
                parts_nested = layout.get('Parts', [])
            if not parts_nested:
                parts_nested = layout.get('parts', [])
            
            layout_info = {
                "plate_index": layout.get('RawPlateIndex', layout.get('raw_plate_index', i)),
                "parts_nested": len(parts_nested),
                "parts": parts_nested
            }
            parsed["layouts"].append(layout_info)
            
            # Generate SVG for visualization
            try:
                # Use full result with Context for SVG generation
                result_for_svg = parsed.get('_full_result', result)
                # Ensure Context is available for SvgCreator
                if 'Context' not in result_for_svg and parsed.get('_context'):
                    # Reconstruct result with Context for SVG generation
                    result_for_svg = {'Context': parsed.get('_context'), **result_for_svg}
                svg = SvgCreator.createSvgNestingLayout(result_for_svg, layout)
                parsed["svg_layouts"].append({
                    "plate_index": i,
                    "svg": svg
                })
            except Exception as e:
                logger.warning(f"Failed to generate SVG for layout {i}: {e}")
    
    # Calculate statistics
    if parsed["total_plates_used"] > 0:
        parsed["statistics"] = {
            "average_parts_per_plate": parsed["total_parts_nested"] / parsed["total_plates_used"] if parsed["total_plates_used"] > 0 else 0,
            "plates_used": parsed["total_plates_used"]
        }
    else:
        logger.warning(f"parse_nesting_result: No layouts found! Result structure: {json.dumps(result, indent=2)[:500]}")
    
    return parsed


@nesting_center_bp.route('/status', methods=['GET'])
def check_status():
    """Check if the Nesting Center API is available."""
    if not NESTING_CENTER_AVAILABLE:
        return jsonify({
            'available': False,
            'error': f'Nesting Center modules not available: {IMPORT_ERROR}',
            'required_packages': ['msal', 'geomdl', 'aiohttp']
        }), 503
    
    return jsonify({
        'available': True,
        'service': 'Nesting Center Cloud API',
        'endpoints': [
            '/api/nesting-center/optimize',
            '/api/nesting-center/status'
        ]
    })


@nesting_center_bp.route('/optimize', methods=['POST'])
def optimize_nesting():
    """
    Main endpoint for Nesting Center cloud optimization.
    
    Expected JSON payload:
    {
        "parts": [
            {
                "id": "part1",
                "length_mm": 200,
                "width_mm": 100,
                "quantity": 5
            }
        ],
        "boards": [
            {
                "id": "board1",
                "width_mm": 3000,
                "height_mm": 1500,
                "quantity": 5
            }
        ],
        "settings": {
            "gap_mm": 5.0,
            "margin_mm": 5.0,
            "rotation": "Fixed90",
            "timeout": 60
        }
    }
    
    If boards are not provided, they will be loaded from the database.
    """
    if not NESTING_CENTER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': f'Nesting Center not available: {IMPORT_ERROR}'
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract parts
        parts_data = data.get('parts', [])
        if not parts_data:
            return jsonify({
                'success': False,
                'error': 'No parts provided'
            }), 400
        
        # Extract or load boards
        boards_data = data.get('boards', [])
        if not boards_data:
            # Try to load from database
            try:
                from DatabaseConfig import get_standard_boards
                boards_data = get_standard_boards()
            except Exception as e:
                logger.warning(f"Failed to load boards from database: {e}")
        
        if not boards_data:
            return jsonify({
                'success': False,
                'error': 'No boards provided and none available in database'
            }), 400
        
        # Extract settings
        settings_data = data.get('settings', {})
        
        # Convert to Nesting Center format
        nesting_parts = convert_parts_to_nesting_format(parts_data)
        raw_plates = convert_boards_to_nesting_format(boards_data)
        
        # Build nesting settings
        nesting_settings = {
            "DistancePartPart": float(settings_data.get('gap_mm', 5.0)),
            "DistancePartRawPlate": float(settings_data.get('margin_mm', 5.0)),
            "RotationControl": settings_data.get('rotation', 'Fixed90')
        }
        
        # Add scrap factor if provided
        scrap_factor = settings_data.get('scrap_factor')
        if scrap_factor is not None:
            scrap_factor = float(scrap_factor)
            if scrap_factor != 1.0:
                nesting_settings["ScrapFactor"] = scrap_factor
                logger.info(f"Including scrap factor {scrap_factor} in nesting request")
        
        stop_conditions = {
            "AllPartsNested": True,
            "Timeout": int(settings_data.get('timeout', 60))
        }
        
        # Build the request
        nesting_request = build_nesting_request(
            nesting_parts,
            raw_plates,
            nesting_settings,
            stop_conditions
        )
        
        logger.info(f"Starting Nesting Center computation with {len(nesting_parts)} parts on {len(raw_plates)} plates")
        
        # Run the async nesting computation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                run_nesting_computation(
                    nesting_request,
                    poll_interval=2.0,
                    max_wait=float(settings_data.get('timeout', 60)) + 30
                )
            )
        finally:
            loop.close()
        
        # Parse and return the result
        parsed_result = parse_nesting_result(result)
        
        return jsonify(parsed_result)
        
    except Exception as e:
        logger.error(f"Nesting Center error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@nesting_center_bp.route('/convert-dxf', methods=['POST'])
def convert_dxf_to_part():
    """
    Convert a DXF file to a nesting part format.
    
    Expected: multipart/form-data with 'file' field containing DXF
    """
    if not NESTING_CENTER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': f'Nesting Center not available: {IMPORT_ERROR}'
        }), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Read the file data
        file_data = file.read()
        
        # Convert using Nesting Center converter
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def convert():
                # Create SSL context that doesn't verify certificates (for testing)
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    return await NestingConverters.convert_part(session, file_data)
            
            result = loop.run_until_complete(convert())
        finally:
            loop.close()
        
        if result:
            return jsonify({
                'success': True,
                'part': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to convert DXF file'
            }), 500
            
    except Exception as e:
        logger.error(f"DXF conversion error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Utility function for direct usage (not through Flask)
# This function uses ONLY the Nesting Center API from .cursor/nesting folder
def run_nesting_sync(
    parts: List[Dict],
    boards: List[Dict],
    gap_mm: float = 5.0,
    margin_mm: float = 5.0,
    rotation: str = "Fixed90",
    timeout: int = 60,
    scrap_factor: float = 1.20
) -> Dict:
    """
    Synchronous wrapper for running nesting computation using Nesting Center API.
    
    This function uses ONLY the Nesting Center cloud API from .cursor/nesting/Nesting.py
    It sends the nesting request to https://api-nesting.nestingcenter.com/
    
    Args:
        parts: List of parts with length_mm, width_mm, quantity
        boards: List of boards with width_mm, height_mm, quantity
        gap_mm: Gap between parts
        margin_mm: Margin from plate edges
        rotation: Rotation control ("Fixed90", "Free", "None")
        timeout: Maximum computation time in seconds
        scrap_factor: Scrap factor (1.20 = 20% waste, 1.0 = no waste)
        
    Returns:
        Nesting result dictionary from Nesting Center API
    """
    if not NESTING_CENTER_AVAILABLE:
        raise ImportError(f"Nesting Center not available: {IMPORT_ERROR}")
    
    # Validate scrap factor
    if scrap_factor < 0.5 or scrap_factor > 3.0:
        logger.warning(f"Scrap factor {scrap_factor} is outside typical range (0.5-3.0). Verify this is correct.")
    
    # Validate inputs
    if not parts:
        raise ValueError("No parts provided for nesting")
    if not boards:
        raise ValueError("No boards provided for nesting")
    
    # Log input validation
    total_parts_quantity = sum(p.get('quantity', 1) for p in parts)
    logger.info(f"run_nesting_sync: Processing {len(parts)} part types ({total_parts_quantity} total pieces) on {len(boards)} board types")
    logger.info(f"run_nesting_sync: Scrap factor = {scrap_factor:.4f} ({(scrap_factor - 1.0) * 100:.2f}% waste)")
    
    # Convert formats for Nesting Center API
    logger.info("Converting parts to Nesting Center format...")
    nesting_parts = convert_parts_to_nesting_format(parts)
    
    logger.info("Converting boards to Nesting Center format...")
    raw_plates = convert_boards_to_nesting_format(boards)
    
    if not nesting_parts:
        raise ValueError("Failed to convert parts to Nesting Center format")
    if not raw_plates:
        raise ValueError("Failed to convert boards to Nesting Center format")
    
    # Log what's being sent to API
    logger.info(f"\n{'='*80}")
    logger.info("📤 SENDING TO NESTING CENTER API:")
    logger.info(f"{'='*80}")
    logger.info(f"PARTS ({len(nesting_parts)} types):")
    total_qty = 0
    for part in nesting_parts:
        qty = part.get('Quantity', 1)
        total_qty += qty
        shape = part.get('RectangularShape', {})
        length = shape.get('Length', 0)
        width = shape.get('Width', 0)
        logger.info(f"  - {part.get('Label', 'no-label')}: {length}×{width}mm × {qty} pcs")
    logger.info(f"  Total pieces: {total_qty}")
    
    logger.info(f"\nBOARDS ({len(raw_plates)} types):")
    for plate in raw_plates:
        qty = plate.get('Quantity', 1)
        shape = plate.get('RectangularShape', {})
        length = shape.get('Length', 0)
        width = shape.get('Width', 0)
        logger.info(f"  - {plate.get('Label', 'no-label')}: {length}×{width}mm × {qty} pcs")
    logger.info(f"{'='*80}\n")
    
    # Build request for Nesting Center API
    # Include scrap factor in settings (API may use it for optimization)
    nesting_settings = {
        "DistancePartPart": float(gap_mm),
        "DistancePartRawPlate": float(margin_mm),
        "RotationControl": rotation
    }
    
    # ALWAYS include scrap factor in settings (even if 1.0) for transparency
    nesting_settings["ScrapFactor"] = float(scrap_factor)
    logger.info(f"Including scrap factor {scrap_factor:.4f} in nesting request settings")
    
    # Log the complete request structure for debugging
    logger.debug(f"Nesting request settings: {json.dumps(nesting_settings, indent=2)}")
    
    nesting_request = build_nesting_request(
        nesting_parts,
        raw_plates,
        nesting_settings,
        {
            "AllPartsNested": True,
            "Timeout": timeout
        }
    )
    
    # Verify scrap factor is in the request
    request_settings = nesting_request.get('Context', {}).get('Settings', {})
    request_problem = nesting_request.get('Context', {}).get('Problem', {})
    
    logger.info("🔍 VERIFYING API REQUEST STRUCTURE:")
    logger.info(f"  ✓ Settings included: {list(request_settings.keys())}")
    logger.info(f"  ✓ ScrapFactor in settings: {'ScrapFactor' in request_settings}")
    if 'ScrapFactor' in request_settings:
        logger.info(f"    ScrapFactor value: {request_settings['ScrapFactor']}")
    logger.info(f"  ✓ Parts in request: {len(request_problem.get('Parts', []))} types")
    logger.info(f"  ✓ RawPlates in request: {len(request_problem.get('RawPlates', []))} types")
    
    # Log full request structure for debugging (truncated if too large)
    try:
        request_json = json.dumps(nesting_request, indent=2)
        if len(request_json) > 2000:
            logger.debug(f"Request structure (first 2000 chars):\n{request_json[:2000]}...")
        else:
            logger.debug(f"Full request structure:\n{request_json}")
    except Exception as e:
        logger.warning(f"Could not serialize request for logging: {e}")
    
    # Run computation using Nesting Center API (from .cursor/nesting/Nesting.py)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # This calls the Nesting Center cloud API via .cursor/nesting/Nesting.py
        result = loop.run_until_complete(
            run_nesting_computation(nesting_request, poll_interval=2.0, max_wait=timeout + 30)
        )
    finally:
        loop.close()
    
    # Parse and return the result from Nesting Center API
    logger.info("📥 PARSING NESTING RESULT FROM API...")
    logger.info(f"🔍 Result structure preview - Top level keys: {list(result.keys())[:20] if result else 'None'}")
    # Store the full result in parsed_result for SVG generation (SvgCreator needs the full Context)
    # Pass the original nesting_request so we can reconstruct Context for SVG generation if needed
    parsed_result = parse_nesting_result(result, original_request_data=nesting_request)
    # Store the original result for SVG generation
    parsed_result['_original_result'] = result
    
    # Add scrap factor info to result for verification
    parsed_result['scrap_factor_sent'] = scrap_factor
    parsed_result['scrap_factor_waste_percentage'] = (scrap_factor - 1.0) * 100
    
    # Log detailed nesting result
    logger.info(f"\n{'='*80}")
    logger.info("📊 NESTING RESULT SUMMARY:")
    logger.info(f"{'='*80}")
    logger.info(f"Success: {parsed_result.get('success', False)}")
    logger.info(f"Total plates used: {parsed_result.get('total_plates_used', 0)}")
    logger.info(f"Total parts nested: {parsed_result.get('total_parts_nested', 0)}")
    logger.info(f"Layouts returned: {len(parsed_result.get('layouts', []))}")
    logger.info(f"SVG layouts generated: {len(parsed_result.get('svg_layouts', []))}")
    
    # Log details of each layout
    layouts = parsed_result.get('layouts', [])
    svg_layouts = parsed_result.get('svg_layouts', [])
    if layouts:
        logger.info(f"\n📐 LAYOUT DETAILS:")
        for i, layout in enumerate(layouts, 1):
            parts_count = layout.get('parts_nested', 0)
            plate_idx = layout.get('plate_index', i - 1)
            area_parts = layout.get('area_parts_nested', 0)
            area_board = layout.get('area_raw_plate_used', layout.get('area_raw_plate_total', 0))
            utilization = (area_parts / area_board * 100) if area_board > 0 else 0
            scrap = 100 - utilization
            
            # Check if SVG was generated
            has_svg = any(sl.get('plate_index') == plate_idx for sl in svg_layouts)
            
            logger.info(f"  Plate {i} (index {plate_idx}):")
            logger.info(f"    - Parts nested: {parts_count}")
            logger.info(f"    - Parts area: {area_parts:,.0f} mm²")
            logger.info(f"    - Board area used: {area_board:,.0f} mm²")
            logger.info(f"    - Utilization: {utilization:.2f}%")
            logger.info(f"    - Scrap: {scrap:.2f}%")
            logger.info(f"    - SVG visualization: {'✅ Available' if has_svg else '❌ Not available'}")
            
            # Log parts on this plate if available
            parts_list = layout.get('parts', [])
            if parts_list and len(parts_list) > 0:
                logger.info(f"    - Parts on plate:")
                for j, part in enumerate(parts_list[:5], 1):  # Log first 5 parts
                    if isinstance(part, dict):
                        length = part.get('Length', part.get('length_mm', 0))
                        width = part.get('Width', part.get('width_mm', 0))
                        label = part.get('Label', part.get('label', f'part_{j}'))
                        x_pos = part.get('X', part.get('x', 'N/A'))
                        y_pos = part.get('Y', part.get('y', 'N/A'))
                        rotation = part.get('Rotation', part.get('rotation', 0))
                        logger.info(f"      {j}. {label}: {length}×{width}mm @ ({x_pos}, {y_pos}) rot:{rotation}°")
                if len(parts_list) > 5:
                    logger.info(f"      ... and {len(parts_list) - 5} more parts")
    
    logger.info(f"{'='*80}\n")
    
    return parsed_result


def register_nesting_center_api(app):
    """Register the Nesting Center API with the Flask app."""
    app.register_blueprint(nesting_center_bp)
    if NESTING_CENTER_AVAILABLE:
        print("✅ Nesting Center API registered successfully")
    else:
        print(f"⚠️ Nesting Center API registered (limited - missing dependencies: {IMPORT_ERROR})")


# Example usage when run directly
if __name__ == "__main__":
    # Example: Run a simple nesting test
    print("Testing Nesting Center API...")
    
    test_parts = [
        {"id": "part1", "length_mm": 200, "width_mm": 100, "quantity": 10}
    ]
    
    test_boards = [
        {"id": "board1", "width_mm": 3000, "height_mm": 1500, "quantity": 5}
    ]
    
    try:
        result = run_nesting_sync(test_parts, test_boards, timeout=30)
        print(f"Success! Nested {result['total_parts_nested']} parts on {result['total_plates_used']} plates")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

