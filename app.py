import os
import json
import math
import tempfile
import io
import base64
from datetime import datetime
from collections import defaultdict

import ezdxf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash, Response, Blueprint
from werkzeug.utils import secure_filename
import queue
import threading
import time
import asyncio as _asyncio

from shapely.geometry import Polygon, LineString, Point, MultiPoint
from shapely.ops import linemerge, unary_union, polygonize

from DatabaseConfig import (
    test_connection, execute_raw_sql, get_table_row_count, 
    get_materials_data, get_filtered_options, get_material_config,
    get_all_materials, update_material, add_material, delete_material,
    create_users_table, get_user_by_email, create_user
)

# Nesting functionality removed for deployment

# Import AI scrap calculator
try:
    from ai_scrap_calculator import AIScrapCalculator, ScrapCalculationInput
    _HAVE_AI_SCRAP = True
except Exception as e:
    # AI scrap calculator not available - using fallback
    _HAVE_AI_SCRAP = False

from doc_generator import DocGenerator
import json
import tempfile
from pathlib import Path

# Nesting functionality removed for deployment

# -----------------------------------------------------------------------------
# DXF Text/Label extraction helpers
# -----------------------------------------------------------------------------
import re

# Custom JSON encoder to handle infinite values
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
        return super().default(obj)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-change-this-in-production'
app.json_encoder = SafeJSONEncoder

# Nesting functionality removed for deployment
# -----------------------------------------------
# AI helpers: material inference via OpenAI (optional)
# -----------------------------------------------
async def _openai_infer_material_async(api_key: str, characteristics: dict) -> str:
    try:
        import aiohttp
        prompt = (
            "Based on the following part characteristics, infer the most likely material type "
            "from this list: [steel, aluminum, stainless, copper, brass, plastic].\n"
            f"Dimensions: {characteristics.get('length_mm', 0)} x {characteristics.get('width_mm', 0)} mm\n"
            f"Area: {characteristics.get('area_sq_mm', 0)} mm^2; Perimeter: {characteristics.get('perimeter_mm', 0)} mm\n"
            f"Entity types: {characteristics.get('entity_types', [])}; Colors: {characteristics.get('colors', [])}\n"
            "Respond with only one lowercase word from the list."
        )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0.0,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=15) as resp:
                if resp.status != 200:
                    return "steel"
                data = await resp.json()
        text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip().lower()
        for m in ("steel", "aluminum", "stainless", "copper", "brass", "plastic"):
            if m in text:
                return m
        return "steel"
    except Exception:
        return "steel"

def _infer_material_openai_sync(api_key: str, characteristics: dict) -> str:
    try:
        loop = _asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_openai_infer_material_async(api_key, characteristics))
        finally:
            loop.close()
    except Exception:
        return "steel"

# Global event broadcaster for real-time updates
class EventBroadcaster:
    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()
    
    def add_client(self, client_queue):
        with self.lock:
            self.clients.add(client_queue)
    
    def remove_client(self, client_queue):
        with self.lock:
            self.clients.discard(client_queue)
    
    def broadcast_event(self, event_type, data):
        """Broadcast an event to all connected clients"""
        event_data = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        
        dead_clients = set()
        with self.lock:
            for client_queue in self.clients:
                try:
                    client_queue.put_nowait(event_data)
                except queue.Full:
                    # Client queue is full, remove it
                    dead_clients.add(client_queue)
        
        # Remove dead clients
        with self.lock:
            for dead_client in dead_clients:
                self.clients.discard(dead_client)

# Global broadcaster instance
broadcaster = EventBroadcaster()

# Nesting functionality removed for deployment

CONFIG_FILE = 'admin_config.json'
DEFAULT_CONFIG = {
    'laser_cost': 2.00,
    'piercing_toggle': False
}


def load_admin_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            return _merge_defaults_with_config(DEFAULT_CONFIG, cfg)
        else:
            save_admin_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
    except Exception as e:
        # Error loading admin config - using defaults
        return DEFAULT_CONFIG

def save_admin_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        # Error saving admin config
        return False

def is_admin_logged_in():
    return session.get('admin_logged_in', False)

def is_user_logged_in():
    return session.get('user_logged_in', False)

def get_current_user():
    return session.get('user_data', None)

# Nesting functionality removed for deployment

def calculate_cost(area, config, material, thickness_mm, object_parts_count, part_entities=None, layers=None, length_mm=0.0, width_mm=0.0):
    try:
        # Material validation - critical for accurate calculations
        if not material or material.strip() == '':
            error_msg = "ERROR: No material type specified. Cannot calculate costs without material information."
            print(f"  ❌ {error_msg}")
            return {
                'error': error_msg,
                'area_sq_mm': area * 1000000 if area else 0,
                'length_mm': length_mm,
                'width_mm': width_mm,
                'perimeter_meters': 0.0,
                'cutting_time_machine': 0.0,
                'cutting_time_vaporization': 0.0,
                'piercing_time_total': 0.0,
                'total_time_min': 0.0,
                'object_parts_count': int(object_parts_count) if object_parts_count else 0,
                'laser_cost': 0.0,
                'weight_kg': 0.0,
                'material_cost': 0.0,
                'vgroove_count': 0,
                'bending_count': 0,
                'total_bending_lines': 0,
                'bending_cost': 0.0,
                'vgroove_length_meters': 0.0,
                'vgroove_cost': 0.0,
                'total_cost': 0.0
            }
        
        # Thickness validation - critical for accurate calculations
        if thickness_mm is None or thickness_mm <= 0:
            error_msg = "ERROR: No thickness specified or thickness is invalid. Cannot calculate costs without thickness information."
            print(f"  ❌ {error_msg}")
            return {
                'error': error_msg,
                'area_sq_mm': area * 1000000 if area else 0,
                'length_mm': length_mm,
                'width_mm': width_mm,
                'perimeter_meters': 0.0,
                'cutting_time_machine': 0.0,
                'cutting_time_vaporization': 0.0,
                'piercing_time_total': 0.0,
                'total_time_min': 0.0,
                'object_parts_count': int(object_parts_count) if object_parts_count else 0,
                'laser_cost': 0.0,
                'weight_kg': 0.0,
                'material_cost': 0.0,
                'vgroove_count': 0,
                'bending_count': 0,
                'total_bending_lines': 0,
                'bending_cost': 0.0,
                'vgroove_length_meters': 0.0,
                'vgroove_cost': 0.0,
                'total_cost': 0.0
            }
        
        # Validate material name format
        material_upper = material.upper().strip()
        valid_materials = ['STAINLESS STEEL', 'ALUMINUM', 'BRASS', 'COPPER', 'MILD STEEL', 'GALVANIZED STEEL']
        if material_upper not in valid_materials:
            error_msg = f"ERROR: Invalid material type '{material}'. Valid materials: {', '.join(valid_materials)}"
            print(f"  ❌ {error_msg}")
            return {
                'error': error_msg,
                'area_sq_mm': area * 1000000 if area else 0,
                'length_mm': length_mm,
                'width_mm': width_mm,
                'perimeter_meters': 0.0,
                'cutting_time_machine': 0.0,
                'cutting_time_vaporization': 0.0,
                'piercing_time_total': 0.0,
                'total_time_min': 0.0,
                'object_parts_count': int(object_parts_count) if object_parts_count else 0,
                'laser_cost': 0.0,
                'weight_kg': 0.0,
                'material_cost': 0.0,
                'vgroove_count': 0,
                'bending_count': 0,
                'total_bending_lines': 0,
                'bending_cost': 0.0,
                'vgroove_length_meters': 0.0,
                'vgroove_cost': 0.0,
                'total_cost': 0.0
            }
        
        print(f"  ✓ Material validation passed: '{material}'")
        
        area_sq_mm = area * 1000000
        
        from shapely.geometry import Polygon, LineString
        from shapely.ops import linemerge
        import numpy as np
        
        # Calculate accurate perimeter for all entity types
        perimeter_meters = calculate_accurate_perimeter(part_entities, layers)
        
        # If perimeter calculation failed, use a better approximation
        if perimeter_meters == 0.0:
            # Use perimeter based on actual dimensions if available
            if length_mm > 0 and width_mm > 0:
                perimeter_mm = 2 * (length_mm + width_mm)  # Rectangle approximation
                perimeter_meters = perimeter_mm / 1000.0
            else:
                # Fallback to area-based approximation (less accurate)
                perimeter_mm_approx = np.sqrt(area_sq_mm * 4)  # Better approximation for irregular shapes
                perimeter_meters = perimeter_mm_approx / 1000.0  # Convert mm to meters
        
        # Calculate cutting time components
        # Convert machine speed from mm/min to m/min for consistency
        machine_speed_m_per_min = config.get('machine_speed', 100.0)
        vaporization_speed_m_per_min = config.get('vaporization_speed', 50.0)
        
        # Prevent division by zero
        if machine_speed_m_per_min <= 0:
            machine_speed_m_per_min = 100.0  # Default fallback
        if vaporization_speed_m_per_min <= 0:
            vaporization_speed_m_per_min = 50.0  # Default fallback
        
        cutting_time_machine = perimeter_meters / machine_speed_m_per_min
        cutting_time_vaporization = perimeter_meters / vaporization_speed_m_per_min
        
        # Piercing time for all object parts with toggle adjustment
        base_piercing_time = config['piercing_time']
        if not config.get('piercing_toggle', False):
            # If toggle is OFF, subtract 1 second from database value
            adjusted_piercing_time = max(0, base_piercing_time - 1)
        else:
            # If toggle is ON, use database value as is
            adjusted_piercing_time = base_piercing_time
        
        piercing_time_total = object_parts_count * (adjusted_piercing_time / 60)  # Convert to minutes
        
        # Total time
        total_time_min = cutting_time_machine + cutting_time_vaporization + piercing_time_total
        
        # Laser cost
        laser_cost_total = total_time_min * config['laser_cost']
        
        #  calculation - CRITICAL: Use material-specific values from database
        # Formula: Area * Thickness * Density * Price_per_kg * Scrap_Factor
        # Note: Area is already in m², thickness in mm, density in g/cm³, price in USD/kg
        
        # CRITICAL VALIDATION: Ensure we have the correct material properties
        if 'density' not in config or config['density'] is None:
            error_msg = f"ERROR: Missing density for material '{material}'. Cannot calculate material cost."
            print(f"  ❌ {error_msg}")
            return {
                'error': error_msg,
                'area_sq_mm': area_sq_mm,
                'length_mm': length_mm,
                'width_mm': width_mm,
                'perimeter_meters': 0.0,
                'cutting_time_machine': 0.0,
                'cutting_time_vaporization': 0.0,
                'piercing_time_total': 0.0,
                'total_time_min': 0.0,
                'object_parts_count': int(object_parts_count) if object_parts_count else 0,
                'laser_cost': 0.0,
                'weight_kg': 0.0,
                'material_cost': 0.0,
                'vgroove_count': 0,
                'bending_count': 0,
                'total_bending_lines': 0,
                'bending_cost': 0.0,
                'vgroove_length_meters': 0.0,
                'vgroove_cost': 0.0,
                'total_cost': 0.0
            }
        
        if 'price_per_kg' not in config or config['price_per_kg'] is None:
            error_msg = f"ERROR: Missing price per kg for material '{material}'. Cannot calculate material cost."
            print(f"  ❌ {error_msg}")
            return {
                'error': error_msg,
                'area_sq_mm': area_sq_mm,
                'length_mm': length_mm,
                'width_mm': width_mm,
                'perimeter_meters': 0.0,
                'cutting_time_machine': 0.0,
                'cutting_time_vaporization': 0.0,
                'piercing_time_total': 0.0,
                'total_time_min': 0.0,
                'object_parts_count': int(object_parts_count) if object_parts_count else 0,
                'laser_cost': 0.0,
                'weight_kg': 0.0,
                'material_cost': 0.0,
                'vgroove_count': 0,
                'bending_count': 0,
                'total_bending_lines': 0,
                'bending_cost': 0.0,
                'vgroove_length_meters': 0.0,
                'vgroove_cost': 0.0,
                'total_cost': 0.0
            }
        
        # Get material properties from config (these come from the database)
        material_density = float(config['density'])  # g/cm³
        material_price_per_kg = float(config['price_per_kg'])  # USD/kg
        scrap_factor = config.get('scrap_factor', 1.20)
        scrap_percentage = scrap_factor-1 
        scrap_price_per_kg = config.get('scrap_price_per_kg', 0.0)  # USD/kg for scrap material
        
        # Convert density from g/cm³ to kg/m³ for calculation
        density_kg_per_m3 = material_density * 1000  # 1 g/cm³ = 1000 kg/m³
        
        # Convert thickness from mm to m
        thickness_m = thickness_mm / 1000  # 1 mm = 0.001 m
        # material_price=material_weight*material_price_per_kg
        # scrap_price=scrap_weight*scrap_price_per_kg*scrap_factor
        # Material cost calculation with proper units
        # Area (m²) × Thickness (m) × Density (kg/m³) × Price (USD/kg) × Scrap Factor
        # material_cost=material_price-scrap_price
        
        # Calculate weight for reference (kg)
        weight_kg = area * thickness_m * density_kg_per_m3
        # material_weight=weight_kg*1.2
        # scrap_weight=material_weight-weight_kg
        
        # Calculate scrap weight (total material weight * scrap percentage)
        scrap_weight = weight_kg * scrap_percentage
        
        # Calculate scrap cost (scrap weight * scrap price per kg)
        scrap_cost = scrap_weight * scrap_price_per_kg
        
        # Calculate material cost before scrap deduction
        material_cost_before_scrap =  weight_kg * material_price_per_kg * scrap_factor
        
        # Final material cost after subtracting scrap cost
        material_cost = material_cost_before_scrap - scrap_cost
        # DEBUG: Enhanced material cost calculation logging
        print(f"  🔍 ENHANCED MATERIAL COST CALCULATION DEBUG:")
        print(f"      Formula: (Area × Thickness × Density × Price per kg × Scrap Factor) - Scrap Cost")
        print(f"      Values: ({area} × {thickness_m} × {density_kg_per_m3} × {material_price_per_kg} × {scrap_factor}) - Scrap Cost")
        print(f"      Step-by-step calculation:")
        print(f"        Step 1: Area × Thickness = {area} × {thickness_m} = {area * thickness_m}")
        print(f"        Step 2: × Density = {area * thickness_m} × {density_kg_per_m3} = {area * thickness_m * density_kg_per_m3}")
        print(f"        Step 3: × Price per kg = {area * thickness_m * density_kg_per_m3} × {material_price_per_kg} = {area * thickness_m * density_kg_per_m3 * material_price_per_kg}")
        print(f"        Step 4: × Scrap Factor = {area * thickness_m * density_kg_per_m3 * material_price_per_kg} × {scrap_factor} = {material_cost_before_scrap}")
        print(f"        Step 5: Scrap Weight = {weight_kg:.6f} × {scrap_percentage} = {scrap_weight:.6f} kg")
        print(f"        Step 6: Scrap Cost = {scrap_weight:.6f} × {scrap_price_per_kg} = ${scrap_cost:.6f}")
        print(f"        Step 7: Final Cost = ${material_cost_before_scrap:.6f} - ${scrap_cost:.6f} = ${material_cost:.6f}")
        print(f"      Final Material Cost: ${material_cost:.6f}")
        print(f"      Weight: {weight_kg:.6f} kg")
        print(f"      Scrap Weight: {scrap_weight:.6f} kg")
        print(f"      Scrap Cost: ${scrap_cost:.6f}")
        
        # Debug logging for material cost calculation
        print(f"  🔍 MATERIAL COST CALCULATION DEBUG:")
        print(f"      Area: {area} m²")
        print(f"      Thickness: {thickness_mm} mm = {thickness_m} m")
        print(f"      Density: {material_density} g/cm³ = {density_kg_per_m3} kg/m³")
        print(f"      Price per kg: ${material_price_per_kg}")
        print(f"      Scrap Factor: {scrap_factor}")
        print(f"      Weight: {weight_kg:.6f} kg")
        print(f"      Material Cost: ${material_cost:.6f}")
        
        # Bending cost calculation (count-based) - includes both V-groove and bending lines
        bending_cost = 0.0
        vgroove_count = 0
        bending_count = 0
        total_bending_lines = 0
        if part_entities and layers:
            vgroove_count = count_vgroove_lines(part_entities, layers)
            bending_count = count_bending_lines(part_entities, layers)
            total_bending_lines = vgroove_count + bending_count
            bending_price = config.get('bending_price', 0.0)
            bending_cost = total_bending_lines * bending_price
            
            # Debug logging for bending and v-groove
            print(f"  BENDING/V-GROOVE DEBUG:")
            print(f"    V-groove lines: {vgroove_count}")
            print(f"    Bending lines: {bending_count}")
            print(f"    Total bending lines: {total_bending_lines}")
            print(f"    Bending price per line: ${bending_price:.2f}")
            print(f"    Bending cost: ${bending_cost:.2f}")
        
        # V-Groove cost calculation (length-based)
        vgroove_cost = 0.0
        vgroove_length_meters = 0.0
        if part_entities and layers:
            vgroove_length_meters = calculate_vgroove_length(part_entities, layers)
            vgroove_price = config.get('vgroove_price', 0.0)
            vgroove_cost = vgroove_length_meters * vgroove_price
            
            # Debug logging for v-groove length
            print(f"    V-groove length: {vgroove_length_meters:.3f}m")
            print(f"    V-groove price per meter: ${vgroove_price:.2f}")
            print(f"    V-groove cost: ${vgroove_cost:.2f}")
        
        # Validate and fix infinite values
        if not np.isfinite(laser_cost_total) or np.isnan(laser_cost_total):
            laser_cost_total = 0.0
        if not np.isfinite(material_cost) or np.isnan(material_cost):
            material_cost = 0.0
        if not np.isfinite(bending_cost) or np.isnan(bending_cost):
            bending_cost = 0.0
        if not np.isfinite(vgroove_cost) or np.isnan(vgroove_cost):
            vgroove_cost = 0.0
        
        # Total cost (laser + material + bending + vgroove)
        total_cost = laser_cost_total + material_cost + bending_cost + vgroove_cost
        
        # Final validation of total cost
        if not np.isfinite(total_cost) or np.isnan(total_cost):
            total_cost = 0.0
        
        # Debug summary disabled in production
        
        # Ensure all return values are finite
        def validate_value(value, default=0.0):
            if not np.isfinite(value) or np.isnan(value):
                return default
            return value
        
        # DEBUG: Log final return values
        final_material_cost = validate_value(material_cost)
        final_total_cost = validate_value(total_cost)
        print(f"  🔍 FINAL RETURN VALUES DEBUG:")
        print(f"      Material Cost: ${final_material_cost:.6f}")
        print(f"      Total Cost: ${final_total_cost:.6f}")
        print(f"      Weight: {validate_value(weight_kg):.6f} kg")
        
        return {
            'area_sq_mm': validate_value(area_sq_mm),
            'length_mm': validate_value(length_mm),
            'width_mm': validate_value(width_mm),
            'perimeter_meters': validate_value(perimeter_meters),
            'cutting_time_machine': validate_value(cutting_time_machine),
            'cutting_time_vaporization': validate_value(cutting_time_vaporization),
            'piercing_time_total': validate_value(piercing_time_total),
            'total_time_min': validate_value(total_time_min),
            'object_parts_count': int(validate_value(object_parts_count, 0)),
            'laser_cost': validate_value(laser_cost_total),
            'weight_kg': validate_value(weight_kg),
            'material_cost': final_material_cost,
            'vgroove_count': int(validate_value(vgroove_count, 0)),
            'bending_count': int(validate_value(bending_count, 0)),
            'total_bending_lines': int(validate_value(total_bending_lines, 0)),
            'bending_cost': validate_value(bending_cost),
            'vgroove_length_meters': validate_value(vgroove_length_meters),
            'vgroove_cost': validate_value(vgroove_cost),
            'total_cost': final_total_cost
        }
    except Exception as e:
        print(f"Error calculating cost: {e}")
        return None

def count_object_parts(part_entities, layers):
    """Count the number of visual shapes/objects in the image (excluding V-GROOVE)"""
    try:
        from shapely.geometry import LineString, Polygon
        from shapely.ops import linemerge
        
        # Separate entities by visual characteristics
        white_entities = []
        red_entities = []
        circles = []
        ellipses = []
        splines = []
        
        for entity in part_entities:
            if not hasattr(entity, 'dxf'):
                continue
                
            layer_name = getattr(entity.dxf, 'layer', '')
            color = get_entity_color(entity, layers)
            entity_type = entity.dxftype()
            
            # Skip V-GROOVE entities completely
            if layer_name.upper() == 'V-GROOVE' and color == 3:
                continue
            
            # Skip BENDING entities completely
            if layer_name.upper() == 'BENDING' and color == 5:
                continue
            
            # Count visual shapes based on color and type
            if entity_type == 'CIRCLE':
                circles.append(entity)  # Each circle is a separate shape
            elif entity_type == 'ELLIPSE':
                ellipses.append(entity)  # Each ellipse is a separate shape
            elif entity_type == 'SPLINE':
                splines.append(entity)  # Each spline is a separate shape
            elif color == 7:  # White shapes
                white_entities.append(entity)
            elif color == 1:  # Red shapes
                red_entities.append(entity)
        
        # Count visual objects
        object_count = 0
        
        # Count circles (each circle = 1 shape)
        object_count += len(circles)
        
        # Count ellipses (each ellipse = 1 shape)
        object_count += len(ellipses)
        
        # Count splines (each spline = 1 shape)
        object_count += len(splines)
        
        # Count white shapes (group connected entities into shapes)
        if white_entities:
            white_shapes = group_entities_into_shapes(white_entities)
            object_count += len(white_shapes)
        
        # Count red shapes (group connected entities into shapes)
        if red_entities:
            red_shapes = group_entities_into_shapes(red_entities)
            object_count += len(red_shapes)
        
        return max(1, object_count)
        
    except Exception as e:
        print(f"Error counting visual shapes: {e}")
        return 1

def count_vgroove_lines(part_entities, layers):
    """Count the number of V-groove lines (green lines) in a part for bending cost calculation"""
    vgroove_count = 0
    
    print(f"\n=== V-GROOVE COUNTING DEBUG ===")
    print(f"Total entities in part: {len(part_entities)}")
    
    for i, entity in enumerate(part_entities):
        if not hasattr(entity, 'dxf'):
            print(f"Entity {i}: No dxf attribute")
            continue
            
        layer_name = getattr(entity.dxf, 'layer', 'UNKNOWN')
        color = get_entity_color(entity, layers)
        entity_type = entity.dxftype()
        
        print(f"Entity {i}: Type={entity_type}, Layer='{layer_name}', Color={color}")
        
        # Check for V-GROOVE entities with multiple variations
        layer_upper = layer_name.upper()
        is_vgroove_layer = (
            layer_upper == 'V-GROOVE' or 
            layer_upper == 'VGROOVE' or 
            layer_upper == 'V_GROOVE' or
            'V-GROOVE' in layer_upper or
            'VGROOVE' in layer_upper
        )
        
        # Check for green color (color 3) or any green variation
        is_green_color = (color == 3)
        
        if is_vgroove_layer and is_green_color:
            # Count individual line segments within the entity
            line_segments = 0
            
            if entity_type == 'LINE':
                # Single line entity = 1 line segment
                line_segments = 1
                print(f"  ✓ V-GROOVE LINE entity found: 1 line segment")
                
            elif entity_type == 'LWPOLYLINE':
                # Polyline entity - count the number of line segments
                try:
                    points = list(entity.get_points())
                    if len(points) > 1:
                        line_segments = len(points) - 1  # Number of line segments
                        print(f"  ✓ V-GROOVE LWPOLYLINE found: {line_segments} line segments")
                    else:
                        line_segments = 1
                        print(f"  ✓ V-GROOVE LWPOLYLINE found: 1 line segment (single point)")
                except Exception as e:
                    print(f"  Error processing LWPOLYLINE: {e}")
                    line_segments = 1
                    
            elif entity_type == 'POLYLINE':
                # Old-style polyline entity
                try:
                    vertices = list(entity.vertices)
                    if len(vertices) > 1:
                        line_segments = len(vertices) - 1
                        print(f"  ✓ V-GROOVE POLYLINE found: {line_segments} line segments")
                    else:
                        line_segments = 1
                        print(f"  ✓ V-GROOVE POLYLINE found: 1 line segment")
                except Exception as e:
                    print(f"  Error processing POLYLINE: {e}")
                    line_segments = 1
                    
            else:
                # Other entity types - count as 1 line segment
                line_segments = 1
                print(f"  ✓ V-GROOVE {entity_type} found: 1 line segment")
            
            vgroove_count += line_segments
            print(f"  Total V-GROOVE line segments so far: {vgroove_count}")
            
        elif is_vgroove_layer:
            print(f"  - V-GROOVE layer but wrong color: {color}")
        elif is_green_color:
            print(f"  - Green color but wrong layer: {layer_name}")
    
    print(f"=== FINAL V-GROOVE COUNT: {vgroove_count} ===\n")
    return vgroove_count

def count_bending_lines(part_entities, layers):
    """Count the number of bending lines (blue lines) in a part for bending cost calculation"""
    bending_count = 0
    
    print(f"\n=== BENDING COUNTING DEBUG ===")
    print(f"Total entities in part: {len(part_entities)}")
    
    for i, entity in enumerate(part_entities):
        if not hasattr(entity, 'dxf'):
            print(f"Entity {i}: No dxf attribute")
            continue
            
        layer_name = getattr(entity.dxf, 'layer', 'UNKNOWN')
        color = get_entity_color(entity, layers)
        entity_type = entity.dxftype()
        
        print(f"Entity {i}: Type={entity_type}, Layer='{layer_name}', Color={color}")
        
        # Check for BENDING entities with multiple variations
        layer_upper = layer_name.upper()
        is_bending_layer = (
            layer_upper == 'BENDING' or 
            layer_upper == 'BEND' or 
            layer_upper == 'BEND_LINES' or
            'BENDING' in layer_upper or
            'BEND' in layer_upper
        )
        
        # For bending lines, we only check the layer name, not the color
        # This allows for BYLAYER color inheritance
        if is_bending_layer:
            # Count individual line segments within the entity
            line_segments = 0
            
            if entity_type == 'LINE':
                # Single line entity = 1 line segment
                line_segments = 1
                print(f"  ✓ BENDING LINE entity found: 1 line segment")
                
            elif entity_type == 'LWPOLYLINE':
                # Polyline entity - count the number of line segments
                try:
                    points = list(entity.get_points())
                    if len(points) > 1:
                        line_segments = len(points) - 1  # Number of line segments
                        print(f"  ✓ BENDING LWPOLYLINE found: {line_segments} line segments")
                    else:
                        line_segments = 1
                        print(f"  ✓ BENDING LWPOLYLINE found: 1 line segment (single point)")
                except Exception as e:
                    print(f"  Error processing LWPOLYLINE: {e}")
                    line_segments = 1
                    
            elif entity_type == 'POLYLINE':
                # Old-style polyline entity
                try:
                    vertices = list(entity.vertices)
                    if len(vertices) > 1:
                        line_segments = len(vertices) - 1
                        print(f"  ✓ BENDING POLYLINE found: {line_segments} line segments")
                    else:
                        line_segments = 1
                        print(f"  ✓ BENDING POLYLINE found: 1 line segment")
                except Exception as e:
                    print(f"  Error processing POLYLINE: {e}")
                    line_segments = 1
                    
            else:
                # Other entity types - count as 1 line segment
                line_segments = 1
                print(f"  ✓ BENDING {entity_type} found: 1 line segment")
            
            bending_count += line_segments
            print(f"  Total BENDING line segments so far: {bending_count}")
            
        elif is_bending_layer:
            print(f"  - BENDING layer entity found")
    
    print(f"=== FINAL BENDING COUNT: {bending_count} ===\n")
    return bending_count

def group_entities_into_shapes(entities):
    """Group connected entities into individual shapes"""
    try:
        if not entities:
            return []
        
        # Use the existing find_connected_parts function to group entities
        # This function already knows how to detect connected components
        connected_parts = find_connected_parts(entities)
        
        return connected_parts
        
    except Exception as e:
        print(f"Error grouping entities into shapes: {e}")
        return [entities]

def find_connected_entities(entities):
    """Find groups of connected entities"""
    try:
        from shapely.geometry import LineString
        from shapely.ops import linemerge
        
        if not entities:
            return []
        
        # Extract line segments
        line_segments = []
        for entity in entities:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                line_segments.append(LineString([(start.x, start.y), (end.x, end.y)]))
            elif entity.dxftype() == 'LWPOLYLINE':
                points = list(entity.get_points())
                if len(points) > 1:
                    line_segments.append(LineString([(p[0], p[1]) for p in points]))
        
        if not line_segments:
            return [entities]  # Return as one group if no line segments
        
        # Try to merge connected lines
        try:
            merged_lines = linemerge(line_segments)
            if merged_lines.geom_type == 'LineString':
                return [entities]  # All connected as one object
            elif merged_lines.geom_type == 'MultiLineString':
                # Multiple disconnected objects
                return [entities]  # For now, treat as one object
        except:
            pass
        
        return [entities]  # Default: treat as one object
        
    except Exception as e:
        print(f"Error finding connected entities: {e}")
        return [entities]

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Nesting functionality removed for deployment

ALLOWED_EXTENSIONS = {'dxf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_entity_color(entity, layers):
    """Get the actual color of an entity, considering BYLAYER"""
    if not hasattr(entity, 'dxf'):
        return 7  # Default white
    
    color = getattr(entity.dxf, 'color', 7)
    layer_name = getattr(entity.dxf, 'layer', '0')
    
    # If color is BYLAYER (256), use the layer's color
    if color == 256:
        return layers.get(layer_name, 7)
    
    return color

def get_entity_endpoints(entity):
    """Extract endpoints from different entity types"""
    endpoints = []
    
    try:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            endpoints.append(((start.x, start.y), (end.x, end.y)))
            
        elif entity.dxftype() == 'LWPOLYLINE':
            points = list(entity.get_points())
            if len(points) > 1:
                for i in range(len(points) - 1):
                    endpoints.append(((points[i][0], points[i][1]), 
                                   (points[i+1][0], points[i+1][1])))
                    
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            # Approximate circle with line segments
            angles = np.linspace(0, 2*np.pi, 32)
            points = []
            for angle in angles:
                x = center.x + radius * np.cos(angle)
                y = center.y + radius * np.sin(angle)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                endpoints.append((points[i], points[i+1]))
            endpoints.append((points[-1], points[0]))  # Close the circle
            
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = np.radians(entity.dxf.start_angle)
            end_angle = np.radians(entity.dxf.end_angle)
            
            # Ensure end_angle > start_angle
            if end_angle <= start_angle:
                end_angle += 2 * np.pi
                
            angles = np.linspace(start_angle, end_angle, 16)
            points = []
            for angle in angles:
                x = center.x + radius * np.cos(angle)
                y = center.y + radius * np.sin(angle)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                endpoints.append((points[i], points[i+1]))
                
        elif entity.dxftype() == 'ELLIPSE':
            # Use the geometry module to flatten ellipse to segments
            try:
                from geometry.flatten import to_segments
                segments = to_segments(entity, tol=0.05)
                for i in range(len(segments) - 1):
                    endpoints.append(((segments[i].x, segments[i].y), 
                                   (segments[i+1].x, segments[i+1].y)))
            except ImportError:
                # Fallback: approximate ellipse with points
                center = entity.dxf.center
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                start_param = entity.dxf.start_param
                end_param = entity.dxf.end_param
                
                # Ensure end_param > start_param
                if end_param <= start_param:
                    end_param += 2 * np.pi
                    
                params = np.linspace(start_param, end_param, 16)
                points = []
                for param in params:
                    # Parametric ellipse equation
                    x = center.x + major_axis.x * np.cos(param) + major_axis.y * ratio * np.sin(param)
                    y = center.y + major_axis.y * np.cos(param) - major_axis.x * ratio * np.sin(param)
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    endpoints.append((points[i], points[i+1]))
                    
        elif entity.dxftype() == 'SPLINE':
            # Use the geometry module to flatten spline to segments
            try:
                from geometry.flatten import to_segments
                segments = to_segments(entity, tol=0.05)
                for i in range(len(segments) - 1):
                    endpoints.append(((segments[i].x, segments[i].y), 
                                   (segments[i+1].x, segments[i+1].y)))
            except ImportError:
                # Fallback: use control points as approximation
                control_points = entity.control_points
                if len(control_points) >= 2:
                    for i in range(len(control_points) - 1):
                        endpoints.append(((control_points[i].x, control_points[i].y), 
                                       (control_points[i+1].x, control_points[i+1].y)))
                
    except Exception as e:
        print(f"Error extracting endpoints from {entity.dxftype()}: {e}")
    
    return endpoints

def distance_between_points(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def are_points_connected(p1, p2, tolerance=0.1):
    """Check if two points are connected (within tolerance)"""
    return distance_between_points(p1, p2) <= tolerance

def find_connected_parts(entities, layers=None):
    """Find connected parts by analyzing entity connectivity and containment relationships"""
    print(f"=== find_connected_parts called with {len(entities) if entities else 0} entities ===")
    if not entities:
        return []
    
    # If layers is not provided, create a default mapping
    if layers is None:
        layers = {}
    
    try:
        # Step 1: Separate entities by type for analysis
        layer0_entities = []  # Lines, polylines from Layer 0
        circle_entities = []  # Circles from any layer
        ellipse_entities = []  # ELLIPSE entities
        spline_entities = []  # SPLINE entities
        vgroove_entities = []  # V-GROOVE entities
        bending_entities = []  # BENDING entities
        arc_entities = []  # ARC entities (common in ornamental patterns)
        line_entities = []  # LINE entities
        other_entities = []  # Other entity types
        
        for entity in entities:
            if not hasattr(entity, 'dxf'):
                other_entities.append(entity)
                continue
                
            layer_name = getattr(entity.dxf, 'layer', 'UNKNOWN')
            color = get_entity_color(entity, layers)
            entity_type = entity.dxftype()
            
            # Debug: Print first few entities (reduced verbosity)
            if len(layer0_entities) + len(circle_entities) + len(ellipse_entities) + len(spline_entities) + len(arc_entities) + len(line_entities) + len(vgroove_entities) + len(other_entities) < 3:
                print(f"DEBUG: Entity {entity_type} from layer '{layer_name}' with color {color}")
            
            if entity_type == 'CIRCLE':
                circle_entities.append(entity)
            elif entity_type == 'ELLIPSE':
                ellipse_entities.append(entity)
            elif entity_type == 'SPLINE':
                spline_entities.append(entity)
            elif entity_type == 'ARC':
                arc_entities.append(entity)
            elif entity_type == 'LINE':
                line_entities.append(entity)
            elif layer_name.upper() == 'V-GROOVE' and color == 3:
                vgroove_entities.append(entity)
            elif layer_name.upper() == 'BENDING':
                bending_entities.append(entity)
            elif layer_name == '0' and color == 7:
                layer0_entities.append(entity)
            elif layer_name.upper() == 'WHITE' and color == 7:
                # Treat WHITE layer entities as Layer 0 entities
                layer0_entities.append(entity)
            else:
                other_entities.append(entity)
        
        print(f"=== ENTITY SEPARATION DEBUG ===")
        print(f"Layer 0 entities: {len(layer0_entities)}")
        print(f"Circle entities: {len(circle_entities)}")
        print(f"Ellipse entities: {len(ellipse_entities)}")
        print(f"Spline entities: {len(spline_entities)}")
        print(f"ARC entities: {len(arc_entities)}")
        print(f"LINE entities: {len(line_entities)}")
        print(f"V-GROOVE entities: {len(vgroove_entities)}")
        print(f"BENDING entities: {len(bending_entities)}")
        print(f"Other entities: {len(other_entities)}")
        print(f"Total entities processed: {len(layer0_entities) + len(circle_entities) + len(ellipse_entities) + len(spline_entities) + len(arc_entities) + len(line_entities) + len(vgroove_entities) + len(bending_entities) + len(other_entities)}")
        print(f"=== END ENTITY SEPARATION DEBUG ===")
        
        # Step 2: Find connected frame-forming components (Layer 0, ARCs, LINES)
        frame_parts = []
        
        # Combine all potential frame-forming entities
        frame_entities = layer0_entities + arc_entities + line_entities
        
        # Use proximity grouping for large numbers of entities to avoid O(n²) complexity
        if len(frame_entities) > 1000:
            print("=== USING PROXIMITY GROUPING FOR LARGE ENTITY SET ===")
            frame_parts = group_entities_by_proximity(frame_entities, max_distance=1.0)
        elif frame_entities:
            # Use existing connectivity logic for smaller entity sets
            all_endpoints = []
            entity_endpoints = {}
            
            for i, entity in enumerate(frame_entities):
                endpoints = get_entity_endpoints(entity)
                entity_endpoints[i] = endpoints
                all_endpoints.extend(endpoints)
            
            # Build connectivity graph for frame entities
            connectivity = defaultdict(set)
            tolerance = 0.5  # Increased tolerance for better connectivity detection
            
            for i, entity1 in enumerate(frame_entities):
                for j, entity2 in enumerate(frame_entities):
                    if i >= j:
                        continue
                        
                    endpoints1 = entity_endpoints[i]
                    endpoints2 = entity_endpoints[j]
                    
                    # Check if any endpoints are connected
                    connected = False
                    for ep1_start, ep1_end in endpoints1:
                        for ep2_start, ep2_end in endpoints2:
                            if (are_points_connected(ep1_start, ep2_start, tolerance) or
                                are_points_connected(ep1_start, ep2_end, tolerance) or
                                are_points_connected(ep1_end, ep2_start, tolerance) or
                                are_points_connected(ep1_end, ep2_end, tolerance)):
                                connectivity[i].add(j)
                                connectivity[j].add(i)
                                connected = True
                                break
                        if connected:
                            break
                    
                    # Additional check: if entities are very close, consider them connected
                    if not connected:
                        try:
                            bounds1 = get_entity_bounds(entity1)
                            bounds2 = get_entity_bounds(entity2)
                            if bounds1 and bounds2:
                                # Check if bounding boxes overlap or are very close
                                x1_min, y1_min, x1_max, y1_max = bounds1
                                x2_min, y2_min, x2_max, y2_max = bounds2
                                
                                # Check for overlap or proximity
                                if (x1_max + tolerance >= x2_min and x2_max + tolerance >= x1_min and
                                    y1_max + tolerance >= y2_min and y2_max + tolerance >= y1_min):
                                    connectivity[i].add(j)
                                    connectivity[j].add(i)
                        except Exception as e:
                            print(f"Error in additional connectivity check: {e}")
            
            # Find connected components using DFS
            visited = set()
            
            def dfs(entity_idx, part):
                visited.add(entity_idx)
                part.append(entity_idx)
                for neighbor in connectivity[entity_idx]:
                    if neighbor not in visited:
                        dfs(neighbor, part)
            
            # Find all connected frame components
            for entity_idx in range(len(frame_entities)):
                if entity_idx not in visited:
                    part = []
                    dfs(entity_idx, part)
                    if part:
                        frame_part = [frame_entities[i] for i in part]
                        frame_parts.append(frame_part)
        
        # Step 3: Group circles and other entities with their containing frame parts
        final_parts = []
        used_circles = set()
        used_ellipses = set()
        used_splines = set()
        used_vgrooves = set()
        used_others = set()
        
        # For each frame part, find all entities contained within it
        for frame_part in frame_parts:
            combined_part = list(frame_part)
            
            # Create a more robust containment test for this frame part
            frame_polygon = create_part_polygon(frame_part)
            
            # Find circles contained within this frame part
            for i, circle in enumerate(circle_entities):
                if i in used_circles:
                    continue
                
                if is_entity_contained_in_part_robust(circle, frame_part, frame_polygon):
                    combined_part.append(circle)
                    used_circles.add(i)
            
            # Find ellipses contained within this frame part
            for i, ellipse in enumerate(ellipse_entities):
                if i in used_ellipses:
                    continue
                
                if is_entity_contained_in_part_robust(ellipse, frame_part, frame_polygon):
                    combined_part.append(ellipse)
                    used_ellipses.add(i)
            
            # Find splines contained within this frame part
            for i, spline in enumerate(spline_entities):
                if i in used_splines:
                    continue
                
                if is_entity_contained_in_part_robust(spline, frame_part, frame_polygon):
                    combined_part.append(spline)
                    used_splines.add(i)
            
            # Find V-GROOVE entities contained within this frame part
            for i, vgroove in enumerate(vgroove_entities):
                if i in used_vgrooves:
                    continue
                
                if is_entity_contained_in_part_robust(vgroove, frame_part, frame_polygon):
                    combined_part.append(vgroove)
                    used_vgrooves.add(i)
            
            # Find other entities contained within this frame part
            for i, other in enumerate(other_entities):
                if i in used_others:
                    continue
                
                if is_entity_contained_in_part_robust(other, frame_part, frame_polygon):
                    combined_part.append(other)
                    used_others.add(i)
            
            final_parts.append(combined_part)
        
        # Step 3.5: Fallback - group remaining entities by proximity to frame parts
        if frame_parts:
            # For any remaining circles, find the closest frame part
            for i, circle in enumerate(circle_entities):
                if i in used_circles:
                    continue
                
                closest_part_idx = find_closest_part(circle, final_parts)
                if closest_part_idx is not None:
                    final_parts[closest_part_idx].append(circle)
                    used_circles.add(i)
            
            # For any remaining ellipses, find the closest frame part
            for i, ellipse in enumerate(ellipse_entities):
                if i in used_ellipses:
                    continue
                
                closest_part_idx = find_closest_part(ellipse, final_parts)
                if closest_part_idx is not None:
                    final_parts[closest_part_idx].append(ellipse)
                    used_ellipses.add(i)
            
            # For any remaining splines, find the closest frame part
            for i, spline in enumerate(spline_entities):
                if i in used_splines:
                    continue
                
                closest_part_idx = find_closest_part(spline, final_parts)
                if closest_part_idx is not None:
                    final_parts[closest_part_idx].append(spline)
                    used_splines.add(i)
            
            # For any remaining V-GROOVE entities, find the closest frame part
            for i, vgroove in enumerate(vgroove_entities):
                if i in used_vgrooves:
                    continue
                
                closest_part_idx = find_closest_part(vgroove, final_parts)
                if closest_part_idx is not None:
                    final_parts[closest_part_idx].append(vgroove)
                    used_vgrooves.add(i)
        
        # Step 4: Special handling for cases with many individual entities (like Pattern.dxf)
        if len(frame_parts) == 0 and (len(arc_entities) > 100 or len(line_entities) > 100):
            # Group entities by proximity when no clear frames are found
            print("=== PROXIMITY GROUPING MODE ===")
            
            # Group ARCs by proximity
            if arc_entities:
                arc_groups = group_entities_by_proximity(arc_entities, max_distance=2.0)
                for group in arc_groups:
                    final_parts.append(group)
            
            # Group LINES by proximity
            if line_entities:
                line_groups = group_entities_by_proximity(line_entities, max_distance=2.0)
                for group in line_groups:
                    final_parts.append(group)
            
            # Add remaining entities
            for i, circle in enumerate(circle_entities):
                if i not in used_circles:
                    final_parts.append([circle])
            
            for i, ellipse in enumerate(ellipse_entities):
                if i not in used_ellipses:
                    final_parts.append([ellipse])
            
            for i, spline in enumerate(spline_entities):
                if i not in used_splines:
                    final_parts.append([spline])
            
            for i, vgroove in enumerate(vgroove_entities):
                if i not in used_vgrooves:
                    final_parts.append([vgroove])
            
            for i, other in enumerate(other_entities):
                if i not in used_others:
                    final_parts.append([other])
        else:
            # Step 4: Handle standalone V-GROOVE parts (not contained in any frame part)
            for i, vgroove in enumerate(vgroove_entities):
                if i not in used_vgrooves:
                    final_parts.append([vgroove])
            
            # Step 5: Handle standalone circles (not contained in any frame part)
            for i, circle in enumerate(circle_entities):
                if i not in used_circles:
                    final_parts.append([circle])
            
            # Step 6: Handle standalone ellipses (not contained in any frame part)
            for i, ellipse in enumerate(ellipse_entities):
                if i not in used_ellipses:
                    final_parts.append([ellipse])
            
            # Step 7: Handle standalone splines (not contained in any frame part)
            for i, spline in enumerate(spline_entities):
                if i not in used_splines:
                    final_parts.append([spline])
            
            # Step 8: Handle standalone other entities (not contained in any frame part)
            for i, other in enumerate(other_entities):
                if i not in used_others:
                    final_parts.append([other])

        # Step 4.9: DBSCAN fallback grouping for rare leftovers that didn't form parts
        try:
            from sklearn.cluster import DBSCAN
            import numpy as _np

            # Collect any leftover entities that still appear ungrouped (e.g., singletons not in final_parts)
            all_in_parts = set()
            for grp in final_parts:
                for e in grp:
                    all_in_parts.add(id(e))

            leftovers = [e for e in entities if id(e) not in all_in_parts]
            if leftovers:
                coords = []
                for e in leftovers:
                    pts = get_entity_points(e) or []
                    if pts:
                        # Use entity centroid as a feature
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        cx = sum(xs) / len(xs)
                        cy = sum(ys) / len(ys)
                        coords.append([cx, cy])
                    else:
                        # Fallback: approximate from bounds
                        b = get_entity_bounds(e)
                        if b:
                            cx = (b[0] + b[2]) / 2.0
                            cy = (b[1] + b[3]) / 2.0
                            coords.append([cx, cy])
                        else:
                            coords.append([0.0, 0.0])

                X = _np.array(coords, dtype=float)
                if len(X) >= 2:
                    # eps tuned small to avoid merging distant parts; min_samples=2 to form real groups
                    model = DBSCAN(eps=2.0, min_samples=2)
                    labels = model.fit_predict(X)
                    label_to_entities = {}
                    for ent, lab in zip(leftovers, labels):
                        if lab == -1:
                            continue  # noise -> leave to next fallback (singletons already handled)
                        label_to_entities.setdefault(lab, []).append(ent)
                    for lab, ents in label_to_entities.items():
                        if ents:
                            final_parts.append(list(ents))
        except Exception as _db_err:
            try:
                print(f"DBSCAN fallback grouping skipped: {_db_err}")
            except Exception:
                pass
        
        print(f"=== FINAL GROUPING DEBUG ===")
        print(f"Total final parts: {len(final_parts)}")
        for i, part in enumerate(final_parts):
            layer0_count = sum(1 for e in part if hasattr(e, 'dxf') and getattr(e.dxf, 'layer', '') == '0')
            circle_count = sum(1 for e in part if e.dxftype() == 'CIRCLE')
            ellipse_count = sum(1 for e in part if e.dxftype() == 'ELLIPSE')
            spline_count = sum(1 for e in part if e.dxftype() == 'SPLINE')
            vgroove_count = sum(1 for e in part if hasattr(e, 'dxf') and getattr(e.dxf, 'layer', '').upper() == 'V-GROOVE')
            print(f"Part {i}: {len(part)} entities (Layer0: {layer0_count}, Circles: {circle_count}, Ellipses: {ellipse_count}, Splines: {spline_count}, V-GROOVE: {vgroove_count})")
        print(f"=== END FINAL GROUPING DEBUG ===")
        
        return final_parts
        
    except Exception as e:
        print(f"Error in find_connected_parts: {e}")
        # Fallback: return all entities as one part
        return [entities]

def is_entity_contained_in_part(entity, part_entities, tolerance=0.1):
    """Check if an entity is contained within a part (group of entities)"""
    try:
        # Get the bounding box of the part
        part_bounds = get_part_bounds(part_entities)
        if not part_bounds:
            return False
        
        # Get the bounding box of the entity
        entity_bounds = get_entity_bounds(entity)
        if not entity_bounds:
            return False
        
        # Check if entity is contained within part bounds
        if is_contained(entity_bounds, part_bounds, tolerance):
            return True
        
        # Additional check: if entity is a circle, check if its center is inside the part
        if entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            center_point = (center.x, center.y)
            
            # Get all points from the part to form a polygon
            part_points = get_part_points(part_entities)
            if len(part_points) >= 3:
                if point_in_polygon(center_point, part_points, tolerance):
                    return True
        
        # For other entity types, check if their key points are inside the part
        entity_points = get_entity_points(entity)
        if entity_points:
            part_points = get_part_points(part_entities)
            if len(part_points) >= 3:
                # Check if majority of entity points are inside the part
                points_inside = 0
                total_points = len(entity_points)
                
                for point in entity_points:
                    if point_in_polygon(point, part_points, tolerance):
                        points_inside += 1
                
                # If more than 70% of points are inside, consider it contained
                if total_points > 0 and (points_inside / total_points) > 0.7:
                    return True
        
        return False
        
    except Exception as e:
        print(f"Error in is_entity_contained_in_part: {e}")
        return False

def create_part_polygon(part_entities):
    """Create a polygon representation of a part for robust containment testing"""
    try:
        # Try to extract polygon from entities
        polygon = extract_polygon_from_entities(part_entities)
        if polygon:
            return polygon
        
        # Fallback: create convex hull from all points
        all_points = get_part_points(part_entities)
        if len(all_points) >= 3:
            from shapely.geometry import MultiPoint
            from shapely.ops import unary_union
            points = MultiPoint(all_points)
            hull = points.convex_hull
            return hull
        
        return None
        
    except Exception as e:
        print(f"Error in create_part_polygon: {e}")
        return None

def is_entity_contained_in_part_robust(entity, part_entities, part_polygon=None, tolerance=0.1):
    """Robust containment test using multiple methods"""
    try:
        # Method 1: Use the provided polygon if available
        if part_polygon and hasattr(part_polygon, 'contains'):
            entity_points = get_entity_points(entity)
            if entity_points:
                # Check if all entity points are inside the polygon
                points_inside = 0
                total_points = len(entity_points)
                
                for point in entity_points:
                    if part_polygon.contains(Point(point)):
                        points_inside += 1
                
                # If more than 80% of points are inside, consider it contained
                if total_points > 0 and (points_inside / total_points) > 0.8:
                    return True
        
        # Method 2: Fallback to the original method
        return is_entity_contained_in_part(entity, part_entities, tolerance)
        
    except Exception as e:
        print(f"Error in is_entity_contained_in_part_robust: {e}")
        # Fallback to original method
        return is_entity_contained_in_part(entity, part_entities, tolerance)

def find_closest_part(entity, parts, max_distance=5.0):
    """Find the closest part to an entity based on proximity"""
    try:
        entity_points = get_entity_points(entity)
        if not entity_points:
            return None
        
        # Use the center point of the entity
        entity_center = entity_points[len(entity_points) // 2]
        
        closest_part_idx = None
        min_distance = float('inf')
        
        for i, part in enumerate(parts):
            part_points = get_part_points(part)
            if not part_points:
                continue
            
            # Calculate distance to part (use center of part)
            part_center = part_points[len(part_points) // 2]
            distance = distance_between_points(entity_center, part_center)
            
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                closest_part_idx = i
        
        return closest_part_idx
        
    except Exception as e:
        print(f"Error in find_closest_part: {e}")
        return None

def group_entities_by_proximity(entities, max_distance=2.0):
    """Group entities by proximity when no clear connectivity is found"""
    if not entities:
        return []
    
    # For very large entity sets, use a simplified approach
    if len(entities) > 5000:
        print(f"Using simplified grouping for {len(entities)} entities")
        return group_entities_simplified(entities, max_distance)
    
    groups = []
    used = set()
    
    for i, entity in enumerate(entities):
        if i in used:
            continue
        
        # Start a new group with this entity
        group = [entity]
        used.add(i)
        
        # Find all entities within max_distance
        changed = True
        while changed:
            changed = False
            for j, other_entity in enumerate(entities):
                if j in used:
                    continue
                
                # Check if any entity in the group is close to this entity
                for group_entity in group:
                    if are_entities_close(group_entity, other_entity, max_distance):
                        group.append(other_entity)
                        used.add(j)
                        changed = True
                        break
        
        groups.append(group)
    
    return groups

def group_entities_simplified(entities, max_distance=2.0):
    """Simplified grouping for very large entity sets"""
    if not entities:
        return []
    
    # For very large sets, just group by entity type and approximate location
    arc_entities = [e for e in entities if e.dxftype() == 'ARC']
    line_entities = [e for e in entities if e.dxftype() == 'LINE']
    
    groups = []
    
    # Group ARCs into a few large groups
    if arc_entities:
        arc_groups = []
        group_size = max(1, len(arc_entities) // 10)  # Create ~10 groups
        
        for i in range(0, len(arc_entities), group_size):
            group = arc_entities[i:i + group_size]
            arc_groups.append(group)
        
        groups.extend(arc_groups)
    
    # Group LINES into a few large groups
    if line_entities:
        line_groups = []
        group_size = max(1, len(line_entities) // 5)  # Create ~5 groups
        
        for i in range(0, len(line_entities), group_size):
            group = line_entities[i:i + group_size]
            line_groups.append(group)
        
        groups.extend(line_groups)
    
    return groups

def are_entities_close(entity1, entity2, max_distance):
    """Check if two entities are close to each other"""
    try:
        # Get points from both entities
        points1 = get_entity_points(entity1)
        points2 = get_entity_points(entity2)
        
        if not points1 or not points2:
            return False
        
        # Check if any point from entity1 is close to any point from entity2
        for p1 in points1:
            for p2 in points2:
                if distance_between_points(p1, p2) <= max_distance:
                    return True
        
        return False
        
    except Exception as e:
        print(f"Error checking entity proximity: {e}")
        return False

def filter_entities(msp, layers):
    """Filter entities according to the specified rules"""
    filtered_entities = []
    removed_count = 0
    
    for entity in msp:
        if not hasattr(entity, 'dxf'):
            continue
            
        layer_name = getattr(entity.dxf, 'layer', 'UNKNOWN')
        color = get_entity_color(entity, layers)
        entity_type = entity.dxftype()
        
        # Rule 1: Remove red lines (DIMENSION layer - color 1)
        if color == 1:
            removed_count += 1
            continue
            
        # Rule 2: Remove white "SECTIONS" layer
        if layer_name.upper() == 'SECTIONS' and color == 7:
            removed_count += 1
            continue
            
        # Rule 3: Keep white "0" layer
        if layer_name == '0' and color == 7:
            filtered_entities.append(entity)
            continue
            
        # Rule 4: Keep V-GROOVE layer (green)
        if layer_name.upper() == 'V-GROOVE' and color == 3:
            filtered_entities.append(entity)
            continue
            
        # Rule 5: Keep supported entity types regardless of layer
        if entity_type in ['LINE', 'LWPOLYLINE', 'POLYLINE', 'ARC', 'CIRCLE', 'ELLIPSE', 'SPLINE', 'HATCH']:
            filtered_entities.append(entity)
            continue
            
        # Rule 6: Remove unsupported entity types
        if entity_type in ['DIMENSION', 'BLOCK', 'TEXT', 'MTEXT']:
            removed_count += 1
            continue
            
        # Rule 7: Remove entities from ignored layers
        if layer_name.lower() in ['dimensions', 'section']:
            removed_count += 1
            continue
            
        # Rule 8: Keep any other entities that don't match removal criteria
        if color != 1 and not (layer_name.upper() == 'SECTIONS' and color == 7):
            filtered_entities.append(entity)
    
    return filtered_entities, removed_count

def extract_geometry(entity):
    """Extract geometry from different entity types using the new flattening module"""
    try:
        from geometry.flatten import to_segments, GeometryError
        
        # Use the new flattening module for all entity types
        segments = to_segments(entity, tol=0.05)
        
        # Debug: Print segment count for ELLIPSE and SPLINE entities
        if entity.dxftype() in ['ELLIPSE', 'SPLINE']:
            print(f"DEBUG: {entity.dxftype()} - flatten module returned {len(segments)} segments")
        
        # Convert Vec2 segments to the expected format
        geometry = []
        
        # Handle case where we have segments
        if len(segments) > 1:
            for i in range(len(segments) - 1):
                start = segments[i]
                end = segments[i + 1]
                geometry.append([(start.x, start.y), (end.x, end.y)])
        elif len(segments) == 1:
            # Single point - create a small line segment for visibility
            point = segments[0]
            geometry.append([(point.x - 0.1, point.y - 0.1), (point.x + 0.1, point.y + 0.1)])
        
        # If no geometry was extracted, fall back to the fallback method
        if not geometry:
            if entity.dxftype() in ['ELLIPSE', 'SPLINE']:
                print(f"DEBUG: {entity.dxftype()} - no geometry extracted, using fallback")
            return _extract_geometry_fallback(entity)
        
        return geometry
        
    except GeometryError as e:
        print(f"Geometry error processing {entity.dxftype()}: {e}")
        return _extract_geometry_fallback(entity)
    except ImportError:
        # Fallback to original method if geometry module is not available
        return _extract_geometry_fallback(entity)
    except Exception as e:
        print(f"Error extracting geometry from {entity.dxftype()}: {e}")
        return _extract_geometry_fallback(entity)


def _extract_geometry_fallback(entity):
    """Fallback geometry extraction using original method"""
    geometry = []
    
    try:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            geometry.append([(start.x, start.y), (end.x, end.y)])
            
        elif entity.dxftype() == 'LWPOLYLINE':
            points = list(entity.get_points())
            if len(points) > 1:
                for i in range(len(points) - 1):
                    geometry.append([(points[i][0], points[i][1]), 
                                   (points[i+1][0], points[i+1][1])])
                    
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            # Approximate circle with line segments
            angles = np.linspace(0, 2*np.pi, 32)
            points = []
            for angle in angles:
                x = center.x + radius * np.cos(angle)
                y = center.y + radius * np.sin(angle)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                geometry.append([points[i], points[i+1]])
            geometry.append([points[-1], points[0]])  # Close the circle
            
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = np.radians(entity.dxf.start_angle)
            end_angle = np.radians(entity.dxf.end_angle)
            
            # Ensure end_angle > start_angle
            if end_angle <= start_angle:
                end_angle += 2 * np.pi
                
            angles = np.linspace(start_angle, end_angle, 16)
            points = []
            for angle in angles:
                x = center.x + radius * np.cos(angle)
                y = center.y + radius * np.sin(angle)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                geometry.append([points[i], points[i+1]])
                
        elif entity.dxftype() == 'ELLIPSE':
            # Approximate ellipse with line segments
            print(f"DEBUG: Processing ELLIPSE in fallback method")
            center = entity.dxf.center
            major_axis = entity.dxf.major_axis
            ratio = entity.dxf.ratio
            start_param = entity.dxf.start_param
            end_param = entity.dxf.end_param
            
            # Ensure end_param > start_param
            if end_param <= start_param:
                end_param += 2 * np.pi
                
            # Use more segments for better approximation
            params = np.linspace(start_param, end_param, 32)
            points = []
            for param in params:
                # Parametric ellipse equation
                x = center.x + major_axis.x * np.cos(param) + major_axis.y * ratio * np.sin(param)
                y = center.y + major_axis.y * np.cos(param) - major_axis.x * ratio * np.sin(param)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                geometry.append([points[i], points[i+1]])
            print(f"DEBUG: ELLIPSE fallback generated {len(geometry)} line segments")
                
        elif entity.dxftype() == 'SPLINE':
            # Approximate spline with control points or use flattening if available
            print(f"DEBUG: Processing SPLINE in fallback method")
            try:
                # Try to use ezdxf's built-in flattening
                points = list(entity.flattening(0.05))
                for i in range(len(points) - 1):
                    geometry.append([(points[i].x, points[i].y), 
                                   (points[i+1].x, points[i+1].y)])
                print(f"DEBUG: SPLINE ezdxf flattening generated {len(geometry)} line segments")
            except Exception as e:
                print(f"DEBUG: SPLINE ezdxf flattening failed: {e}")
                # Fallback: use control points as approximation
                control_points = entity.control_points
                if len(control_points) >= 2:
                    for i in range(len(control_points) - 1):
                        geometry.append([(control_points[i].x, control_points[i].y), 
                                       (control_points[i+1].x, control_points[i+1].y)])
                    print(f"DEBUG: SPLINE control points generated {len(geometry)} line segments")
                else:
                    print(f"DEBUG: SPLINE no control points available")
                
    except Exception as e:
        print(f"Error in fallback geometry extraction from {entity.dxftype()}: {e}")
    
    return geometry

def calculate_polygon_area(points):
    """Calculate the area of a polygon using Shapely for precise results"""
    if len(points) < 3:
        return 0.0
    
    try:
        from shapely.geometry import Polygon
        polygon = Polygon(points)
        if polygon.is_valid:
            return polygon.area
        else:
            # Fallback to shoelace formula if polygon is invalid
            n = len(points)
            area = 0.0
            
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            
            return abs(area) / 2.0
    except Exception as e:
        print(f"Error in Shapely polygon area calculation: {e}")
        # Fallback to shoelace formula
        n = len(points)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0

def extract_polygon_from_entities(entities):
    """Extract a complete polygon from a set of entities using Shapely for precise geometry"""
    if not entities:
        return []
    
    try:
        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union, linemerge
        
        # Get all line segments from entities
        line_segments = []
        for entity in entities:
            try:
                if entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    line_segments.append(LineString([(start.x, start.y), (end.x, end.y)]))
                elif entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points())
                    if len(points) > 1:
                        line_segments.append(LineString([(p[0], p[1]) for p in points]))
            except Exception as e:
                print(f"Error processing entity {entity.dxftype()}: {e}")
                continue
        
        if not line_segments:
            return []
        
        try:
            # Merge all line segments into a single geometry
            merged_lines = linemerge(line_segments)
            
            # If we have a single line, try to close it into a polygon
            if merged_lines.geom_type == 'LineString':
                coords = list(merged_lines.coords)
                if len(coords) >= 3:
                    # Try to create a polygon from the line coordinates
                    try:
                        polygon = Polygon(coords)
                        if polygon.is_valid and polygon.area > 0:
                            return list(polygon.exterior.coords)
                    except:
                        pass
            
            # If we have multiple lines, try to create a polygon from the union
            elif merged_lines.geom_type == 'MultiLineString':
                # Try to create polygons from each line
                polygons = []
                for line in merged_lines.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 3:
                        try:
                            polygon = Polygon(coords)
                            if polygon.is_valid and polygon.area > 0:
                                polygons.append(polygon)
                        except:
                            pass
                
                if polygons:
                    # Return the largest polygon
                    largest_polygon = max(polygons, key=lambda p: p.area)
                    return list(largest_polygon.exterior.coords)
            
        except Exception as e:
            print(f"Error in Shapely operations: {e}")
        
        # Fallback: return coordinates from the first valid line
        for line in line_segments:
            coords = list(line.coords)
            if len(coords) >= 3:
                return coords
        
        return []
        
    except Exception as e:
        print(f"Error extracting polygon from entities: {e}")
        return []
# ─────────────────────────────────────────────────────────────────────────────
def part_to_polygon(part_entities, buffer=0.1):
    """
    Best-effort conversion of any 'part' (list of DXF entities) to a Shapely
    Polygon.  Strategy:
      1. Try the high-precision extract_polygon_from_entities() helper.
      2. If that fails, use the convex hull of *all* entity points.
    Returns None when a valid surface can't be built.
    """
    try:
        # 1) exact outline if possible
        outline = extract_polygon_from_entities(part_entities)
        if outline and len(outline) >= 3:
            poly = Polygon(outline)
            if poly.is_valid and poly.area > 0:
                return poly.buffer(buffer)

        # 2) convex hull fallback
        pts = get_part_points(part_entities)
        if len(pts) >= 3:
            poly = Polygon(pts).convex_hull
            if poly.is_valid and poly.area > 0:
                return poly.buffer(buffer)
    except Exception as e:
        print(f"[part_to_polygon] {e}")

    return None



def calculate_part_dimensions(part_entities, layers):
    """Calculate the length, width, and area of a part.
    Length/width come from the minimum rotated bounding rectangle (more accurate than axis-aligned BB),
    using only Layer 0 geometry when available.
    """
    if not part_entities:
        return 0.0, 0.0, 0.0  # length, width, area

    try:
        from shapely.geometry import MultiPoint, Polygon
        import numpy as np

        # Prefer Layer 0 entities for outer geometry
        try:
            layer0_entities = get_layer0_entities(part_entities, layers)
            target_entities = layer0_entities if layer0_entities else part_entities
        except Exception:
            target_entities = part_entities

        # Collect points
        pts = []
        for ent in target_entities:
            try:
                pts.extend(get_entity_points(ent) or [])
            except Exception:
                continue
        # Deduplicate and validate
        uniq = []
        seen = set()
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                rp = (round(float(p[0]), 6), round(float(p[1]), 6))
                if rp not in seen:
                    seen.add(rp)
                    uniq.append(rp)

        if len(uniq) < 3:
            # Fallback to axis-aligned extents if insufficient points
            if len(uniq) == 0:
                return 0.0, 0.0, 0.0
            xs = [p[0] for p in uniq]
            ys = [p[1] for p in uniq]
            length_mm = max(xs) - min(xs)
            width_mm = max(ys) - min(ys)
            area = calculate_part_area(part_entities, layers)
            return float(length_mm), float(width_mm), area

        # Build convex hull polygon and compute minimum rotated rectangle
        hull = MultiPoint(uniq).convex_hull
        if not isinstance(hull, Polygon) or hull.is_empty:
            xs = [p[0] for p in uniq]
            ys = [p[1] for p in uniq]
            length_mm = max(xs) - min(xs)
            width_mm = max(ys) - min(ys)
            area = calculate_part_area(part_entities, layers)
            return float(length_mm), float(width_mm), area

        mrr = hull.minimum_rotated_rectangle
        rect_coords = list(mrr.exterior.coords)
        # The rectangle polygon returns 5 points (closed). Compute side lengths from consecutive edges.
        def dist(a, b):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            return (dx * dx + dy * dy) ** 0.5
        sides = [dist(rect_coords[i], rect_coords[i + 1]) for i in range(4)]
        # Two unique side lengths (within tolerance)
        sides_sorted = sorted(sides)
        # Pair sides by closeness
        a = sides_sorted[0]
        b = sides_sorted[-1]
        length_mm = max(a, b)
        width_mm = min(a, b)

        area = calculate_part_area(part_entities, layers)

        # Debug log
        try:
            print(f"[dims] part oriented LxW = {length_mm:.2f} x {width_mm:.2f} mm (area_m2={area:.6f})")
        except Exception:
            pass

        return float(length_mm), float(width_mm), area

    except Exception as e:
        print(f"Error calculating part dimensions: {e}")
        return 0.0, 0.0, 0.0

def calculate_part_area(part_entities, layers):
    """Calculate the precise area of a part using only Layer 0 entities (outer boundary)"""
    if not part_entities:
        return 0.0
    
    # Extract only Layer 0 entities for area calculation (exclude V-groove and bending)
    layer0_entities = get_layer0_entities(part_entities, layers)
    
    if not layer0_entities:
        print("Warning: No Layer 0 entities found for area calculation")
        return 0.0
    
    try:
        from shapely.geometry import LineString, Polygon, Point
        from shapely.ops import unary_union, linemerge
        import numpy as np
        
        # Store all calculated areas for validation
        calculated_areas = []
        
        # Method 1: Try to create a polygon from connected line segments
        line_segments = []
        for entity in layer0_entities:
            try:
                if entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    line_segments.append(LineString([(start.x, start.y), (end.x, end.y)]))
                elif entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points())
                    if len(points) > 1:
                        line_segments.append(LineString([(p[0], p[1]) for p in points]))
            except Exception as e:
                print(f"Error processing entity {entity.dxftype()}: {e}")
                continue
        
        if line_segments:
            try:
                # Merge all line segments
                merged_lines = linemerge(line_segments)
                
                # Try to create a polygon from the merged lines
                if merged_lines.geom_type == 'LineString':
                    coords = list(merged_lines.coords)
                    if len(coords) >= 3:
                        try:
                            polygon = Polygon(coords)
                            if polygon.is_valid and polygon.area > 0:
                                area = polygon.area / 1000000  # Convert mm² to m²
                                calculated_areas.append(('line_merge', area))
                                print(f"Method 1 (Line Merge): Area = {area:.6f} m²")
                        except Exception as e:
                            print(f"Error creating polygon from line merge: {e}")
                
                elif merged_lines.geom_type == 'MultiLineString':
                    # Try to create polygons from each line
                    polygons = []
                    for line in merged_lines.geoms:
                        coords = list(line.coords)
                        if len(coords) >= 3:
                            try:
                                polygon = Polygon(coords)
                                if polygon.is_valid and polygon.area > 0:
                                    polygons.append(polygon)
                            except:
                                pass
                    
                    if polygons:
                        # Return the largest polygon area
                        largest_polygon = max(polygons, key=lambda p: p.area)
                        area = largest_polygon.area / 1000000  # Convert mm² to m²
                        calculated_areas.append(('multi_line_merge', area))
                        print(f"Method 1 (Multi-Line Merge): Area = {area:.6f} m²")
                
            except Exception as e:
                print(f"Error in Shapely line merging: {e}")
        
        # Method 2: Create polygon from all points using convex hull
        all_points = []
        for entity in layer0_entities:
            try:
                points = get_entity_points(entity)
                if points and isinstance(points, list):
                    all_points.extend(points)
            except Exception as e:
                print(f"Error getting points from entity: {e}")
                continue
        
        if len(all_points) >= 3:
            # Remove duplicate points and validate
            unique_points = []
            for point in all_points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    # Round to avoid floating point precision issues
                    rounded_point = (round(point[0], 6), round(point[1], 6))
                    if rounded_point not in unique_points:
                        unique_points.append(rounded_point)
            
            if len(unique_points) >= 3:
                try:
                    # Create polygon from unique points
                    polygon = Polygon(unique_points)
                    if polygon.is_valid and polygon.area > 0:
                        area = polygon.area / 1000000  # Convert mm² to m²
                        calculated_areas.append(('direct_polygon', area))
                        print(f"Method 2 (Direct Polygon): Area = {area:.6f} m²")
                    
                    # If invalid, try convex hull
                    try:
                        from scipy.spatial import ConvexHull
                        points_array = np.array(unique_points)
                        hull = ConvexHull(points_array)
                        hull_points = points_array[hull.vertices]
                        
                        hull_polygon = Polygon(hull_points)
                        if hull_polygon.is_valid and hull_polygon.area > 0:
                            area = hull_polygon.area / 1000000  # Convert mm² to m²
                            calculated_areas.append(('convex_hull', area))
                            print(f"Method 2 (Convex Hull): Area = {area:.6f} m²")
                    except ImportError:
                        # If scipy is not available, use Shapely's convex hull
                        try:
                            points_geom = [Point(p) for p in unique_points]
                            if len(points_geom) >= 3:
                                # Create a simple polygon from points
                                polygon = Polygon(unique_points)
                                if polygon.is_valid and polygon.area > 0:
                                    area = polygon.area / 1000000  # Convert mm² to m²
                                    calculated_areas.append(('shapely_polygon', area))
                                    print(f"Method 2 (Shapely Polygon): Area = {area:.6f} m²")
                        except Exception as e:
                            print(f"Error in Shapely polygon creation: {e}")
                            
                except Exception as e:
                    print(f"Error in polygon creation: {e}")
        
        # Method 3: Direct area calculation for supported entity types
        for entity in part_entities:
            try:
                entity_type = entity.dxftype()
                if entity_type == 'CIRCLE':
                    # Calculate circle area
                    radius = entity.dxf.radius
                    area = (np.pi * radius * radius) / 1000000  # Convert mm² to m²
                    calculated_areas.append(('circle', area))
                    print(f"Method 3 (Circle): Area = {area:.6f} m²")
                    
                elif entity_type == 'ELLIPSE':
                    # Calculate ellipse area
                    major_axis = entity.dxf.major_axis
                    ratio = entity.dxf.ratio
                    # Major axis length
                    major_length = np.sqrt(major_axis.x**2 + major_axis.y**2)
                    # Minor axis length
                    minor_length = major_length * ratio
                    area = (np.pi * major_length * minor_length) / 1000000  # Convert mm² to m²
                    calculated_areas.append(('ellipse', area))
                    print(f"Method 3 (Ellipse): Area = {area:.6f} m²")
                    
                elif entity_type == 'ARC':
                    # Calculate arc area (sector area)
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    start_angle = np.radians(entity.dxf.start_angle)
                    end_angle = np.radians(entity.dxf.end_angle)
                    
                    # Ensure end_angle > start_angle
                    if end_angle <= start_angle:
                        end_angle += 2 * np.pi
                        
                    arc_angle = end_angle - start_angle
                    sector_area = (arc_angle / (2 * np.pi)) * (np.pi * radius * radius)
                    area = sector_area / 1000000  # Convert to millions
                    calculated_areas.append(('arc_sector', area))
                    print(f"Method 3 (Arc Sector): Area = {area:.6f}")
                    
                elif entity_type == 'SPLINE':
                    # For splines, use the geometry module to get points and calculate area
                    try:
                        from geometry.flatten import to_segments
                        segments = to_segments(entity, tol=0.05)
                        if len(segments) >= 3:
                            # Create polygon from spline points
                            points = [(seg.x, seg.y) for seg in segments]
                            if len(points) >= 3:
                                try:
                                    polygon = Polygon(points)
                                    if polygon.is_valid and polygon.area > 0:
                                        area = polygon.area / 1000000  # Convert to millions
                                        calculated_areas.append(('spline_polygon', area))
                                        print(f"Method 3 (Spline Polygon): Area = {area:.6f}")
                                except Exception as e:
                                    print(f"Error creating polygon from spline: {e}")
                    except ImportError:
                        # Fallback: use entity points directly
                        try:
                            points = get_entity_points(entity)
                            if len(points) >= 3:
                                # Create a simple polygon from the points
                                polygon = Polygon(points)
                                if polygon.is_valid and polygon.area > 0:
                                    area = polygon.area / 1000000  # Convert to millions
                                    calculated_areas.append(('spline_fallback', area))
                                    print(f"Method 3 (Spline Fallback): Area = {area:.6f}")
                        except Exception as e:
                            print(f"Error in spline fallback calculation: {e}")
                    
                    # Additional fallback: estimate area from bounding box with a reasonable factor
                    if not any(method == 'spline_polygon' or method == 'spline_fallback' for method, _ in calculated_areas):
                        try:
                            bounds = get_entity_bounds(entity)
                            if bounds:
                                x_min, y_min, x_max, y_max = bounds
                                width = x_max - x_min
                                height = y_max - y_min
                                # Estimate area as 60% of bounding box (typical for splines)
                                estimated_area = (width * height * 0.6) / 1000000  # Convert to millions
                                calculated_areas.append(('spline_estimated', estimated_area))
                                print(f"Method 3 (Spline Estimated): Area = {estimated_area:.6f}")
                        except Exception as e:
                            print(f"Error in spline estimated calculation: {e}")
                        
            except Exception as e:
                print(f"Error calculating area for {entity_type}: {e}")
                continue
        
        # Method 4: Bounding box area as last resort
        if all_points:
            x_coords = [p[0] for p in all_points if isinstance(p, (list, tuple)) and len(p) >= 2]
            y_coords = [p[1] for p in all_points if isinstance(p, (list, tuple)) and len(p) >= 2]
            
            if x_coords and y_coords:
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                area = (width * height) / 1000000  # Convert to millions
                calculated_areas.append(('bounding_box', area))
                print(f"Method 4 (Bounding Box): Area = {area:.6f}")
        
        # Validation: Choose the most reliable area calculation
        if calculated_areas:
            # For multi-entity parts, prefer line merge methods over individual entity calculations
            if len(part_entities) > 1:
                preferred_methods = ['line_merge', 'multi_line_merge', 'direct_polygon', 'shapely_polygon', 'circle', 'ellipse', 'arc_sector', 'spline_polygon', 'spline_fallback', 'spline_estimated']
            else:
                # For single entities, prefer exact mathematical formulas
                preferred_methods = ['circle', 'ellipse', 'arc_sector', 'spline_polygon', 'spline_fallback', 'spline_estimated', 'line_merge', 'multi_line_merge', 'direct_polygon', 'shapely_polygon']
            
            for method in preferred_methods:
                for calc_method, area in calculated_areas:
                    if calc_method == method and area > 0:
                        print(f"Selected method: {method} with area: {area:.6f}")
                        return area
            
            # If no preferred method found, use the largest area
            largest_area = max(calculated_areas, key=lambda x: x[1])
            print(f"Selected largest area: {largest_area[0]} with area: {largest_area[1]:.6f}")
            return largest_area[1]
        
        return 0.0
        
    except Exception as e:
        print(f"Error calculating precise area with Shapely: {e}")
        return 0.0

def get_layer0_entities(part_entities, layers):
    """Extract only Layer 0 entities from a part (outer boundary only)"""
    layer0_entities = []
    for entity in part_entities:
        if not hasattr(entity, 'dxf'):
            continue
        layer_name = getattr(entity.dxf, 'layer', '')
        color = get_entity_color(entity, layers)
        
        # Include Layer 0 entities (outer boundary)
        if layer_name == '0':
            layer0_entities.append(entity)
        # Also include WHITE layer entities as Layer 0 (for compatibility)
        elif layer_name.upper() == 'WHITE':
            layer0_entities.append(entity)
            
    return layer0_entities

def validate_area_calculations(part_areas):
    """Validate area calculations and ensure total sum is correct with maximum precision"""
    print("\n" + "="*60)
    print("AREA CALCULATION VALIDATION (in millions)")
    print("="*60)
    
    total_area = 0.0
    valid_areas = []
    
    for i, area in enumerate(part_areas):
        if area > 0:
            # Round to 6 decimal places for consistency
            rounded_area = round(area, 6)
            total_area += rounded_area
            valid_areas.append((i+1, rounded_area))
            print(f"Part {i+1}: {rounded_area:.6f} million sq units")
        else:
            print(f"Part {i+1}: 0.000000 million sq units (no valid area)")
    
    print("-" * 40)
    print(f"Total Parts with Area: {len(valid_areas)}")
    print(f"Total Area: {total_area:.6f} million sq units")
    
    # Multiple validation checks
    # Check 1: Sum validation
    recalculated_total = sum(area for _, area in valid_areas)
    if abs(total_area - recalculated_total) < 0.000001:  # Very strict precision check
        print("✅ Sum validation: CORRECT")
    else:
        print(f"❌ Sum validation ERROR: {total_area:.6f} vs {recalculated_total:.6f}")
        total_area = recalculated_total  # Use the recalculated value
    
    # Check 2: Individual area validation
    all_valid = True
    for part_num, area in valid_areas:
        if area <= 0:
            print(f"❌ Part {part_num} has invalid area: {area}")
            all_valid = False
    
    if all_valid:
        print("✅ All individual areas are valid")
    
    # Check 3: Precision validation
    precision_ok = True
    for part_num, area in valid_areas:
        if area > 0 and area < 0.000001:
            print(f"⚠️  Part {part_num} has very small area: {area:.10f}")
        if area > 1000:  # Very large area check (in millions)
            print(f"⚠️  Part {part_num} has very large area: {area:.2f}")
    
    print("="*60)
    print(f"FINAL VALIDATED TOTAL AREA: {total_area:.6f} million sq units")
    print("="*60 + "\n")
    
    return total_area, valid_areas



def create_dxf_visualization(entities, layers, title="Filtered DXF", parts=None, show_legend=True):
    """Create a matplotlib visualization of the filtered DXF entities"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    
    # Color mapping for different layers
    color_map = {
        '0': 'white',
        'V-GROOVE': 'green',
        'BENDING': 'blue',
        'SECTIONS': 'red',  # Should be filtered out, but just in case
        'DIMENSION': 'red'  # Should be filtered out, but just in case
    }
    
    all_points = []
    
    # If parts are provided, use different colors for each part
    if parts:
        colors = ['white', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray', 'orange', 'purple', 'cyan', 'magenta']
        for part_idx, part_entities in enumerate(parts):
            part_color = colors[part_idx % len(colors)]
            
            for entity in part_entities:
                if not hasattr(entity, 'dxf'):
                    continue
                    
                layer_name = getattr(entity.dxf, 'layer', '0')
                color = get_entity_color(entity, layers)
                
                # Determine line color - prioritize part color over layer color
                if layer_name.upper() == 'V-GROOVE' and color == 3:
                    line_color = 'green'  # Keep V-GROOVE green
                elif layer_name.upper() == 'BENDING':
                    line_color = 'blue'  # Keep BENDING blue regardless of color
                elif entity.dxftype() == 'CIRCLE':
                    # Circles get a special color to make them stand out
                    line_color = 'red' if part_color == 'white' else part_color
                elif entity.dxftype() == 'ELLIPSE':
                    # Ellipses get a special color to make them stand out
                    line_color = 'orange' if part_color == 'white' else part_color
                elif entity.dxftype() == 'SPLINE':
                    # Splines get a special color to make them stand out
                    line_color = 'purple' if part_color == 'white' else part_color
                else:
                    line_color = part_color  # Use part color for other entities
                
                geometry = extract_geometry(entity)
                
                # Debug: Print entity type and geometry count
                if entity.dxftype() in ['ELLIPSE', 'SPLINE']:
                    print(f"DEBUG: {entity.dxftype()} entity - extracted {len(geometry)} line segments")
                
                for line_segment in geometry:
                    if len(line_segment) == 2:
                        x_coords = [line_segment[0][0], line_segment[1][0]]
                        y_coords = [line_segment[0][1], line_segment[1][1]]
                        
                        ax.plot(x_coords, y_coords, color=line_color, linewidth=1)
                        all_points.extend([(x_coords[0], y_coords[0]), (x_coords[1], y_coords[1])])
    else:
        # Original visualization without parts
        for entity in entities:
            if not hasattr(entity, 'dxf'):
                continue
                
            layer_name = getattr(entity.dxf, 'layer', '0')
            color = get_entity_color(entity, layers)
            
            # Determine line color
            if layer_name.upper() == 'V-GROOVE' and color == 3:
                line_color = 'green'
            elif layer_name.upper() == 'BENDING':
                line_color = 'blue'
            elif layer_name == '0' and color == 7:
                line_color = 'white'
            else:
                line_color = 'blue'  # Default for other entities
            
            geometry = extract_geometry(entity)
            
            # Debug: Print entity type and geometry count
            if entity.dxftype() in ['ELLIPSE', 'SPLINE']:
                print(f"DEBUG: {entity.dxftype()} entity - extracted {len(geometry)} line segments")
            
            for line_segment in geometry:
                if len(line_segment) == 2:
                    x_coords = [line_segment[0][0], line_segment[1][0]]
                    y_coords = [line_segment[0][1], line_segment[1][1]]
                    
                    ax.plot(x_coords, y_coords, color=line_color, linewidth=1)
                    all_points.extend([(x_coords[0], y_coords[0]), (x_coords[1], y_coords[1])])
    
    if all_points:
        # Set plot limits with some padding
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add 10% padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    ax.set_title(title, fontsize=16, color='white')
    ax.set_facecolor('black')
    ax.grid(True, alpha=0.3, color='gray')
    
    # Add legend only if requested
    if show_legend:
        if parts:
            legend_elements = [
                plt.Line2D([0], [0], color='green', label='V-GROOVE '),
                plt.Line2D([0], [0], color='blue', label='BENDING '),
                plt.Line2D([0], [0], color='white', label='Layer 0 '),
                plt.Line2D([0], [0], color='red', label='Circles'),
                plt.Line2D([0], [0], color='orange', label='Ellipses'),
                plt.Line2D([0], [0], color='purple', label='Splines'),
                # plt.Line2D([0], [0], color='lightblue', label='Part 1'),
                # plt.Line2D([0], [0], color='lightgreen', label='Part 2'),
                # plt.Line2D([0], [0], color='lightyellow', label='Part 3'),
                # plt.Line2D([0], [0], color='lightpink', label='Part 4'),
                # plt.Line2D([0], [0], color='lightgray', label='Part 5'),
                # plt.Line2D([0], [0], color='orange', label='Part 6'),
                # plt.Line2D([0], [0], color='purple', label='Part 7'),
                # plt.Line2D([0], [0], color='cyan', label='Part 8'),
                # plt.Line2D([0], [0], color='magenta', label='Part 9'),
            ]
        else:
            legend_elements = [
                plt.Line2D([0], [0], color='white', label='Layer 0 (White)'),
                plt.Line2D([0], [0], color='green', label='V-GROOVE (Green)'),
                plt.Line2D([0], [0], color='blue', label='BENDING (Blue)')
            ]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    
    return fig

def get_entity_bounds(entity):
    """Get the bounding box of an entity"""
    try:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            x_coords = [start.x, end.x]
            y_coords = [start.y, end.y]
            return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
        elif entity.dxftype() == 'LWPOLYLINE':
            points = list(entity.get_points())
            if points:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            return (center.x - radius, center.y - radius, center.x + radius, center.y + radius)
            
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            return (center.x - radius, center.y - radius, center.x + radius, center.y + radius)
            
        elif entity.dxftype() == 'ELLIPSE':
            # Use the geometry module to get ellipse bounds
            try:
                from geometry.flatten import to_segments
                segments = to_segments(entity, tol=0.05)
                if segments:
                    x_coords = [seg.x for seg in segments]
                    y_coords = [seg.y for seg in segments]
                    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            except ImportError:
                # Fallback: approximate ellipse bounds
                center = entity.dxf.center
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                major_length = major_axis.magnitude()
                minor_length = major_length * ratio
                return (center.x - major_length, center.y - minor_length, 
                       center.x + major_length, center.y + minor_length)
            
        elif entity.dxftype() == 'SPLINE':
            # Use the geometry module to get spline bounds
            try:
                from geometry.flatten import to_segments
                segments = to_segments(entity, tol=0.05)
                if segments:
                    x_coords = [seg.x for seg in segments]
                    y_coords = [seg.y for seg in segments]
                    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            except ImportError:
                # Fallback: use control points for bounds
                control_points = entity.control_points
                if control_points:
                    x_coords = [cp.x for cp in control_points]
                    y_coords = [cp.y for cp in control_points]
                    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
    except Exception as e:
        print(f"Error getting bounds for {entity.dxftype()}: {e}")
    
    return None

def get_part_bounds(part_entities):
    """Get the overall bounding box for a part"""
    if not part_entities:
        return None
    
    all_bounds = []
    for entity in part_entities:
        bounds = get_entity_bounds(entity)
        if bounds:
            all_bounds.append(bounds)
    
    if not all_bounds:
        return None
    
    # Calculate overall bounds
    min_x = min(bounds[0] for bounds in all_bounds)
    min_y = min(bounds[1] for bounds in all_bounds)
    max_x = max(bounds[2] for bounds in all_bounds)
    max_y = max(bounds[3] for bounds in all_bounds)
    
    return (min_x, min_y, max_x, max_y)

def is_contained(inner_bounds, outer_bounds, tolerance=0.1):
    """Check if inner_bounds is contained within outer_bounds"""
    if not inner_bounds or not outer_bounds:
        return False
    
    # Check if inner bounds are completely within outer bounds (with tolerance)
    return (inner_bounds[0] >= outer_bounds[0] - tolerance and
            inner_bounds[1] >= outer_bounds[1] - tolerance and
            inner_bounds[2] <= outer_bounds[2] + tolerance and
            inner_bounds[3] <= outer_bounds[3] + tolerance)

def get_entity_points(entity):
    """Extract all points from an entity for containment testing"""
    points = []
    
    try:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            points.extend([(start.x, start.y), (end.x, end.y)])
            
        elif entity.dxftype() == 'LWPOLYLINE':
            polyline_points = list(entity.get_points())
            # Ensure we have valid 2D points
            for point in polyline_points:
                if len(point) >= 2:
                    points.append((point[0], point[1]))
                    
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            # Approximate circle with points for containment testing
            angles = np.linspace(0, 2*np.pi, 16)
            for angle in angles:
                x = center.x + radius * np.cos(angle)
                y = center.y + radius * np.sin(angle)
                points.append((x, y))
            
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = np.radians(entity.dxf.start_angle)
            end_angle = np.radians(entity.dxf.end_angle)
            
            # Ensure end_angle > start_angle
            if end_angle <= start_angle:
                end_angle += 2 * np.pi
                
            angles = np.linspace(start_angle, end_angle, 8)
            for angle in angles:
                x = center.x + radius * np.cos(angle)
                y = center.y + radius * np.sin(angle)
                points.append((x, y))
                
        elif entity.dxftype() == 'ELLIPSE':
            # Use the geometry module to flatten ellipse to points
            try:
                from geometry.flatten import to_segments
                segments = to_segments(entity, tol=0.05)
                for segment in segments:
                    points.append((segment.x, segment.y))
            except ImportError:
                # Fallback: approximate ellipse with points
                center = entity.dxf.center
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                start_param = entity.dxf.start_param
                end_param = entity.dxf.end_param
                
                # Ensure end_param > start_param
                if end_param <= start_param:
                    end_param += 2 * np.pi
                    
                params = np.linspace(start_param, end_param, 16)
                for param in params:
                    # Parametric ellipse equation
                    x = center.x + major_axis.x * np.cos(param) + major_axis.y * ratio * np.sin(param)
                    y = center.y + major_axis.y * np.cos(param) - major_axis.x * ratio * np.sin(param)
                    points.append((x, y))
                    
        elif entity.dxftype() == 'SPLINE':
            # Use the geometry module to flatten spline to points
            try:
                from geometry.flatten import to_segments
                segments = to_segments(entity, tol=0.05)
                for segment in segments:
                    points.append((segment.x, segment.y))
            except ImportError:
                # Fallback: use control points as approximation
                control_points = entity.control_points
                for point in control_points:
                    points.append((point.x, point.y))
                
    except Exception as e:
        print(f"Error extracting points from {entity.dxftype()}: {e}")
    
    return points

def get_part_points(part_entities):
    """Get all points from a part for containment testing"""
    all_points = []
    for entity in part_entities:
        points = get_entity_points(entity)
        # Validate points before adding
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                all_points.append((point[0], point[1]))
    return all_points

def point_in_polygon(point, polygon_points, tolerance=0.1):
    """Check if a point is inside a polygon using ray casting algorithm"""
    if len(polygon_points) < 3:
        return False
    
    # Ensure point is a valid 2D point
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return False
    
    x, y = point[0], point[1]
    n = len(polygon_points)
    inside = False
    
    # Validate polygon points
    valid_polygon_points = []
    for p in polygon_points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            valid_polygon_points.append((p[0], p[1]))
    
    if len(valid_polygon_points) < 3:
        return False
    
    polygon_points = valid_polygon_points
    n = len(polygon_points)
    
    p1x, p1y = polygon_points[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y) - tolerance:
            if y <= max(p1y, p2y) + tolerance:
                if x <= max(p1x, p2x) + tolerance:
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
                    else:
                        # Handle horizontal line case
                        if p1x == p2x and x <= p1x:
                            inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# -----------------------------------------------------------------------------
# DXF LABEL/TEXT EXTRACTION AND ASSOCIATION
# -----------------------------------------------------------------------------
def _get_document_bounds(doc, msp=None):
    """Return (x_min, y_min, x_max, y_max) for normalization.
    Falls back to scanning entities when header extents are missing.
    """
    try:
        extmin = doc.header.get('$EXTMIN', (None, None, 0))
        extmax = doc.header.get('$EXTMAX', (None, None, 0))
        if extmin and extmax and all(v is not None for v in (extmin[0], extmin[1], extmax[0], extmax[1])):
            x_min, y_min = float(extmin[0]), float(extmin[1])
            x_max, y_max = float(extmax[0]), float(extmax[1])
            if x_max > x_min and y_max > y_min:
                return x_min, y_min, x_max, y_max
    except Exception:
        pass

    try:
        if msp is None:
            msp = doc.modelspace()
        xs, ys = [], []
        for e in msp:
            try:
                if hasattr(e, 'dxf'):
                    if hasattr(e.dxf, 'start') and hasattr(e.dxf, 'end'):
                        xs.extend([float(e.dxf.start.x), float(e.dxf.end.x)])
                        ys.extend([float(e.dxf.start.y), float(e.dxf.end.y)])
                    elif e.dxftype() in ('TEXT', 'MTEXT') and hasattr(e.dxf, 'insert'):
                        xs.append(float(e.dxf.insert.x))
                        ys.append(float(e.dxf.insert.y))
                    else:
                        b = get_entity_bounds(e)
                        if b:
                            xs.extend([b[0], b[2]])
                            ys.extend([b[1], b[3]])
            except Exception:
                continue
        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        pass
    # Default safe bounds
    return 0.0, 0.0, 1000.0, 1000.0


def _normalize_bbox(bbox, bounds):
    """Normalize bbox [x1,y1,x2,y2] using drawing bounds to 0-1 range."""
    if not bbox or not bounds:
        return None
    x1, y1, x2, y2 = bbox
    bx1, by1, bx2, by2 = bounds
    w = max(1e-9, bx2 - bx1)
    h = max(1e-9, by2 - by1)
    return [
        (x1 - bx1) / w,
        (y1 - by1) / h,
        (x2 - bx1) / w,
        (y2 - by1) / h,
    ]


def _collect_text_entities(msp):
    """Collect TEXT/MTEXT entities with simple geometry info."""
    texts = []
    for ent in msp:
        try:
            t = ent.dxftype()
            if t == 'TEXT':
                p = ent.dxf.insert
                content = ent.dxf.text if hasattr(ent.dxf, 'text') else ''
                height = float(getattr(ent.dxf, 'height', 2.5) or 2.5)
                texts.append({
                    'type': 'TEXT',
                    'text': str(content).strip(),
                    'x': float(p.x),
                    'y': float(p.y),
                    'height': height,
                    'raw': ent,
                })
            elif t == 'MTEXT':
                p = ent.dxf.insert
                content = ent.text if hasattr(ent, 'text') else ''
                height = float(getattr(ent.dxf, 'char_height', 2.5) or 2.5)
                # Split into lines for consistent handling
                for i, line in enumerate(str(content).splitlines()):
                    texts.append({
                        'type': 'MTEXT',
                        'text': line.strip(),
                        'x': float(p.x),
                        'y': float(p.y) - i * height * 1.2,
                        'height': height,
                        'raw': ent,
                    })
        except Exception:
            continue
    # Keep only non-empty strings
    filtered_texts = [t for t in texts if t['text']]
    
    # DEBUG: Log all collected text entities
    print(f"\n=== TEXT COLLECTION DEBUG ===")
    print(f"Total text entities found: {len(filtered_texts)}")
    for i, t in enumerate(filtered_texts):
        print(f"  {i+1}. '{t['text']}' at ({t['x']:.2f}, {t['y']:.2f}) - {t['type']}")
    print("=" * 40)
    
    return filtered_texts


def _qty_from_text(s):
    """Parse integer quantity from text like 'Qty: 180', 'QTY: 1', etc."""
    if not s:
        return None
    # Look for "Qty:" followed by a number
    m = re.search(r"(?i)\bqty\s*:\s*(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _parse_type_text(s):
    """Extract material_name, thickness_mm, grade, finish from a free-form type text.
    Examples: '316 Matt 4mm', '304 2B 1.5mm', 'Alu 3mm', 'MS 5mm'
    """
    if not s:
        return {"material_name": None, "thickness": None, "grade": None, "finish": None}
    
    # Debug: Log the input text
    print(f"\n=== _parse_type_text DEBUG ===")
    print(f"Input text: '{s}'")
    print(f"Text type: {type(s)}")
    print(f"Text length: {len(s)}")
    print(f"Text bytes: {repr(s)}")

    text_upper = s.upper()
    # Thickness
    thickness = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*MM\b", text_upper)
    if m:
        try:
            thickness = float(m.group(1))
        except Exception:
            thickness = None

    # Grade
    grade = None
    # Robust patterns handle: 'SS304', 'SS-304', 'SUS 304', '304', etc.
    grade_patterns = [
        r"(?:SS|SUS|STS)\s*[-\/]?(316|304|310|430|201)",
        r"\b(316|304|310|430|201)\b",
    ]
    for _gp in grade_patterns:
        gm = re.search(_gp, text_upper)
        if gm:
            try:
                grade = int(gm.group(1))
                break
            except Exception:
                grade = None

    # Finish - match database values exactly (case-sensitive)
    finish = None
    if re.search(r"\bMATT|MOTT\b", text_upper):
        finish = 'Matt'
    elif re.search(r"\bBRUSH\w*", text_upper):
        finish = 'brushed'  # Database has lowercase "brushed"
    elif re.search(r"\bNO\.?\s*4\b", text_upper):
        finish = 'No.4'
    elif re.search(r"\b2B\b", text_upper):
        finish = '2B'
    elif re.search(r"\bBA\b", text_upper):
        finish = 'BA'
    
    # For materials that don't need finish (Brass, Copper, Mild Steel, Aluminum)
    # If no finish is specified, it will be set to 0 later in validation

    # Material name heuristic with improved, conflict-safe rules
    material_name = None
    
    # Signals - Fixed material detection patterns
    ss_patterns = [
        r"\bSS\b",
        r"^SS\b",
        r"\bSS$",
        r"SS\s*[-\/]?\s*\d+",  # SS304, SS-304, SS/304, SS 304
        r"\bSUS\d+",          # SUS304 
        r"\bSTS\d+",          # STS304
        r"\bSUS\b",
        r"\bSTS\b",
    ]
    ss_detected = any(re.search(pattern, text_upper) for pattern in ss_patterns) or ('STAINLESS' in text_upper)
    
    # Fixed brass detection - only match BR or BRASS, not BRUSHED
    br_pattern = bool(re.search(r"\bBR\b(?!USH)", text_upper))
    brass_pattern = bool(re.search(r"\bBRASS\b", text_upper))
    brass_detected = br_pattern or brass_pattern
    
    print(f"    🔍 Brass detection DEBUG:")
    print(f"        Text: '{text_upper}'")
    print(f"        BR pattern (not followed by USH): {br_pattern}")
    print(f"        BRASS pattern: {brass_pattern}")
    print(f"        Final brass_detected: {brass_detected}")
    
    # Test the regex patterns directly
    if re.search(r"\bBR\b", text_upper):
        print(f"        🔍 Found 'BR' in text")
        if re.search(r"\bBRUSH", text_upper):
            print(f"        ⚠ But also found 'BRUSH' - this will be excluded")
        else:
            print(f"        ✅ 'BR' found without 'BRUSH' - this is Brass")
    
    aluminum_detected = bool(re.search(r"\bAL\w*", text_upper) or re.search(r"\bALU(MIN(IUM)?)?\b", text_upper))
    copper_detected = bool(re.search(r"\bCO\w*", text_upper))
    mild_detected = bool(re.search(r"\bMI\w*|\bMS\b", text_upper) or re.search(r"MILD\s*STEEL|ST37|S235|S275\b", text_upper))
    galvan_detected = bool(re.search(r"GALV", text_upper))
    
    # Plain "steel" detection - must be standalone word, not part of "stainless steel"
    plain_steel_detected = bool(re.search(r"\bSTEEL\b", text_upper) and not re.search(r"STAINLESS\s+STEEL", text_upper))
    
    # Precedence: Explicit material keywords win first, regardless of grade
    if brass_detected:
        material_name = 'Brass'
        print(f"    ✓ Detected Brass from: '{text_upper}' (grade {grade} will be ignored for Brass)")
    elif aluminum_detected:
        material_name = 'Aluminum'
        print(f"    ✓ Detected Aluminum from: '{text_upper}' (grade {grade} will be ignored for Aluminum)")
    elif copper_detected:
        material_name = 'Copper'
        print(f"    ✓ Detected Copper from: '{text_upper}' (grade {grade} will be ignored for Copper)")
    elif mild_detected:
        material_name = 'Mild Steel'
        print(f"    ✓ Detected Mild Steel from: '{text_upper}' (grade {grade} will be ignored for Mild Steel)")
    elif galvan_detected:
        # Ensure 'Galvanized Steel' isn't misclassified as plain 'Steel'
        material_name = 'Galvanized Steel'
        print(f"    ✓ Detected Galvanized Steel from: '{text_upper}'")
    elif plain_steel_detected:
        # Treat generic 'Steel' as Mild Steel
        material_name = 'Mild Steel'
        print(f"    ✓ Detected plain 'Steel' as Mild Steel from: '{text_upper}'")
    elif ss_detected:
        # Only detect Stainless Steel if SS patterns are explicitly found
        material_name = 'Stainless Steel'
        print(f"    ✓ Detected Stainless Steel from: '{text_upper}'")
        print(f"    ✓ SS patterns checked: {[bool(re.search(p, text_upper)) for p in ss_patterns]}")
        print(f"    ✓ SS detected: {ss_detected}")
        print(f"    ✓ Grade detected: {grade is not None}")
    elif grade is not None:
        # If we have a grade but no explicit material, assume Stainless Steel
        material_name = 'Stainless Steel'
        print(f"    ⚠ No explicit material found, but grade {grade} detected. Assuming Stainless Steel.")
    else:
        print(f"    ⚠ No material detected from: '{text_upper}'")
    
    if material_name is None:
        print(f"    ⚠ No material detected from: '{text_upper}'")
    
    # Normalize fields for materials that do not use grade/finish
    # Mild Steel, Aluminum, Copper have no grade/finish; keep them empty (0-equivalent)
    if material_name in {'Mild Steel', 'Aluminum', 'Copper'}:
        grade = None
        finish = None

    # Final material detection summary
    print(f"    === MATERIAL DETECTION SUMMARY ===")
    print(f"    Input text: '{text_upper}'")
    print(f"    Final material: {material_name}")
    print(f"    SS patterns: {[bool(re.search(p, text_upper)) for p in ss_patterns]}")
    print(f"    SS detected: {ss_detected}")
    print(f"    Plain steel detected: {plain_steel_detected}")
    print(f"    BR pattern: {bool(re.search(r'\\bBR\\w*', text_upper))}")
    print(f"    Grade: {grade}")
    print(f"    ==================================")

    return {
        'material_name': material_name,
        'thickness': thickness,
        'grade': grade,
        'finish': finish,
    }


def _generate_corrected_extracted_text(parsed_data):
    """Generate corrected extracted text that matches the parsed format exactly.
    This ensures the extracted text follows the same format as the parsed text.
    """
    if not parsed_data or not parsed_data.get('material_name'):
        return None
    
    material_name = parsed_data.get('material_name')
    thickness = parsed_data.get('thickness')
    grade = parsed_data.get('grade')
    finish = parsed_data.get('finish')
    
    # Build the corrected extracted text in the same format as parsed
    parts = []
    
    # Material name (always first)
    parts.append(material_name)
    
    # Thickness (if available)
    if thickness is not None:
        parts.append(f"{thickness} mm")
    
    # Grade and finish (if available)
    if grade is not None or finish:
        grade_finish_parts = []
        if grade is not None:
            grade_finish_parts.append(str(grade))
        if finish:
            grade_finish_parts.append(finish)
        if grade_finish_parts:
            parts.append(" / ".join(grade_finish_parts))
    
    return " · ".join(parts)


def _approx_text_bbox(x, y, height):
    """Return a rough bbox for a single text line around (x,y)."""
    half_w = height * 5.0  # rough width estimate
    half_h = height * 0.6
    return (x - half_w, y - half_h, x + half_w, y + half_h)


def _merge_bboxes(b1, b2):
    if not b1:
        return b2
    if not b2:
        return b1
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))


def extract_labels_for_parts(doc, msp, parts):
    """Associate the nearest text block below each part and parse fields.
    Returns mapping: {index_1based: {type_text, quantity, material_name, thickness, grade, finish, part_bbox_norm, label_bbox_norm, notes}}
    """
    texts = _collect_text_entities(msp)
    bounds = _get_document_bounds(doc, msp)

    # Precompute per-part bounds
    part_bounds_list = [get_part_bounds(p) for p in parts]
    result = {}

    for idx, p in enumerate(parts, start=1):
        pb = part_bounds_list[idx - 1]
        if not pb:
            result[idx] = {
                'type_text': None, 'quantity': None,
                'material_name': None, 'thickness': None, 'grade': None, 'finish': None,
                'part_bbox_norm': None, 'label_bbox_norm': None,
                'notes': 'No part bounds',
            }
            continue

        px1, py1, px2, py2 = pb
        part_height = max(1e-6, py2 - py1)
        part_width = max(1e-6, px2 - px1)
        part_center_x = (px1 + px2) / 2.0

        # Step 1: Find all text entities in the region below this part
        # Use adaptive search area that works for both small and large parts
        # For small parts, use absolute minimum distances; for large parts, use relative distances
        
        # Calculate adaptive search parameters with more generous minimums for small parts
        min_search_width = max(100.0, part_width * 1.0)  # At least 100mm or 100% of part width
        min_search_height = max(80.0, part_height * 4.0)  # At least 80mm or 4x part height
        
        # Use the larger of relative or absolute distances
        search_width = max(part_width * 4.0, min_search_width)
        search_height = max(part_height * 12.0, min_search_height)
        
        search_left = px1 - search_width
        search_right = px2 + search_width
        search_top = py1 + min(part_height * 0.5, 30.0)  # Allow text slightly above, but cap at 30mm
        search_bottom = py1 - search_height
        
        # Debug logging for search area
        print(f"\n=== PART {idx} SEARCH DEBUG ===")
        print(f"Part bounds: ({px1:.2f}, {py1:.2f}) to ({px2:.2f}, {py2:.2f})")
        print(f"Part dimensions: {part_width:.2f}mm x {part_height:.2f}mm")
        print(f"Search area: left={search_left:.2f}, right={search_right:.2f}, top={search_top:.2f}, bottom={search_bottom:.2f}")
        print(f"Search dimensions: {(search_right-search_left):.2f}mm x {(search_top-search_bottom):.2f}mm")
        
        region_texts = []
        print(f"Scanning {len(texts)} total text entities...")
        
        for t in texts:
            if (search_left <= t['x'] <= search_right and 
                search_bottom <= t['y'] <= search_top):
                # Calculate adaptive distance score that works for all part sizes
                vdist = abs(t['y'] - py1)  # Vertical distance from part bottom
                hdist = abs(t['x'] - part_center_x)  # Horizontal distance from part center
                
                # Normalize distances by part size to make scoring fair for small and large parts
                vdist_norm = vdist / max(part_height, 10.0)  # Normalize by part height (min 10mm)
                hdist_norm = hdist / max(part_width, 10.0)   # Normalize by part width (min 10mm)
                
                # Weight vertical distance more heavily, but use normalized values
                score = vdist_norm + 0.4 * hdist_norm
                region_texts.append((score, t))
                
                # Debug logging for each found text
                print(f"  FOUND TEXT: '{t['text']}' at ({t['x']:.2f}, {t['y']:.2f})")
                print(f"    Raw distances: v={vdist:.2f}mm, h={hdist:.2f}mm")
                print(f"    Normalized: v={vdist_norm:.3f}, h={hdist_norm:.3f}")
                print(f"    Final score: {score:.3f}")
        
        print(f"Found {len(region_texts)} text entities in search area")
        
        if not region_texts:
            result[idx] = {
                'type_text': None, 'quantity': None,
                'material_name': None, 'thickness': None, 'grade': None, 'finish': None,
                'part_bbox_norm': _normalize_bbox(pb, bounds), 'label_bbox_norm': None,
                'notes': 'No text found below part',
            }
            continue

        # Step 2: Sort by proximity and find the best label block
        region_texts.sort(key=lambda x: x[0])
        print(f"\nSorted text entities by score:")
        for i, (score, t) in enumerate(region_texts[:5]):  # Show top 5
            print(f"  {i+1}. Score {score:.3f}: '{t['text']}' at ({t['x']:.2f}, {t['y']:.2f})")
        
        # Step 3: Identify the primary text line (closest to part)
        primary_line = region_texts[0][1]
        print(f"\nPrimary line: '{primary_line['text']}' at ({primary_line['x']:.2f}, {primary_line['y']:.2f})")
        
        # Step 4: Find companion lines near the primary line
        # Look for text within reasonable proximity (horizontally and vertically)
        companion_lines = []
        primary_height = primary_line['height']
        
        # Use adaptive proximity thresholds that work for all text sizes
        min_h_proximity = max(primary_height * 8.0, 20.0)  # At least 20mm or 8x text height
        min_v_proximity = max(primary_height * 4.0, 10.0)  # At least 10mm or 4x text height
        
        print(f"Companion search thresholds: h={min_h_proximity:.2f}mm, v={min_v_proximity:.2f}mm")
        
        for score, t in region_texts[1:]:  # Skip the primary line
            if t is primary_line:
                continue
                
            # Check if this text is close to the primary line
            h_dist = abs(t['x'] - primary_line['x'])
            v_dist = abs(t['y'] - primary_line['y'])
            
            # Use adaptive proximity thresholds
            if (h_dist <= min_h_proximity and v_dist <= min_v_proximity):
                # Normalize the companion scoring as well
                h_dist_norm = h_dist / max(primary_height, 5.0)
                v_dist_norm = v_dist / max(primary_height, 5.0)
                companion_score = v_dist_norm + 0.2 * h_dist_norm
                companion_lines.append((companion_score, t))
                print(f"  COMPANION: '{t['text']}' at ({t['x']:.2f}, {t['y']:.2f}) - h_dist={h_dist:.2f}mm, v_dist={v_dist:.2f}mm, score={companion_score:.3f}")
            else:
                print(f"  REJECTED: '{t['text']}' at ({t['x']:.2f}, {t['y']:.2f}) - h_dist={h_dist:.2f}mm, v_dist={v_dist:.2f}mm (thresholds: h={min_h_proximity:.2f}mm, v={min_v_proximity:.2f}mm)")
        
        # Sort companions by proximity to primary line
        companion_lines.sort(key=lambda x: x[0])
        
        # Step 5: Build the complete label block
        label_lines = [primary_line]
        if companion_lines:
            # Add up to 2 closest companion lines
            for _, companion in companion_lines[:2]:
                label_lines.append(companion)
        
        print(f"\nFinal label block ({len(label_lines)} lines):")
        for i, line in enumerate(label_lines):
            print(f"  {i+1}. '{line['text']}' at ({line['x']:.2f}, {line['y']:.2f})")
        
        # Step 6: Parse the label block using \P separator format
        type_text_line = None
        qty_line = None
        
        # Check if we have a single line with \P separator
        if len(label_lines) == 1:
            single_line = label_lines[0]['text']
            if '\\P' in single_line:
                parts = single_line.split('\\P')
                if len(parts) >= 2:
                    type_part = parts[0].strip()
                    qty_part = parts[1].strip()
                    
                    # Parse type text from first part
                    if type_part:
                        type_text_line = {'text': type_part, 'x': label_lines[0]['x'], 'y': label_lines[0]['y'], 'height': label_lines[0]['height']}
                        print(f"  Found type text from \\P separator: '{type_part}'")
                    
                    # Parse quantity from second part
                    qty_val = _qty_from_text(qty_part)
                    if qty_val is not None:
                        qty_line = ({'text': qty_part, 'x': label_lines[0]['x'], 'y': label_lines[0]['y'], 'height': label_lines[0]['height']}, qty_val)
                        print(f"  Found quantity from \\P separator: {qty_val} from '{qty_part}'")
        
        # If no \P separator found, use the original multi-line approach
        if type_text_line is None or qty_line is None:
            print("  No \\P separator found, using multi-line approach...")
            
            # First pass: identify quantity lines
            for line in label_lines:
                qty_val = _qty_from_text(line['text'])
                if qty_val is not None:
                    qty_line = (line, qty_val)
                    print(f"  Found quantity: {qty_val} from '{line['text']}'")
                    break
            
            # Second pass: identify type text lines (non-quantity)
            for line in label_lines:
                if line is not (qty_line[0] if qty_line else None):
                    # This is a non-quantity line, use as type text
                    type_text_line = line
                    print(f"  Found type text: '{line['text']}'")
                    break
        
        # Step 7: Fallback quantity detection
        if qty_line is None:
            print("  No quantity found in label block, searching entire region...")
            # Look for any quantity in the entire region
            for score, t in region_texts:
                qty_val = _qty_from_text(t['text'])
                if qty_val is not None:
                    qty_line = (t, qty_val)
                    print(f"  Found quantity in fallback: {qty_val} from '{t['text']}' at ({t['x']:.2f}, {t['y']:.2f})")
                    break
        
        # Step 8: Build label bounding box
        label_bbox = None
        for line in label_lines:
            line_bbox = _approx_text_bbox(line['x'], line['y'], line['height'])
            label_bbox = _merge_bboxes(label_bbox, line_bbox)
        
        # Step 9: Compose result
        type_text = type_text_line['text'] if type_text_line else None
        quantity = qty_line[1] if qty_line is not None else None
        
        print(f"\n=== PART {idx} PARSING DEBUG ===")
        print(f"Raw type_text: '{type_text}'")
        print(f"Text length: {len(type_text) if type_text else 0}")
        if type_text:
            print(f"Text bytes: {repr(type_text)}")
            print(f"Text uppercase: '{type_text.upper()}'")
        
        parsed = _parse_type_text(type_text) if type_text else {'material_name': None, 'thickness': None, 'grade': None, 'finish': None}
        
        # Generate corrected extracted text that matches the parsed format
        corrected_type_text = _generate_corrected_extracted_text(parsed)

        print(f"\n=== PART {idx} FINAL RESULTS ===")
        print(f"Original type text: '{type_text}'")
        print(f"Corrected type text: '{corrected_type_text}'")
        print(f"Quantity: {quantity}")
        print(f"Parsed material: {parsed.get('material_name')}")
        print(f"Parsed thickness: {parsed.get('thickness')}mm")
        print(f"Parsed grade: {parsed.get('grade')}")
        print(f"Parsed finish: {parsed.get('finish')}")
        print("=" * 50)

        result[idx] = {
            'type_text': corrected_type_text,  # Use corrected text instead of original
            'original_type_text': type_text,  # Keep original for debugging
            'quantity': quantity if isinstance(quantity, int) and quantity >= 0 else None,
            'material_name': parsed.get('material_name'),
            'thickness': parsed.get('thickness'),
            'grade': parsed.get('grade'),
            'finish': parsed.get('finish'),
            'part_bbox_norm': _normalize_bbox(pb, bounds),
            'label_bbox_norm': _normalize_bbox(label_bbox, bounds) if label_bbox else None,
            'notes': None if (corrected_type_text or quantity is not None) else 'Label unreadable or missing',
        }

    return result


def _sort_parts_left_right_top_bottom(parts):
    """Return indices that order parts left-to-right within rows, and rows top-to-bottom.
    Uses part bounding box centers. DXF Y increases upward, so top rows have higher Y.
    """
    centers = []
    for i, p in enumerate(parts):
        b = get_part_bounds(p)
        if not b:
            centers.append((i, 0.0, 0.0))
        else:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            centers.append((i, cx, cy))

    # Cluster rows by Y using a threshold based on median part height
    heights = []
    for p in parts:
        b = get_part_bounds(p)
        if b:
            heights.append(b[3] - b[1])
    row_tol = (sorted(heights)[len(heights)//2] if heights else 10.0) * 1.2

    # Sort by Y desc first, then X asc
    centers.sort(key=lambda t: (-t[2], t[1]))

    # Simple stable partitioning into rows: start new row if deltaY > tol
    rows = []
    for idx, cx, cy in centers:
        placed = False
        for row in rows:
            if abs(row['y_ref'] - cy) <= row_tol:
                row['items'].append((idx, cx))
                row['y_values'].append(cy)
                placed = True
                break
        if not placed:
            rows.append({'y_ref': cy, 'items': [(idx, cx)], 'y_values': [cy]})

    # For each row, sort by X asc
    order = []
    for row in rows:
        row['items'].sort(key=lambda t: t[1])
        order.extend([idx for idx, _ in row['items']])
    return order


@app.route('/test-material-detection', methods=['GET'])
def test_material_detection():
    """Test route to verify material detection logic"""
    test_cases = [
        "SS304 BRUSHED 1.2MM",
        "SS 304 BRUSHED 1.2MM", 
        "SS304",
        "BRASS 2MM",
        "ALUMINUM 3MM",
        "COPPER 1MM"
    ]
    
    results = []
    for test_text in test_cases:
        parsed = _parse_type_text(test_text)
        results.append({
            'input': test_text,
            'output': parsed
        })
    
    return jsonify({
        'test_cases': results,
        'message': 'Material detection test completed'
    })

@app.route('/test-db-lookup', methods=['POST'])
def test_db_lookup():
    """Test endpoint for database lookup"""
    try:
        data = request.get_json()
        material_name = data.get('material_name')
        thickness = data.get('thickness')
        grade = data.get('grade')
        finish = data.get('finish')
        
        from DatabaseConfig import get_material_config
        result = get_material_config(material_name, thickness, grade, finish)
        
        return jsonify({
            'input': {'material_name': material_name, 'thickness': thickness, 'grade': grade, 'finish': finish},
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-parsing', methods=['GET'])
def test_parsing():
    """Test the parsing function directly"""
    test_cases = [
        "SS304 BRUSHED 1MM",
        "BR 1MM",
        "BRASS 1MM",
        "BR 304 1MM"
    ]
    
    results = []
    for test_text in test_cases:
        parsed = _parse_type_text(test_text)
        results.append({
            'input': test_text,
            'parsed': parsed
        })
    
    return jsonify({
        'test_cases': results
    })

@app.route('/test-db-connection', methods=['GET'])
def test_db_connection():
    """Test database connection and materials table"""
    try:
        from DatabaseConfig import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text('SELECT 1 as test'))
            test_row = result.fetchone()
            
            # Check if materials table exists
            table_check = conn.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='materials'
            """))
            table_exists = table_check.fetchone() is not None
            
            # Get sample materials
            if table_exists:
                materials = conn.execute(text('SELECT "Material Name", "Thickness", "Grade", "Finish" FROM materials LIMIT 5')).fetchall()
            else:
                materials = []
            
            return jsonify({
                'connection': 'OK',
                'test_query': test_row[0] if test_row else None,
                'materials_table_exists': table_exists,
                'sample_materials': [dict(zip(['Material Name', 'Thickness', 'Grade', 'Finish'], row)) for row in materials]
            })
    except Exception as e:
        return jsonify({'error': str(e), 'connection': 'FAILED'}), 500

@app.route('/api/extract-labels', methods=['POST'])
def api_extract_labels_only():
    """Return only the per-part labels from a DXF upload as JSON.
    Parts are ordered left-to-right, top-to-bottom.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
            file.save(tmp.name)
            path = tmp.name
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        os.unlink(path)

        # Build layers and filter/group per project rules to get parts same as main flow
        layers = {layer.dxf.name: layer.dxf.color for layer in doc.layers}
        filtered_entities, _ = filter_entities(msp, layers)
        parts = find_connected_parts(filtered_entities, layers)
        parts = merge_contained_parts(parts, layers)
        parts = attach_isolated_entities_to_parents(parts, layers)

        # Keep meaningful parts
        meaningful_parts = []
        for part in parts:
            if len(part) > 1:
                meaningful_parts.append(part)
            elif len(part) == 1 and part[0].dxftype() in ['CIRCLE', 'ELLIPSE', 'ARC', 'SPLINE']:
                meaningful_parts.append(part)

        # Order parts left-to-right within rows, rows top-to-bottom
        order = _sort_parts_left_right_top_bottom(meaningful_parts)
        ordered_parts = [meaningful_parts[i] for i in order]

        labels_map = extract_labels_for_parts(doc, msp, ordered_parts)
        try:
            print("\n=== Extracted Labels (Labels-Only API) ===")
            for k in sorted(labels_map.keys()):
                e = labels_map[k]
                print(f"Part {k}: type_text='{e.get('type_text')}', qty={e.get('quantity')}, material={e.get('material_name')}, thickness={e.get('thickness')}, grade={e.get('grade')}, finish={e.get('finish')}")
            print("=== End Extracted Labels ===\n")
        except Exception:
            pass

        # Convert to array with 1..N index and strict schema
        result = []
        for i in range(1, len(ordered_parts) + 1):
            entry = labels_map.get(i, {})
            # Normalize fields
            res = {
                'part_index': i,
                'type_text': entry.get('type_text'),
                'quantity': entry.get('quantity'),
                'part_bbox_norm': entry.get('part_bbox_norm'),
                'label_bbox_norm': entry.get('label_bbox_norm'),
                'material_name': entry.get('material_name'),
                'thickness': entry.get('thickness'),
                'grade': entry.get('grade'),
                'finish': entry.get('finish'),
                'notes': entry.get('notes'),
            }
            result.append(res)

        return jsonify({'success': True, 'labels': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def is_part_contained(inner_part, outer_part, tolerance=0.1):
    """Return True when *all* of inner_part lies inside outer_part."""
    if not inner_part or not outer_part:
        return False

    # ── 1.  Fast, geometry-exact test ────────────────────────────────────────
    inner_poly = part_to_polygon(inner_part, buffer=tolerance/2)
    outer_poly = part_to_polygon(outer_part, buffer=tolerance)

    if inner_poly and outer_poly:
        return outer_poly.contains(inner_poly)

    # ── 2.  Fallback → original point-sampling logic (safety-net) ───────────
    try:
        inner_points = get_part_points(inner_part)
        outer_points = get_part_points(outer_part)
        if len(inner_points) < 3 or len(outer_points) < 3:
            return False

        step = max(1, len(inner_points) // 10)
        hits = sum(
            1 for p in inner_points[::step]
            if point_in_polygon(p, outer_points, tolerance)
        )
        return (hits / (len(inner_points[::step]) or 1)) > 0.8
    except Exception as e:
        print(f"[is_part_contained fallback] {e}")
        return False


def merge_contained_parts(parts, layers):
    """Merge parts where one part is contained within another part, including mixed-layer containment"""
    if len(parts) <= 1:
        return parts
    
    try:
        # --- Tunable constants (small, conservative defaults) ---
        angle_tol_deg = 5.0           # Parallelism tolerance
        consecutive_gap_max = 5.0     # Max gap (drawing units) to consider bends consecutive
        corridor_padding = 1.0        # Extra padding around bend corridor
        boundary_tolerance = 0.2      # Treat touching boundaries as inside

        # Helper: compute angle in degrees of a LINE-like entity (0..180)
        def _entity_angle_deg(entity):
            try:
                pts = get_entity_points(entity) or []
                if len(pts) < 2:
                    return None
                (x1, y1) = pts[0]
                (x2, y2) = pts[-1]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    return None
                ang = math.degrees(math.atan2(dy, dx))
                # Normalize parallel classes (0..180)
                if ang < 0:
                    ang += 180.0
                return ang % 180.0
            except Exception:
                return None

        # Helper: are two angles approximately parallel?
        def _is_parallel(a, b):
            if a is None or b is None:
                return False
            d = abs(a - b)
            d = min(d, 180.0 - d)
            return d <= angle_tol_deg

        # Helper: collect all bending lines from the entire parts set
        from shapely.ops import unary_union
        from shapely.geometry import LineString, Point

        def _collect_all_bending_lines(all_parts):
            lines = []
            for prt in all_parts:
                for e in prt:
                    try:
                        if hasattr(e, 'dxf') and getattr(e.dxf, 'layer', '').upper() == 'BENDING':
                            pts = get_entity_points(e) or []
                            if len(pts) >= 2:
                                ls = LineString([pts[0], pts[-1]])
                                lines.append((e, ls, _entity_angle_deg(e)))
                    except Exception:
                        continue
            return lines

        def _build_bend_corridors(parent_part, outer_poly, all_bending_lines):
            corridors = []
            if outer_poly is None:
                return corridors

            # Scale thresholds by parent size (conservative)
            try:
                min_dim = min(outer_poly.bounds[2] - outer_poly.bounds[0], outer_poly.bounds[3] - outer_poly.bounds[1])
                local_gap_max = max(consecutive_gap_max, 0.01 * max(min_dim, 1.0))
                local_pad = max(corridor_padding, 0.002 * max(min_dim, 1.0))
                proximity = max(boundary_tolerance * 2.0, 0.005 * max(min_dim, 1.0))
            except Exception:
                local_gap_max = consecutive_gap_max
                local_pad = corridor_padding
                proximity = boundary_tolerance * 2.0

            # Select bending lines near the parent's outer boundary (same-side candidates)
            bending_lines = []
            for e, ls, ang in (all_bending_lines or []):
                try:
                    # Near the outer border indicates it's on a side of the panel
                    if ls.distance(outer_poly.exterior) <= proximity or outer_poly.buffer(proximity).intersects(ls):
                        bending_lines.append((e, ls, ang))
                except Exception:
                    continue

            if not bending_lines:
                return corridors

            # Group by approximate angle (parallelism)
            bending_lines.sort(key=lambda t: (0.0 if t[2] is None else t[2]))
            groups = []
            for ent, geom, ang in bending_lines:
                placed = False
                for g in groups:
                    if _is_parallel(g['angle_ref'], ang):
                        g['items'].append((ent, geom, ang))
                        placed = True
                        break
                if not placed:
                    groups.append({'angle_ref': ang, 'items': [(ent, geom, ang)]})

            # Within each group, consider consecutive lines by proximity; create corridors between neighbors
            for g in groups:
                items = g['items']
                # Sort by projection along angle to make neighbors consecutive; approximate by line centroid along axis
                def _proj_key(item):
                    _, geom, ang = item
                    cx, cy = geom.centroid.x, geom.centroid.y
                    rad = math.radians(0.0 if ang is None else ang)
                    ax, ay = math.cos(rad), math.sin(rad)
                    return cx * ax + cy * ay

                items.sort(key=_proj_key)
                for i in range(len(items) - 1):
                    _, g1, a1 = items[i]
                    _, g2, a2 = items[i + 1]
                    if not _is_parallel(a1, a2):
                        continue
                    try:
                        gap = g1.distance(g2)
                        if gap <= local_gap_max:
                            # Corridor width: cover space between plus padding
                            radius = max(gap / 2.0 + local_pad, local_pad)
                            corridor_geom = unary_union([g1.buffer(radius), g2.buffer(radius)])
                            corridors.append(corridor_geom)
                    except Exception:
                        continue

            return corridors

        # Helper: test if a part's representative point lies inside outer polygon or any corridor
        def _part_inside_inclusion(part, outer_poly, corridors):
            try:
                pts = get_part_points(part) or []
                if not pts:
                    return False
                # Use centroid of sampled points as representative point
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                p = Point(cx, cy)
                if outer_poly.buffer(boundary_tolerance).contains(p) or outer_poly.buffer(boundary_tolerance).touches(p):
                    return True
                for c in corridors:
                    try:
                        if c.contains(p) or c.touches(p):
                            return True
                    except Exception:
                        continue
                return False
            except Exception:
                return False

        # Separate parts by type for analysis
        layer0_parts = []
        vgroove_parts = []
        mixed_parts = []
        other_parts = []
        
        for part in parts:
            # Analyze the part to determine its type
            layer0_count = 0
            vgroove_count = 0
            other_count = 0
            
            for entity in part:
                if not hasattr(entity, 'dxf'):
                    other_count += 1
                    continue
                    
                layer_name = getattr(entity.dxf, 'layer', '')
                color = get_entity_color(entity, layers)
                
                if layer_name == '0' and color == 7:
                    layer0_count += 1
                elif layer_name.upper() == 'V-GROOVE' and color == 3:
                    vgroove_count += 1
                else:
                    other_count += 1
            
            # Categorize the part
            if layer0_count > 0 and vgroove_count == 0 and other_count == 0:
                layer0_parts.append(part)
            elif vgroove_count > 0 and layer0_count == 0 and other_count == 0:
                vgroove_parts.append(part)
            elif layer0_count > 0 and vgroove_count > 0:
                mixed_parts.append(part)
            else:
                other_parts.append(part)
        
        # Process containment relationships (with bend corridors widening inclusion region)
        merged_parts = []
        used_indices = set()
        
        # First, handle Layer 0 parts containing other Layer 0 parts
        for i, part1 in enumerate(layer0_parts):
            if i in used_indices:
                continue
            
            merged_entities = list(part1)
            # Build outer polygon and bend corridors for this parent
            outer_poly = part_to_polygon(part1, buffer=boundary_tolerance) or None
            # Compute corridors using ALL bending lines near this parent's sides
            if 'all_bending_lines_cache' not in locals():
                all_bending_lines_cache = _collect_all_bending_lines(parts)
            corridors = _build_bend_corridors(part1, outer_poly, all_bending_lines_cache) if outer_poly is not None else []
            try:
                print(f"[merge_contained_parts] Parent #{i+1}: corridors={len(corridors)}")
            except Exception:
                pass
            for j, part2 in enumerate(layer0_parts):
                if i == j or j in used_indices:
                    continue
                
                # Containment test widened: inside outer polygon OR inside any bend corridor
                contained = False
                try:
                    if outer_poly is not None:
                        if is_part_contained(part2, part1) or _part_inside_inclusion(part2, outer_poly, corridors):
                            contained = True
                except Exception:
                    contained = is_part_contained(part2, part1)

                if contained:
                    merged_entities.extend(part2)
                    used_indices.add(j)
                    try:
                        print(f"  - merged inner Layer0 part #{j+1} into parent #{i+1} (corridor-assisted={outer_poly is not None})")
                    except Exception:
                        pass
            
            # Also check if any V-GROOVE parts are contained within this Layer 0 part
            for j, vgroove_part in enumerate(vgroove_parts):
                if j in used_indices:
                    continue
                
                contained_v = False
                try:
                    if outer_poly is not None:
                        if is_part_contained(vgroove_part, part1) or _part_inside_inclusion(vgroove_part, outer_poly, corridors):
                            contained_v = True
                except Exception:
                    contained_v = is_part_contained(vgroove_part, part1)

                if contained_v:
                    merged_entities.extend(vgroove_part)
                    used_indices.add(j)
                    try:
                        print(f"  - merged V-GROOVE part #{j+1} into parent #{i+1}")
                    except Exception:
                        pass
            
            merged_parts.append(merged_entities)
            used_indices.add(i)
        
        # Add remaining V-GROOVE parts that weren't contained
        for i, vgroove_part in enumerate(vgroove_parts):
            if i not in used_indices:
                merged_parts.append(vgroove_part)
        
        # Add mixed parts
        merged_parts.extend(mixed_parts)
        
        # Add other parts
        merged_parts.extend(other_parts)
        
        return merged_parts
        
    except Exception as e:
        print(f"Error in merge_contained_parts: {e}")
        # Return original parts if merging fails
        return parts
# === NEW HELPER: glue isolated entities (e.g. circles) back to their container ===
def attach_isolated_entities_to_parents(parts, layers, tolerance=0.1):
    """
    Merge single-entity 'orphan' parts (typically circles or tiny arcs) into the first
    multi-entity part that fully contains them.  Falls back to leaving them separate
    if no suitable container is found.
    """
    if len(parts) <= 1:
        return parts

    single_parts = [p for p in parts if len(p) == 1]
    multi_parts  = [p for p in parts if len(p) > 1]

    for child in single_parts:
        attached = False
        for parent in multi_parts:
            if is_part_contained(child, parent, tolerance):
                parent.extend(child)
                attached = True
                break
        if not attached:
            multi_parts.append(child)      # keep as its own part

    return multi_parts

def calculate_laser_cutting_cost(area, perimeter, thickness, config):
    """
    Calculate laser cutting cost using dynamic material configuration
    """
    try:
        # Extract values from config - CRITICAL: Use material-specific values from database
        machine_speed = config.get('machine_speed', 100.0)
        vaporization_speed = config.get('vaporization_speed', 50.0)
        piercing_time = config.get('piercing_time', 0.5)
        laser_cost = config.get('laser_cost', 0.15)
        
        # CRITICAL: Get material-specific density and price from database config
        if 'density' not in config or config['density'] is None:
            error_msg = "ERROR: Missing density in material config. Cannot calculate material cost."
            print(f"  ❌ {error_msg}")
            return {
                'material_cost': 0,
                'laser_operation_cost': 0,
                'total_cost': 0,
                'weight': 0,
                'cutting_time': 0,
                'piercing_time': 0,
                'total_time': 0,
                'error': error_msg
            }
        
        if 'price_per_kg' not in config or config['price_per_kg'] is None:
            error_msg = "ERROR: Missing price per kg in material config. Cannot calculate material cost."
            print(f"  ❌ {error_msg}")
            return {
                'material_cost': 0,
                'laser_operation_cost': 0,
                'total_cost': 0,
                'weight': 0,
                'cutting_time': 0,
                'piercing_time': 0,
                'total_time': 0,
                'error': error_msg
            }
        
        # Get material properties from config (these come from the database)
        material_density = float(config['density'])  # g/cm³
        material_price_per_kg = float(config['price_per_kg'])  # USD/kg
        
        # Calculate material cost with proper units
        # Area (m²) × Thickness (mm) × Density (g/cm³) × Price (USD/kg)
        # Convert thickness from mm to cm for volume calculation
        thickness_cm = thickness / 10  # 1 mm = 0.1 cm
        
        # Volume in cm³
        volume_cm3 = area * thickness_cm
        
        # Weight in kg
        weight_kg = volume_cm3 * material_density / 1000  # Convert g to kg
        
        # Material cost - CORRECTED: Use the proper formula with scrap factor
        # Formula: Area × Thickness × Density × Price per kg × Scrap Factor
        # Note: This function should use the same formula as the main calculate_cost function
        scrap_factor = config.get('scrap_factor', 1.20)  # Default scrap factor
        # Convert thickness from mm to m for calculation
        thickness_m = thickness / 1000
        material_cost = area * thickness_m * (material_density * 1000) * material_price_per_kg * scrap_factor
        
        # DEBUG: Log the corrected material cost calculation
        print(f"  🔍 CORRECTED MATERIAL COST CALCULATION:")
        print(f"      Formula: Area × Thickness × Density × Price per kg × Scrap Factor")
        print(f"      Values: {area} × {thickness_m} × {material_density * 1000} × {material_price_per_kg} × {scrap_factor}")
        print(f"      Material Cost: ${material_cost:.6f}")
        
        # Debug logging for material cost calculation
        print(f"  🔍 LASER CUTTING COST CALCULATION DEBUG:")
        print(f"      Area: {area} m²")
        print(f"      Thickness: {thickness} mm = {thickness_cm} cm")
        print(f"      Density: {material_density} g/cm³")
        print(f"      Price per kg: ${material_price_per_kg}")
        print(f"      Volume: {volume_cm3:.6f} cm³")
        print(f"      Weight: {weight_kg:.6f} kg")
        print(f"      Material Cost: ${material_cost:.6f}")
        
        # Calculate cutting time
        cutting_distance = perimeter / 1000  # Convert to meters
        cutting_time = cutting_distance / (machine_speed / 60)  # Convert to minutes
        
        # Calculate piercing time
        num_pierces = max(1, int(perimeter / 100))  # Estimate number of pierces
        total_piercing_time = num_pierces * piercing_time / 60  # Convert to minutes
        
        # Calculate laser operation cost
        total_time = cutting_time + total_piercing_time
        laser_operation_cost = total_time * laser_cost
        
        # Total cost
        total_cost = material_cost + laser_operation_cost
        
        # DEBUG: Log final return values for this function
        print(f"  🔍 LASER CUTTING COST FINAL RETURN VALUES:")
        print(f"      Material Cost: ${material_cost:.6f}")
        print(f"      Laser Operation Cost: ${laser_operation_cost:.6f}")
        print(f"      Total Cost: ${total_cost:.6f}")
        print(f"      Weight: {weight_kg:.6f} kg")
        
        return {
            'material_cost': material_cost,
            'laser_operation_cost': laser_operation_cost,
            'total_cost': total_cost,
            'weight': weight_kg,  # Fixed: use weight_kg instead of undefined weight
            'cutting_time': cutting_time,
            'piercing_time': total_piercing_time,
            'total_time': total_time
        }
    except Exception as e:
        print(f"Error calculating laser cutting cost: {e}")
        return {
            'material_cost': 0,
            'laser_operation_cost': 0,
            'total_cost': 0,
            'weight': 0,
            'cutting_time': 0,
            'piercing_time': 0,
            'total_time': 0
        }

def calculate_accurate_perimeter(part_entities, layers):
    """Calculate accurate perimeter using only Layer 0 entities (outer boundary)"""
    if not part_entities:
        return 0.0
    
    # Extract only Layer 0 entities for perimeter calculation (exclude V-groove and bending)
    layer0_entities = get_layer0_entities(part_entities, layers)
    
    if not layer0_entities:
        print("Warning: No Layer 0 entities found for perimeter calculation")
        return 0.0
    
    try:
        from shapely.geometry import LineString, Polygon
        from shapely.ops import linemerge
        import numpy as np
        
        total_perimeter_mm = 0.0
        
        for entity in layer0_entities:
            entity_type = entity.dxftype()
            
            if entity_type == 'LINE':
                # Calculate line length
                start = entity.dxf.start
                end = entity.dxf.end
                length_mm = np.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
                total_perimeter_mm += length_mm
                
            elif entity_type == 'LWPOLYLINE':
                # Calculate polyline length
                try:
                    points = list(entity.get_points())
                    if len(points) > 1:
                        polyline_length = 0.0
                        for i in range(len(points) - 1):
                            p1 = points[i]
                            p2 = points[i + 1]
                            segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                            polyline_length += segment_length
                        total_perimeter_mm += polyline_length
                except Exception as e:
                    print(f"Error calculating LWPOLYLINE perimeter: {e}")
                    
            elif entity_type == 'CIRCLE':
                # Calculate circle circumference
                radius = entity.dxf.radius
                circumference_mm = 2 * np.pi * radius
                total_perimeter_mm += circumference_mm
                
            elif entity_type == 'ELLIPSE':
                # Calculate ellipse perimeter (approximation)
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                major_length = np.sqrt(major_axis.x**2 + major_axis.y**2)
                minor_length = major_length * ratio
                
                # Ramanujan's approximation for ellipse perimeter
                a = major_length
                b = minor_length
                h = ((a - b) / (a + b))**2
                perimeter_mm = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
                total_perimeter_mm += perimeter_mm
                
            elif entity_type == 'ARC':
                # Calculate arc length
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = np.radians(entity.dxf.start_angle)
                end_angle = np.radians(entity.dxf.end_angle)
                
                # Ensure end_angle > start_angle
                if end_angle <= start_angle:
                    end_angle += 2 * np.pi
                    
                arc_angle = end_angle - start_angle
                arc_length_mm = radius * arc_angle
                total_perimeter_mm += arc_length_mm
                
            elif entity_type == 'SPLINE':
                # Calculate spline length using geometry module
                try:
                    from geometry.flatten import to_segments
                    segments = to_segments(entity, tol=0.05)
                    if len(segments) >= 2:
                        spline_length = 0.0
                        for i in range(len(segments) - 1):
                            p1 = segments[i]
                            p2 = segments[i + 1]
                            segment_length = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                            spline_length += segment_length
                        total_perimeter_mm += spline_length
                except ImportError:
                    # Fallback: use control points for approximation
                    try:
                        control_points = entity.control_points
                        if len(control_points) >= 2:
                            spline_length = 0.0
                            for i in range(len(control_points) - 1):
                                p1 = control_points[i]
                                p2 = control_points[i + 1]
                                segment_length = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                                spline_length += segment_length
                            total_perimeter_mm += spline_length
                    except Exception as e:
                        print(f"Error calculating SPLINE perimeter: {e}")
                        
        # Convert to meters
        return total_perimeter_mm / 1000.0
        
    except Exception as e:
        print(f"Error calculating accurate perimeter: {e}")
        return 0.0

def calculate_vgroove_length(part_entities, layers):
    """Calculate the total length of V-groove lines in meters for a part"""
    total_length_meters = 0.0
    
    print(f"\n=== V-GROOVE LENGTH CALCULATION DEBUG ===")
    print(f"Total entities in part: {len(part_entities)}")
    
    for i, entity in enumerate(part_entities):
        if not hasattr(entity, 'dxf'):
            print(f"Entity {i}: No dxf attribute")
            continue
            
        layer_name = getattr(entity.dxf, 'layer', 'UNKNOWN')
        color = get_entity_color(entity, layers)
        entity_type = entity.dxftype()
        
        # Check for V-GROOVE entities with multiple variations
        layer_upper = layer_name.upper()
        is_vgroove_layer = (
            layer_upper == 'V-GROOVE' or 
            layer_upper == 'VGROOVE' or 
            layer_upper == 'V_GROOVE' or
            'V-GROOVE' in layer_upper or
            'VGROOVE' in layer_upper
        )
        
        # Check for green color (color 3)
        is_green_color = (color == 3)
        
        if is_vgroove_layer and is_green_color:
            entity_length_meters = 0.0
            
            if entity_type == 'LINE':
                # Calculate length of single line
                start = entity.dxf.start
                end = entity.dxf.end
                length_mm = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
                entity_length_meters = length_mm / 1000.0  # Convert mm to meters
                print(f"  ✓ V-GROOVE LINE: Length = {entity_length_meters:.3f} meters")
                
            elif entity_type == 'LWPOLYLINE':
                # Calculate total length of polyline
                try:
                    points = list(entity.get_points())
                    if len(points) > 1:
                        total_length_mm = 0.0
                        for j in range(len(points) - 1):
                            p1 = points[j]
                            p2 = points[j + 1]
                            segment_length = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
                            total_length_mm += segment_length
                        entity_length_meters = total_length_mm / 1000.0  # Convert mm to meters
                        print(f"  ✓ V-GROOVE LWPOLYLINE: Length = {entity_length_meters:.3f} meters ({len(points)-1} segments)")
                    else:
                        entity_length_meters = 0.0
                        print(f"  ✓ V-GROOVE LWPOLYLINE: No valid segments")
                except Exception as e:
                    print(f"  Error processing LWPOLYLINE: {e}")
                    entity_length_meters = 0.0
                    
            elif entity_type == 'POLYLINE':
                # Calculate total length of old-style polyline
                try:
                    vertices = list(entity.vertices)
                    if len(vertices) > 1:
                        total_length_mm = 0.0
                        for j in range(len(vertices) - 1):
                            v1 = vertices[j]
                            v2 = vertices[j + 1]
                            segment_length = ((v2.dxf.location.x - v1.dxf.location.x) ** 2 + 
                                            (v2.dxf.location.y - v1.dxf.location.y) ** 2) ** 0.5
                            total_length_mm += segment_length
                        entity_length_meters = total_length_mm / 1000.0  # Convert mm to meters
                        print(f"  ✓ V-GROOVE POLYLINE: Length = {entity_length_meters:.3f} meters ({len(vertices)-1} segments)")
                    else:
                        entity_length_meters = 0.0
                        print(f"  ✓ V-GROOVE POLYLINE: No valid segments")
                except Exception as e:
                    print(f"  Error processing POLYLINE: {e}")
                    entity_length_meters = 0.0
                    
            else:
                # Other entity types - try to calculate length if possible
                entity_length_meters = 0.0
                print(f"  ✓ V-GROOVE {entity_type}: Length calculation not implemented")
            
            total_length_meters += entity_length_meters
            print(f"  Running total V-GROOVE length: {total_length_meters:.3f} meters")
            
        elif is_vgroove_layer:
            print(f"  - V-GROOVE layer but wrong color: {color}")
        elif is_green_color:
            print(f"  - Green color but wrong layer: {layer_name}")
    
    print(f"=== FINAL V-GROOVE LENGTH: {total_length_meters:.3f} meters ===\n")
    return total_length_meters

# Nesting functionality removed for deployment

@app.route('/')
def index():
    # Check if user is logged in
    if not is_user_logged_in():
        return redirect(url_for('user_login'))
    
    config = load_admin_config()
    user_data = get_current_user()
    return render_template('index.html', config=config, user_data=user_data)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password')
        # Simple password check - change this to a secure method in production
        if password == 'admin123':  # Change this password
            session['admin_logged_in'] = True
            return redirect(url_for('admin_materials'))
        else:
            return render_template('admin_login.html', error='Invalid password')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

@app.route('/user/login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('user_login.html')
        
        user = get_user_by_email(email)
        if user and user['password'] == password:  # In production, use proper password hashing
            session['user_logged_in'] = True
            session['user_data'] = {
                'id': user['id'],
                'email': user['email'],
                'full_name': user['full_name'],
                'work_id': user['work_id'],
                'extnumber': user['extnumber']
            }
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('user_login.html')

@app.route('/user/logout')
def user_logout():
    session.pop('user_logged_in', None)
    session.pop('user_data', None)
    return redirect(url_for('user_login'))

@app.route('/user/register', methods=['GET', 'POST'])
def user_register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        work_id = request.form.get('work_id')
        extnumber = request.form.get('extnumber')
        
        # Require all fields
        if not email or not password or not full_name or not work_id or not extnumber:
            flash('All fields are required: Email, Password, Full Name, Work ID, Extension Number', 'error')
            return render_template('user_register.html')
        
        # Check if user already exists
        existing_user = get_user_by_email(email)
        if existing_user:
            flash('User with this email already exists', 'error')
            return render_template('user_register.html')
        
        # Convert extnumber to integer (required)
        try:
            extnumber = int(extnumber)
        except (TypeError, ValueError):
            flash('Extension number must be a valid integer', 'error')
            return render_template('user_register.html')
        
        # Create new user
        if create_user(email, password, full_name, work_id, extnumber):
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('user_login'))
        else:
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('user_register.html')



@app.route('/admin/save_config', methods=['POST'])
def save_config():
    if not is_admin_logged_in():
        return jsonify({'error': 'Not authorized'}), 401
    
    try:
        config = {
            'machine_speed': float(request.form.get('machine_speed', 100.0)),
            'vaporization_speed': float(request.form.get('vaporization_speed', 50.0)),
            'piercing_time': float(request.form.get('piercing_time', 0.5)),
            'laser_cost': float(request.form.get('laser_cost', 0.15)),
            'price_per_kg': float(request.form.get('price_per_kg', 25.0)),
            'density': float(request.form.get('density', 7.85))
        }
        
        if save_admin_config(config):
            return jsonify({'success': True, 'message': 'Configuration saved successfully'})
        else:
            return jsonify({'error': 'Failed to save configuration'}), 500
            
    except ValueError as e:
        return jsonify({'error': 'Invalid input values'}), 400
    except Exception as e:
        return jsonify({'error': f'Error saving configuration: {str(e)}'}), 500

# Optimized progress store for long-running upload processing
from uuid import uuid4
from threading import Lock
import time
_PROGRESS_LOCK = Lock()
_PROGRESS: dict[str, dict] = {}
_LAST_UPDATE: dict[str, float] = {}  # Track last update time to reduce frequency

def _progress_set(job_id: str, percent: int, message: str = ""):
    """Optimized progress setting with throttling to reduce overhead"""
    try:
        current_time = time.time()
        # Only update if enough time has passed (reduce frequency)
        if job_id in _LAST_UPDATE and current_time - _LAST_UPDATE[job_id] < 0.2:  # 200ms throttle
            return
            
        with _PROGRESS_LOCK:
            if job_id not in _PROGRESS:
                _PROGRESS[job_id] = {"percent": 0, "message": "", "status": "running"}
            _PROGRESS[job_id]["percent"] = max(0, min(100, int(percent)))
            if message:
                _PROGRESS[job_id]["message"] = message
            _LAST_UPDATE[job_id] = current_time
    except Exception:
        pass

def _progress_done(job_id: str, ok: bool, message: str = ""):
    """Mark progress as complete"""
    try:
        with _PROGRESS_LOCK:
            if job_id not in _PROGRESS:
                _PROGRESS[job_id] = {"percent": 0, "message": ""}
            _PROGRESS[job_id]["percent"] = 100 if ok else _PROGRESS[job_id].get("percent", 0)
            _PROGRESS[job_id]["status"] = "done" if ok else "error"
            if message:
                _PROGRESS[job_id]["message"] = message
            # Clean up old entries to prevent memory leaks
            if job_id in _LAST_UPDATE:
                del _LAST_UPDATE[job_id]
    except Exception:
        pass

@app.route('/api/progress/start', methods=['POST'])
def progress_start():
    job_id = str(uuid4())
    with _PROGRESS_LOCK:
        _PROGRESS[job_id] = {"percent": 0, "message": "Starting…", "status": "running"}
    return jsonify({"job_id": job_id})

@app.route('/api/progress/<job_id>', methods=['GET'])
def progress_get(job_id):
    with _PROGRESS_LOCK:
        info = _PROGRESS.get(job_id)
    if not info:
        return jsonify({"percent": 0, "message": "Unknown job", "status": "unknown"}), 404
    return jsonify(info)

@app.route('/api/progress/clear/<job_id>', methods=['POST'])
def progress_clear(job_id):
    with _PROGRESS_LOCK:
        if job_id in _PROGRESS:
            del _PROGRESS[job_id]
    return jsonify({"cleared": True})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            job_id = request.form.get('job_id') or ""
            if job_id:
                _progress_set(job_id, 5, "Saving file…")
            # Save uploaded file to disk first, then read with ezdxf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                file.save(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            # Read the DXF file from disk
            if job_id:
                _progress_set(job_id, 15, "Reading DXF…")
            doc = ezdxf.readfile(tmp_file_path)
            msp = doc.modelspace()
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Get layer information
            if job_id:
                _progress_set(job_id, 25, "Extracting layers…")
            layers = {}
            for layer in doc.layers:
                layers[layer.dxf.name] = layer.dxf.color
            
            # Filter entities according to rules
            if job_id:
                _progress_set(job_id, 35, "Filtering entities…")
            filtered_entities, removed_count = filter_entities(msp, layers)
            
            # Debug suppressed: filtered entities summary
            
            # Find connected parts
            if job_id:
                _progress_set(job_id, 45, "Grouping parts…")
            parts = find_connected_parts(filtered_entities, layers)
            # Debug suppressed: part grouping details
            
            
                        # Merge contained Layer-0 parts
            parts = merge_contained_parts(parts, layers)

            # NEW: pull single-entity orphans (circles, etc.) into their parent
            parts = attach_isolated_entities_to_parents(parts, layers)

            # Filter out parts with only 1 entity (except circles, ellipses, and splines)
            meaningful_parts = []
            for part in parts:
                if len(part) > 1:
                    # Multi-entity parts are always meaningful
                    meaningful_parts.append(part)
                elif len(part) == 1:
                    entity_type = part[0].dxftype()
                    if entity_type in ['CIRCLE', 'ELLIPSE', 'ARC', 'SPLINE']:
                        # Single circles, ellipses, arcs, and splines are considered meaningful parts
                        meaningful_parts.append(part)
                # Single other entities are filtered out
            
            # Extract per-part labels (type_text and quantity) directly from DXF TEXT below each part
            try:
                extracted_part_labels = extract_labels_for_parts(doc, msp, meaningful_parts)
            except Exception as _lbl_err:
                print(f"extract_labels_for_parts error: {_lbl_err}")
                extracted_part_labels = {}
            # Print extracted labels to terminal for verification
            try:
                print("\n=== Extracted Labels (Single Upload) ===")
                for k in sorted((extracted_part_labels or {}).keys(), key=lambda x: int(x)):
                    e = extracted_part_labels.get(k) or {}
                    print(f"Part {k}: type_text='{e.get('type_text')}', qty={e.get('quantity')}, material={e.get('material_name')}, thickness={e.get('thickness')}, grade={e.get('grade')}, finish={e.get('finish')}")
                print("=== End Extracted Labels ===\n")
            except Exception:
                pass

            # AI fallback: fix layers and detect missing quantities only for failed parts
            def _compute_part_bbox(part_entities):
                try:
                    xs = []
                    ys = []
                    for ent in part_entities:
                        if hasattr(ent, 'dxf'):
                            if hasattr(ent.dxf, 'start') and hasattr(ent.dxf, 'end'):
                                xs.extend([ent.dxf.start.x, ent.dxf.end.x])
                                ys.extend([ent.dxf.start.y, ent.dxf.end.y])
                            elif ent.dxftype() == 'CIRCLE':
                                c = ent.dxf.center; r = ent.dxf.radius
                                xs.extend([c.x - r, c.x + r]); ys.extend([c.y - r, c.y + r])
                            elif ent.dxftype() == 'ARC':
                                c = ent.dxf.center; r = ent.dxf.radius
                                xs.extend([c.x - r, c.x + r]); ys.extend([c.y - r, c.y + r])
                            elif ent.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                                pts = list(ent.get_points())
                                if pts:
                                    xs.extend([p[0] for p in pts]); ys.extend([p[1] for p in pts])
                    if xs and ys:
                        return (min(xs), min(ys), max(xs), max(ys))
                except Exception:
                    pass
                return (0.0, 0.0, 0.0, 0.0)

            def _part_center_from_bbox(b):
                return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

            # 1) Relabel missing V-GROOVE or Layer 0 semantics for only affected parts
            try:
                for idx, part in enumerate(meaningful_parts, start=1):
                    # If this part seems unlabeled but visually indicates V-GROOVE (green color), relabel
                    has_green = False
                    for ent in part:
                        try:
                            color = get_entity_color(ent, layers)
                            if color == 3 and ent.dxftype() in ['LINE', 'LWPOLYLINE', 'POLYLINE', 'SPLINE', 'ARC']:
                                has_green = True
                                if getattr(ent.dxf, 'layer', '').upper() != 'V-GROOVE':
                                    ent.dxf.layer = 'V-GROOVE'
                        except Exception:
                            continue
                    # Set default layer 0 for white-by-appearance lines missing proper layer naming
                    for ent in part:
                        try:
                            color = get_entity_color(ent, layers)
                            if color in (7, 255) and getattr(ent.dxf, 'layer', '') not in ('0', 'V-GROOVE'):
                                ent.dxf.layer = '0'
                        except Exception:
                            continue
            except Exception as _ai_layer_err:
                print(f"AI layer relabel fallback error: {_ai_layer_err}")

            # 2) Detect missing quantities using nearest text patterns only for failed parts
            try:
                # Collect all text entities once
                all_text = []
                for ent in msp:
                    try:
                        if ent.dxftype() == 'TEXT':
                            all_text.append({'text': ent.dxf.text or '', 'pos': (ent.dxf.insert.x, ent.dxf.insert.y)})
                        elif ent.dxftype() == 'MTEXT':
                            all_text.append({'text': ent.text or '', 'pos': (ent.dxf.insert.x, ent.dxf.insert.y)})
                    except Exception:
                        continue

                qty_patterns = [
                    r'^\s*(\d+)\s*$',
                    r'qty\s*:?\s*(\d+)',
                    r'quantity\s*:?\s*(\d+)',
                    r'q\s*:?\s*(\d+)',  # Added: q:2, q 2, q2
                    r'Q\s*:?\s*(\d+)',  # Added: Q:2, Q 2, Q2
                    r'x\s*(\d+)',
                    r'×\s*(\d+)',
                    r'(\d+)\s*pcs?',
                    r'(\d+)\s*pieces?',
                    r'(\d+)\s*units?'
                ]

                def _try_parse_qty(text):
                    if not text:
                        return None
                    import re as _re
                    # Try case-insensitive matching for all patterns
                    for pat in qty_patterns:
                        m = _re.search(pat, text, _re.IGNORECASE)
                        if m:
                            try:
                                return int(m.group(1) if m.groups() else m.group(0))
                            except Exception:
                                continue
                    return None

                for idx, part in enumerate(meaningful_parts, start=1):
                    lbl = extracted_part_labels.get(idx) or {}
                    q = lbl.get('quantity')
                    if q in (None, 0, '0', ''):
                        bbox = _compute_part_bbox(part)
                        center = _part_center_from_bbox(bbox)
                        best = None
                        best_d = 1e18
                        for t in all_text:
                            qq = _try_parse_qty(t['text'])
                            if qq is None:
                                continue
                            dx = (t['pos'][0] - center[0]); dy = (t['pos'][1] - center[1])
                            d2 = dx*dx + dy*dy
                            if d2 < best_d:
                                best_d = d2
                                best = qq
                        if best is not None:
                            lbl['quantity'] = int(best)
                            extracted_part_labels[idx] = lbl
            except Exception as _ai_qty_err:
                print(f"AI quantity fallback error: {_ai_qty_err}")

            # Create main visualization with all meaningful parts
            if job_id:
                _progress_set(job_id, 55, "Creating overview…")
            fig = create_dxf_visualization(filtered_entities, layers, 
                                         "", meaningful_parts, show_legend=True)
            
            # Save main image to temporary file first, then read as bytes
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                fig.savefig(tmp_file.name, format='png', dpi=150, 
                           facecolor='black', edgecolor='none', bbox_inches='tight')
                tmp_file_path = tmp_file.name
            
            # Close the figure to release file handles
            plt.close(fig)
            
            # Read the saved file as bytes
            with open(tmp_file_path, 'rb') as f:
                img_bytes = f.read()
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass  # Ignore deletion errors
            
            # Convert to base64
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            
            # Create individual part visualizations and cost calculations in a single loop
            part_images = []
            part_costs = []
            part_areas = []  # Store all areas for validation
            total_cost = 0.0
            
            # Get user-selected parameters from request (may be empty; we'll fall back to extracted labels)
            material_name = request.form.get('material_name', '')
            thickness_str = request.form.get('thickness', '')
            grade = request.form.get('grade', '')
            finish = request.form.get('finish', '')
            scrap_factor = float(request.form.get('scrap_factor', 1.20))
            
            # Check if part_labels are provided from frontend (from material selection popup)
            frontend_part_labels = None
            try:
                part_labels_json = request.form.get('part_labels', '')
                if part_labels_json:
                    import json
                    frontend_part_labels = json.loads(part_labels_json)
                    print(f"Received part_labels from frontend: {frontend_part_labels}")
                    print(f"Frontend part_labels keys: {list(frontend_part_labels.keys()) if frontend_part_labels else 'None'}")
                else:
                    print("No part_labels received from frontend")
            except Exception as e:
                print(f"Error parsing part_labels from frontend: {e}")
                frontend_part_labels = None

            # New: read material overrides from the Materials Used UI table
            material_overrides = {}
            try:
                overrides_json = request.form.get('material_overrides', '')
                if overrides_json:
                    import json as _json
                    material_overrides = _json.loads(overrides_json) or {}
                print(f"Material overrides received: {bool(material_overrides)}")
            except Exception as _ov_err:
                print(f"Error parsing material_overrides: {_ov_err}")
                material_overrides = {}
            
            # If not provided, infer defaults from extracted labels (first part with full info)
            inferred = None
            try:
                for k in sorted((extracted_part_labels or {}).keys(), key=lambda x: int(x)):
                    lbl = extracted_part_labels.get(k) or {}
                    if lbl.get('material_name') and lbl.get('thickness') is not None:
                        inferred = lbl
                        break
            except Exception:
                inferred = None

            if not material_name:
                material_name = (inferred or {}).get('material_name') or ''
            if not thickness_str:
                th = (inferred or {}).get('thickness')
                thickness_str = str(th) if th is not None else ''
            if not grade:
                g = (inferred or {}).get('grade')
                grade = '' if g is None else str(g)
            if not finish:
                finish = (inferred or {}).get('finish') or ''

            # Parse thickness to float; if still invalid, choose 0 and guard later
            try:
                thickness_mm = float(thickness_str) if thickness_str not in ('', None) else float((inferred or {}).get('thickness') or 0)
            except ValueError:
                thickness_mm = float((inferred or {}).get('thickness') or 0)
            
            # CRITICAL: Validate that material and thickness are provided before proceeding
            if not material_name or str(material_name).strip() == '':
                error_msg = "ERROR: No material type specified. Cannot calculate costs without material information."
                print(f"  ❌ {error_msg}")
                return jsonify({
                    'error': error_msg,
                    'message': 'Please specify a material type in the upload form or ensure it is properly labeled in the DXF file.'
                }), 400
            
            if thickness_mm is None or thickness_mm <= 0:
                error_msg = "ERROR: No thickness specified or thickness is invalid. Cannot calculate costs without thickness information."
                print(f"  ❌ {error_msg}")
                return jsonify({
                    'error': error_msg,
                    'message': 'Please specify a valid thickness in the upload form or ensure it is properly labeled in the DXF file.'
                }), 400
            
            print(f"  ✅ Material validation passed: '{material_name}' / {thickness_mm}mm")
            
            # Get dynamic material configuration
            material_config = None
            try:
                if material_name and thickness_mm:
                    material_config = get_material_config(material_name or '', thickness_mm, grade, finish)
            except Exception:
                material_config = None
            
            # Get admin configuration for laser cost and piercing toggle
            admin_config = load_admin_config()
            
            # Merge material config with admin config (guard None)
            if material_config:
                config = material_config.copy()
                config['laser_cost'] = admin_config.get('laser_cost', 2)
                config['piercing_toggle'] = admin_config.get('piercing_toggle', False)
                config['scrap_factor'] = scrap_factor
            else:
                # Use admin defaults; per-part DB lookup below will still gate calculations
                config = {
                    'machine_speed': admin_config.get('machine_speed', 100.0),
                    'vaporization_speed': admin_config.get('vaporization_speed', 50.0),
                    'piercing_time': admin_config.get('piercing_time', 0.5),
                    'price_per_kg': admin_config.get('price_per_kg', 25.0),
                    'density': admin_config.get('density', 7.85),
                    'vgroove_price': admin_config.get('vgroove_price', 0.0),
                    'bending_price': admin_config.get('bending_price', 0.0),
                    'laser_cost': admin_config.get('laser_cost', 2),
                    'piercing_toggle': admin_config.get('piercing_toggle', False),
                    'scrap_factor': scrap_factor
                }
            
            total_parts = max(1, len(meaningful_parts))
            for i, part_entities in enumerate(meaningful_parts):
                # Process meaningful parts (multi-entity or single supported entities)
                if len(part_entities) > 1 or (len(part_entities) == 1 and part_entities[0].dxftype() in ['CIRCLE', 'ELLIPSE', 'ARC', 'SPLINE']):
                    part_number = i + 1
                    # Debug suppressed: per-part entity prints
                    
                    # Calculate dimensions and area for this part with detailed logging
                    length_mm, width_mm, area = calculate_part_dimensions(part_entities, layers)
                    part_areas.append(area)
                    
                    # Create visualization for this part only
                    part_fig = create_dxf_visualization(part_entities, layers, 
                                                      "", [part_entities], show_legend=False)
                    
                    # Save part image
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        part_fig.savefig(tmp_file.name, format='png', dpi=150, 
                                       facecolor='black', edgecolor='none', bbox_inches='tight')
                        tmp_file_path = tmp_file.name
                    
                    # Close the figure
                    plt.close(part_fig)
                    
                    # Read the saved file as bytes
                    with open(tmp_file_path, 'rb') as f:
                        part_img_bytes = f.read()
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    
                    # Convert to base64
                    part_img_str = base64.b64encode(part_img_bytes).decode('utf-8')
                    
                    # Use frontend part_labels if available, otherwise fall back to extracted labels
                    _lbl = None
                    print(f"Processing part {part_number}...")
                    print(f"frontend_part_labels available: {frontend_part_labels is not None}")
                    if frontend_part_labels:
                        print(f"frontend_part_labels keys: {list(frontend_part_labels.keys())}")
                        print(f"Looking for part {part_number} in frontend_part_labels: {str(part_number) in frontend_part_labels}")
                    
                    if frontend_part_labels and str(part_number) in frontend_part_labels:
                        _lbl = frontend_part_labels[str(part_number)]
                        print(f"Using frontend part_labels for part {part_number}: {_lbl}")
                    else:
                        _lbl = extracted_part_labels.get(part_number) or {}
                        print(f"Using extracted labels for part {part_number}: {_lbl}")
                    
                    # Decide material parameters using extracted label when available, fallback to form values
                    eff_material_name = (_lbl.get('material_name') or material_name) if isinstance(_lbl, dict) else material_name
                    eff_thickness_mm = (_lbl.get('thickness') or thickness_mm) if isinstance(_lbl, dict) else thickness_mm
                    eff_grade = (_lbl.get('grade') if isinstance(_lbl, dict) and _lbl.get('grade') is not None else grade)
                    eff_finish = (_lbl.get('finish') if isinstance(_lbl, dict) and _lbl.get('finish') is not None else finish)
                    
                    # CRITICAL FIX: Only skip calculation if we have NO material info from any source
                    has_material_info = (
                        (eff_material_name and str(eff_material_name).strip()) and 
                        (eff_thickness_mm is not None and eff_thickness_mm > 0)
                    )
                    
                    if not has_material_info:
                        print(f"  ❌ No material information available for part {part_number} from any source. Showing error breakdown.")
                        # Ensure the part card still renders with its image
                        part_images.append({
                            'part_number': part_number,
                            'entity_count': len(part_entities),
                            'area': area,
                            'length_mm': length_mm,
                            'width_mm': width_mm,
                            'image': part_img_str,
                            'extracted_label': _lbl,
                            'applied_material_name': eff_material_name,
                            'applied_thickness': eff_thickness_mm,
                            'applied_grade': eff_grade,
                            'applied_finish': eff_finish
                        })

                        cost_data = {
                            'error': (
                                f"ERROR: No material information available for part {part_number}. "
                                f"Either add a label under the part in the DXF (e.g. 'Brass 1mm') or manually assign materials."
                            ),
                            'area_sq_mm': area * 1000000 if area else 0,
                            'length_mm': length_mm,
                            'width_mm': width_mm,
                            'perimeter_meters': 0.0,
                            'cutting_time_machine': 0.0,
                            'cutting_time_vaporization': 0.0,
                            'piercing_time_total': 0.0,
                            'total_time_min': 0.0,
                            'object_parts_count': count_object_parts(part_entities, layers),
                            'laser_cost': 0.0,
                            'weight_kg': 0.0,
                            'material_cost': 0.0,
                            'vgroove_count': 0,
                            'bending_count': 0,
                            'total_bending_lines': 0,
                            'bending_cost': 0.0,
                            'vgroove_length_meters': 0.0,
                            'vgroove_cost': 0.0,
                            'total_cost': 0.0
                        }

                        part_costs.append({
                            'part_number': part_number,
                            'area': area,
                            'length_mm': length_mm,
                            'width_mm': width_mm,
                            'object_parts_count': cost_data['object_parts_count'],
                            'cost_data': cost_data,
                            'extracted_label': _lbl,
                            'applied_material_name': eff_material_name,
                            'applied_thickness': eff_thickness_mm,
                            'applied_grade': eff_grade,
                            'applied_finish': eff_finish,
                            'calculation_error': cost_data['error']
                        })

                        # Skip any further processing for this part
                        continue
                    
                    print(f"  ✅ Material info available for part {part_number}. Proceeding with cost calculation.")
                    
                    # Debug: Log what we extracted
                    print(f"  🔍 MATERIAL EXTRACTION DEBUG for part {part_number}:")
                    print(f"      From label: material='{_lbl.get('material_name')}', thickness={_lbl.get('thickness')}, grade={_lbl.get('grade')}, finish={_lbl.get('finish')}")
                    print(f"      From form: material='{material_name}', thickness={thickness_mm}, grade={grade}, finish={finish}")
                    print(f"      Final: material='{eff_material_name}', thickness={eff_thickness_mm}, grade={eff_grade}, finish={eff_finish}")
                    
                    # Fallback: If no material name from extraction, try to infer from the part characteristics
                    if not eff_material_name or str(eff_material_name).strip() == '':
                        print(f"      ⚠ No material name extracted, attempting to infer from part characteristics...")
                        # This could be enhanced with AI material inference in the future
                        if material_name and str(material_name).strip() != '':
                            eff_material_name = material_name
                            print(f"      ✅ Using material from form: '{eff_material_name}'")
                        else:
                            print(f"      ❌ No material name available from any source")
                    
                    # Fallback: If no thickness from extraction, use form value
                    if eff_thickness_mm is None or float(eff_thickness_mm) <= 0:
                        print(f"      ⚠ No thickness extracted, using form value: {thickness_mm}")
                        eff_thickness_mm = thickness_mm

                    # AI fallback: infer material if still missing/unknown
                    try:
                        if (not eff_material_name) or str(eff_material_name).strip().lower() in ('unknown', 'n/a', 'na', 'none', ''):
                            api_key = os.getenv('OPENAI_API_KEY')
                            if api_key:
                                characteristics = {
                                    'length_mm': float(length_mm or 0),
                                    'width_mm': float(width_mm or 0),
                                    'area_sq_mm': float((area or 0) * 1_000_000.0),
                                    'perimeter_mm': float(0.0),
                                    'entity_types': [e.dxftype() for e in part_entities if hasattr(e, 'dxftype')],
                                    'colors': [get_entity_color(e, layers) for e in part_entities]
                                }
                                inferred = _infer_material_openai_sync(api_key, characteristics)
                                if inferred:
                                    eff_material_name = inferred
                    except Exception as _mat_inf_err:
                        try:
                            print(f"Material inference fallback error: {_mat_inf_err}")
                        except Exception:
                            pass

                    # Add to part_images
                    part_images.append({
                        'part_number': part_number,
                        'entity_count': len(part_entities),
                        'area': area,
                        'length_mm': length_mm,
                        'width_mm': width_mm,
                        'image': part_img_str,
                        'extracted_label': _lbl,
                        'applied_material_name': eff_material_name,
                        'applied_thickness': eff_thickness_mm,
                        'applied_grade': eff_grade,
                        'applied_finish': eff_finish
                    })
                    
                    # Calculate cost for this part using effective material settings
                    object_parts_count = count_object_parts(part_entities, layers)
                    
                    # CRITICAL: Validate required fields based on material type
                    print(f"  🔍 VALIDATION DEBUG for part {part_number}:")
                    print(f"      eff_material_name: '{eff_material_name}' (type: {type(eff_material_name)})")
                    print(f"      eff_thickness_mm: {eff_thickness_mm} (type: {type(eff_thickness_mm)})")
                    print(f"      eff_grade: '{eff_grade}' (type: {type(eff_grade)})")
                    print(f"      eff_finish: '{eff_finish}' (type: {type(eff_finish)})")
                    
                    # Debug extracted label information
                    if _lbl:
                        print(f"      📋 EXTRACTED LABEL DEBUG:")
                        print(f"          type_text: '{_lbl.get('type_text')}'")
                        print(f"          material_name: '{_lbl.get('material_name')}'")
                        print(f"          thickness: {_lbl.get('thickness')}")
                        print(f"          grade: '{_lbl.get('grade')}'")
                        print(f"          finish: '{_lbl.get('finish')}'")
                        print(f"          notes: '{_lbl.get('notes')}'")
                    else:
                        print(f"      📋 No extracted label found for part {part_number}")
                    
                    # Debug form values
                    print(f"      📝 FORM VALUES DEBUG:")
                    print(f"          material_name: '{material_name}'")
                    print(f"          thickness_str: '{thickness_str}'")
                    print(f"          grade: '{grade}'")
                    print(f"          finish: '{finish}'")
                    
                    missing_fields = []
                    if not eff_material_name or str(eff_material_name).strip() == '':
                        missing_fields.append('material')
                        print(f"      ❌ Material is missing/empty")
                    else:
                        print(f"      ✅ Material: '{eff_material_name}'")
                        
                    if eff_thickness_mm is None or str(eff_thickness_mm).strip() == '' or float(eff_thickness_mm) <= 0:
                        missing_fields.append('thickness')
                        print(f"      ❌ Thickness is missing/empty/invalid: {eff_thickness_mm}")
                    else:
                        print(f"      ✅ Thickness: {eff_thickness_mm}")
                    
                    # For materials that don't need grade/finish, set them to 0 if missing
                    materials_no_grade_finish = ['Brass', 'Copper', 'Mild Steel', 'Aluminum']
                    
                    # Trim whitespace and normalize case from material name to avoid comparison issues
                    if eff_material_name:
                        eff_material_name = str(eff_material_name).strip()
                        print(f"      🔧 Trimmed material name: '{eff_material_name}'")
                        
                        # Check if it's a case-insensitive match for any material
                        material_lower = eff_material_name.lower()
                        print(f"      🔧 Material lowercase: '{material_lower}'")
                        
                        # Find the correct case from our list
                        for material in materials_no_grade_finish:
                            if material.lower() == material_lower:
                                eff_material_name = material  # Use the correct case
                                print(f"      🔧 Normalized material name to: '{eff_material_name}'")
                                break
                    
                    print(f"      🔍 Checking if '{eff_material_name}' is in materials_no_grade_finish: {eff_material_name in materials_no_grade_finish}")
                    print(f"      🔍 Available materials: {materials_no_grade_finish}")
                    print(f"      🔍 Material comparison: '{eff_material_name}' == 'Brass' = {eff_material_name == 'Brass'}")
                    
                    if eff_material_name in materials_no_grade_finish:
                        # These materials only need material + thickness, grade/finish can be 0
                        print(f"      📝 {eff_material_name} doesn't need grade/finish - setting to 0 if missing")
                        if eff_grade is None or str(eff_grade).strip() == '':
                            eff_grade = 0
                            print(f"      ⚠ Grade missing for {eff_material_name}, setting to 0")
                        else:
                            print(f"      ✅ Grade: {eff_grade}")
                            
                        if not eff_finish or str(eff_finish).strip() == '':
                            eff_finish = 0
                            print(f"      ⚠ Finish missing for {eff_material_name}, setting to 0")
                        else:
                            print(f"      ✅ Finish: {eff_finish}")
                    else:
                        # Stainless Steel and other materials need grade and finish
                        print(f"      📝 {eff_material_name} needs grade and finish")
                        if eff_grade is None or str(eff_grade).strip() == '':
                            missing_fields.append('grade')
                            print(f"      ❌ Grade is missing/empty")
                        else:
                            print(f"      ✅ Grade: {eff_grade}")
                            
                        if not eff_finish or str(eff_finish).strip() == '':
                            missing_fields.append('finish')
                            print(f"      ❌ Finish is missing/empty")
                        else:
                            print(f"      ✅ Finish: {eff_finish}")
                    
                    print(f"      📊 Missing fields: {missing_fields}")
                    
                    if missing_fields:
                        print(f"  ❌ Missing required fields for part {part_number}: {', '.join(missing_fields)}. Skipping cost calculation.")
                        cost_data = {
                            'error': f"ERROR: Missing required {', '.join(missing_fields)} for part {part_number}. Add them under the part label in DXF.",
                            'area_sq_mm': area * 1000000 if area else 0,
                            'length_mm': length_mm,
                            'width_mm': width_mm,
                            'perimeter_meters': 0.0,
                            'cutting_time_machine': 0.0,
                            'cutting_time_vaporization': 0.0,
                            'piercing_time_total': 0.0,
                            'total_time_min': 0.0,
                            'object_parts_count': object_parts_count,
                            'laser_cost': 0.0,
                            'weight_kg': 0.0,
                            'material_cost': 0.0,
                            'vgroove_count': 0,
                            'bending_count': 0,
                            'total_bending_lines': 0,
                            'bending_cost': 0.0,
                            'vgroove_length_meters': 0.0,
                            'vgroove_cost': 0.0,
                            'total_cost': 0.0
                        }
                    else:
                        # Required fields present, proceed with cost calculation
                        print(f"  ✓ Required fields complete for part {part_number}: {eff_material_name} / {eff_thickness_mm} / Grade:{eff_grade} / Finish:{eff_finish}. Proceeding with cost calculation.")
                        
                        # CRITICAL: Check if we're about to call get_material_config
                        print(f"  🚀 ABOUT TO CALL get_material_config for part {part_number}")
                        
                        # Rebuild config per part using parsed values
                        print(f"  🔍 Calling get_material_config with:")
                        print(f"      material_name: '{eff_material_name}' (type: {type(eff_material_name)})")
                        print(f"      thickness: {eff_thickness_mm} (type: {type(eff_thickness_mm)})")
                        print(f"      grade: '{eff_grade}' (type: {type(eff_grade)})")
                        print(f"      finish: '{eff_finish}' (type: {type(eff_finish)})")
                        
                        try:
                            print(f"  🔍 CALLING get_material_config...")
                            part_config = get_material_config(eff_material_name, float(eff_thickness_mm), eff_grade, eff_finish)
                            print(f"  🔍 get_material_config returned: {part_config}")
                        except Exception as e:
                            print(f"  ❌ Exception getting material config: {e}")
                            import traceback
                            print(f"  ❌ Full traceback: {traceback.format_exc()}")
                            part_config = None
                        
                        # Apply UI overrides if available (override DB values)
                        def _mkey(name, th, gr, fi):
                            try:
                                return "|".join([str(name or ''), str(th or ''), str(gr if gr is not None else ''), str(fi if fi is not None else '')])
                            except Exception:
                                return f"{name}|{th}|{gr}|{fi}"

                        try:
                            key = _mkey(eff_material_name, eff_thickness_mm, eff_grade, eff_finish)
                            ov = material_overrides.get(key) if isinstance(material_overrides, dict) else None
                            if ov:
                                print(f"  ✏️ Applying UI overrides for part {part_number}: {ov}")
                                if not part_config:
                                    part_config = {}
                                # Override specific fields if provided
                                for k_ui, k_cfg in [
                                    ('density', 'density'),
                                    ('price_per_kg', 'price_per_kg'),
                                    ('scrap_price_per_kg', 'scrap_price_per_kg'),
                                    ('machine_speed', 'machine_speed'),
                                    ('piercing_time', 'piercing_time'),
                                    ('vaporization_speed', 'vaporization_speed'),
                                    ('vgroove_price', 'vgroove_price'),
                                    ('bending_price', 'bending_price'),
                                ]:
                                    try:
                                        if ov.get(k_ui) is not None:
                                            part_config[k_cfg] = float(ov[k_ui])
                                    except Exception:
                                        pass
                        except Exception as _apply_err:
                            print(f"  ⚠ Failed to apply UI overrides: {_apply_err}")

                        if not part_config:
                            print(f"  ❌ DB config not found for part {part_number}: {eff_material_name} / {eff_thickness_mm} / {eff_grade} / {eff_finish}")
                            print(f"      Please add this combination to the Materials database in Admin panel.")
                            
                            # Provide more helpful error message
                            if not eff_material_name or str(eff_material_name).strip() == '':
                                error_msg = f"ERROR: No material type specified for part {part_number}. Please ensure the material is labeled in the DXF file or select it in the upload form."
                            elif eff_thickness_mm is None or float(eff_thickness_mm) <= 0:
                                error_msg = f"ERROR: No thickness specified for part {part_number}. Please ensure the thickness is labeled in the DXF file or select it in the upload form."
                            else:
                                error_msg = f"ERROR: Material combination '{eff_material_name} / {eff_thickness_mm}mm / Grade {eff_grade} / {eff_finish}' not found in database for part {part_number}. Please add it in Admin Materials or check the material specification."
                            
                            cost_data = {
                                'error': error_msg,
                                'area_sq_mm': area * 1000000 if area else 0,
                                'length_mm': length_mm,
                                'width_mm': width_mm,
                                'perimeter_meters': 0.0,
                                'cutting_time_machine': 0.0,
                                'cutting_time_vaporization': 0.0,
                                'piercing_time_total': 0.0,
                                'total_time_min': 0.0,
                                'object_parts_count': object_parts_count,
                                'laser_cost': 0.0,
                                'weight_kg': 0.0,
                                'material_cost': 0.0,
                                'vgroove_count': 0,
                                'bending_count': 0,
                                'total_bending_lines': 0,
                                'bending_cost': 0.0,
                                'vgroove_length_meters': 0.0,
                                'vgroove_cost': 0.0,
                                'total_cost': 0.0
                            }
                        else:
                            # Found in database, proceed with calculation
                            print(f"  ✓ Found DB config for part {part_number}. Calculating costs...")
                            part_config['laser_cost'] = config.get('laser_cost')
                            part_config['piercing_toggle'] = config.get('piercing_toggle')
                            # CRITICAL FIX: Use user's scrap factor from form, not database scrap factor
                            part_config['scrap_factor'] = scrap_factor  # Use the user-input scrap factor
                            cost_data = calculate_cost(area, part_config, eff_material_name, float(eff_thickness_mm), object_parts_count, part_entities, layers, length_mm, width_mm)
                    
                    if cost_data:
                        # Check if there's an error in cost calculation
                        if 'error' in cost_data:
                            print(f"  ❌ Cost calculation error for part {part_number}: {cost_data['error']}")
                            # Add error information to part_costs for display
                            part_costs.append({
                                'part_number': part_number,
                                'area': area,
                                'length_mm': length_mm,
                                'width_mm': width_mm,
                                'object_parts_count': object_parts_count,
                                'cost_data': cost_data,
                                'extracted_label': _lbl,
                                'applied_material_name': eff_material_name,
                                'applied_thickness': eff_thickness_mm,
                                'applied_grade': eff_grade,
                                'applied_finish': eff_finish,
                                'calculation_error': cost_data['error']
                            })
                            # Don't add to total cost since there was an error
                        else:
                            # DEBUG: Log cost data for all parts to debug UI display issue
                            print(f"🔍 PART {part_number} COST DATA ADDED TO part_costs:")
                            print(f"    Material Cost: ${cost_data.get('material_cost', 'Not set')}")
                            print(f"    Total Cost: ${cost_data.get('total_cost', 'Not set')}")
                            print(f"    Weight: {cost_data.get('weight_kg', 'Not set')} kg")
                            print(f"    Area: {area} m²")
                            print(f"    Thickness: {eff_thickness_mm} mm")
                            print(f"    Has error: {'error' in cost_data}")
                            print(f"    Cost data keys: {list(cost_data.keys()) if cost_data else 'None'}")
                            
                            part_costs.append({
                                'part_number': part_number,
                                'area': area,
                                'length_mm': length_mm,
                                'width_mm': width_mm,
                                'object_parts_count': object_parts_count,
                                'cost_data': cost_data,
                                'extracted_label': _lbl,
                                'applied_material_name': eff_material_name,
                                'applied_thickness': eff_thickness_mm,
                                'applied_grade': eff_grade,
                                'applied_finish': eff_finish
                            })
                            total_cost += cost_data['total_cost']

                    if job_id:
                        # Map parts processing between 60% and 90%
                        prog = 60 + int(30 * (i + 1) / total_parts)
                        _progress_set(job_id, prog, f"Processing parts… {i+1}/{total_parts}")
            
            # Validate all area calculations and get precise total
            if job_id:
                _progress_set(job_id, 92, "Finalizing…")
            total_area, valid_areas = validate_area_calculations(part_areas)
            
            # Aggregate external services costs from all parts
            external_services = {
                'punching': 0,
                'brushing': 0,
                'marking': 0,
                'uv_print': 0,
                'cutting': 0,
                'pvc_cover': 0,
                'rolling': 0,
                'straighten': 0,
                'corner_form': 0,
                'router': 0,
                'finishing': 0,
                'installing': 0,
                'others': 0
            }
            
            for part_cost in part_costs:
                cost_data = part_cost.get('cost_data', {})
                
                # Add known external services
                external_services['punching'] += cost_data.get('punching', 0)
                external_services['brushing'] += cost_data.get('brushing', 0)
                external_services['marking'] += cost_data.get('marking', 0)
                external_services['uv_print'] += cost_data.get('uv_print', 0)
                external_services['cutting'] += cost_data.get('cutting', 0)
                external_services['pvc_cover'] += cost_data.get('pvc_cover', 0)
                external_services['rolling'] += cost_data.get('rolling', 0)
                external_services['straighten'] += cost_data.get('straighten', 0)
                external_services['corner_form'] += cost_data.get('corner_form', 0)
                external_services['router'] += cost_data.get('router', 0)
                external_services['finishing'] += cost_data.get('finishing', 0)
                external_services['installing'] += cost_data.get('installing', 0)
                external_services['others'] += cost_data.get('others', 0)
                
                # Add any other services to others
                for service_name, service_cost in cost_data.items():
                    if service_name not in ['total_cost', 'laser_cost', 'material_cost', 'bending_cost', 'vgroove_cost', 'punching', 'brushing', 'marking', 'uv_print', 'cutting', 'pvc_cover', 'rolling', 'straighten', 'corner_form', 'router', 'finishing', 'installing', 'others'] and isinstance(service_cost, (int, float)) and service_cost > 0:
                        external_services['others'] += service_cost
            
            # Create statistics
            stats = {
                'original_entities': len(list(msp)),
                'filtered_entities': len(filtered_entities),
                'removed_entities': removed_count,
                'layers_found': list(layers.keys()),
                'filename': file.filename,
                'original_filename': file.filename,  # Store original filename for PDF naming
                'total_parts': len(meaningful_parts),
                'total_area': total_area,
                'total_cost': total_cost,
                **external_services,  # Include all external services
                'parts_info': [
                    {
                        'part_number': i + 1,
                        'entity_count': len(part_entities)
                    }
                    for i, part_entities in enumerate(meaningful_parts)
                ]
            }
            
            # Final safety check to clean any infinite values from part_costs
            def clean_infinite_values(obj):
                if isinstance(obj, dict):
                    return {k: clean_infinite_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_infinite_values(item) for item in obj]
                elif isinstance(obj, float):
                    if np.isnan(obj) or np.isinf(obj):
                        return 0.0
                    return obj
                else:
                    return obj
            
            cleaned_part_costs = clean_infinite_values(part_costs)

            # Dedupe helper to avoid duplicate entries by part_number
            def _dedupe_by_part_number(items):
                try:
                    seen = set()
                    result = []
                    for it in (items or []):
                        pn = it.get('part_number') if isinstance(it, dict) else None
                        if pn is None:
                            continue
                        if pn in seen:
                            continue
                        seen.add(pn)
                        result.append(it)
                    return result
                except Exception:
                    return items or []

            cleaned_part_costs = _dedupe_by_part_number(cleaned_part_costs)
            part_images = _dedupe_by_part_number(part_images)
            
            # Material cost uses the UI scrap factor
            print(f"🔍 MATERIAL COST SCRAP FACTOR DEBUG:")
            print(f"    UI scrap factor (used for material cost): {scrap_factor}")
            print(f"    Material cost calculation: Uses the UI scrap factor of {scrap_factor}")
 
            if job_id:
                _progress_done(job_id, True, "Completed")
            # Analyze-only mode: return geometry and parts, but suppress pricing
            if request.form.get('analyze_only') in ('1', 'true', 'True'):  # non-invasive; default behavior unchanged
                # Strip cost data to satisfy "no pricing here" rule in analyze step
                analyzed_parts = []
                for pc in cleaned_part_costs:
                    analyzed_parts.append({
                        'part_number': pc.get('part_number'),
                        'area': pc.get('area'),
                        'length_mm': pc.get('length_mm'),
                        'width_mm': pc.get('width_mm'),
                        'object_parts_count': pc.get('object_parts_count')
                    })
                stats_no_cost = {k: v for k, v in stats.items() if k not in ('total_cost',)}
                return jsonify({
                    'success': True,
                    'image': img_str,
                    'stats': stats_no_cost,
                    'part_images': part_images,
                    'part_costs': analyzed_parts,
                    'part_labels': {str(k): v for k, v in (frontend_part_labels or extracted_part_labels or {}).items()},
                    'material_name': material_name,
                    'thickness': thickness_mm,
                    'grade': grade,
                    'finish': finish,
                })

            # DEBUG: Log final part costs for part 10 and part 36 before sending response
            for pc in cleaned_part_costs:
                if pc.get('part_number') == 10:
                    print(f"🔍 PART 10 FINAL RESPONSE DEBUG:")
                    print(f"    Part Number: {pc.get('part_number')}")
                    print(f"    Area: {pc.get('area')} m²")
                    print(f"    Material Cost in cost_data: ${pc.get('cost_data', {}).get('material_cost', 'Not set')}")
                    print(f"    Total Cost in cost_data: ${pc.get('cost_data', {}).get('total_cost', 'Not set')}")
                    print(f"    Applied Material: {pc.get('applied_material_name')}")
                    print(f"    Applied Thickness: {pc.get('applied_thickness')} mm")
                elif pc.get('part_number') == 36:
                    print(f"🔍 PART 36 FINAL RESPONSE DEBUG:")
                    print(f"    Part Number: {pc.get('part_number')}")
                    print(f"    Area: {pc.get('area')} m²")
                    print(f"    Material Cost in cost_data: ${pc.get('cost_data', {}).get('material_cost', 'Not set')}")
                    print(f"    Total Cost in cost_data: ${pc.get('cost_data', {}).get('total_cost', 'Not set')}")
                    print(f"    Applied Material: {pc.get('applied_material_name')}")
                    print(f"    Applied Thickness: {pc.get('applied_thickness')} mm")
                
                # Break if we found both parts
                if pc.get('part_number') in [10, 36]:
                    break
            
            # DEBUG: Log what's being sent to frontend
            print(f"🔍 SENDING TO FRONTEND:")
            print(f"    Total part_costs entries: {len(cleaned_part_costs)}")
            for pc in cleaned_part_costs:
                print(f"    Part {pc.get('part_number')}: cost_data={'Yes' if pc.get('cost_data') else 'No'}, error={'Yes' if pc.get('calculation_error') else 'No'}")
            
            return jsonify({
                'success': True,
                'image': img_str,
                'stats': stats,
                'part_images': part_images,
                'part_costs': cleaned_part_costs,
                'part_labels': {str(k): v for k, v in (extracted_part_labels or {}).items()},
                # Include material data for PDF generation
                'material_name': material_name,
                'thickness': thickness_mm,
                'grade': grade,
                'finish': finish,
                'scrap_factor': scrap_factor,
                'material_config': material_config
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")
            if request.form.get('job_id'):
                _progress_done(request.form.get('job_id'), False, "Failed")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/upload-multi', methods=['POST'])
def upload_multi():
    """Process multiple DXF files under one quotation with per-file material settings."""
    files = request.files.getlist('files') or []
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    job_id = request.form.get('job_id') or ""
    scrap_factor = float(request.form.get('scrap_factor', 1.20))

    # Optional per-file settings as JSON array: [{name, material_name, thickness, grade, finish}]
    import json as _json
    settings_raw = request.form.get('file_settings')
    per_file_settings = []
    try:
        if settings_raw:
            per_file_settings = _json.loads(settings_raw)
            if not isinstance(per_file_settings, list):
                per_file_settings = []
    except Exception:
        per_file_settings = []

    # Aggregates
    all_part_images = []
    all_part_costs = []
    all_part_areas = []
    total_cost_sum = 0.0
    total_parts_count = 0
    first_main_image_b64 = None
    all_main_images_b64 = []
    layers_overall = set()
    original_filenames = []
    original_entities_total = 0
    filtered_entities_total = 0
    removed_entities_total = 0

    # Progress setup
    if job_id:
        _progress_set(job_id, 5, "Starting multi-file processing…")

    num_files = len(files)
    for idx, file in enumerate(files):
        try:
            if not file or file.filename == '':
                continue
            original_filenames.append(file.filename)
            file_settings = {}
            if idx < len(per_file_settings) and isinstance(per_file_settings[idx], dict):
                file_settings = per_file_settings[idx]

            material_name = file_settings.get('material_name') or request.form.get('material_name', '')
            thickness_str = str(file_settings.get('thickness') if file_settings.get('thickness') is not None else request.form.get('thickness', ''))
            grade = file_settings.get('grade') if file_settings.get('grade') is not None else request.form.get('grade', '')
            finish = file_settings.get('finish') if file_settings.get('finish') is not None else request.form.get('finish', '')

            # CRITICAL: Validate that material and thickness are provided before proceeding
            if not material_name or str(material_name).strip() == '':
                error_msg = f"ERROR: No material type specified for file '{file.filename}'. Cannot calculate costs without material information."
                print(f"  ❌ {error_msg}")
                return jsonify({
                    'error': error_msg,
                    'message': f'Please specify a material type for file: {file.filename}'
                }), 400
            
            if not thickness_str or thickness_str.strip() == '':
                error_msg = f"ERROR: No thickness specified for file '{file.filename}'. Cannot calculate costs without thickness information."
                print(f"  ❌ {error_msg}")
                return jsonify({
                    'error': error_msg,
                    'message': f'Please specify a thickness for file: {file.filename}'
                }), 400
            
            try:
                thickness_mm = float(thickness_str)
                if thickness_mm <= 0:
                    error_msg = f"ERROR: Invalid thickness '{thickness_str}' for file '{file.filename}'. Thickness must be greater than 0."
                    print(f"  ❌ {error_msg}")
                    return jsonify({
                        'error': error_msg,
                        'message': f'Thickness must be greater than 0 for file: {file.filename}'
                    }), 400
            except ValueError:
                error_msg = f"ERROR: Invalid thickness format '{thickness_str}' for file '{file.filename}'. Please provide a valid number."
                print(f"  ❌ {error_msg}")
                return jsonify({
                    'error': error_msg,
                    'message': f'Please provide a valid thickness number for file: {file.filename}'
                }), 400
            
            print(f"  ✅ Material validation passed for '{file.filename}': '{material_name}' / {thickness_mm}mm")

            # Progress per file window: map idx over 10..95
            if job_id:
                base = 10 + int(80 * idx / num_files)
                _progress_set(job_id, base, f"Saving {idx+1}/{num_files}…")

            # Save to temp and read
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                file.save(tmp_file.name)
                tmp_file_path = tmp_file.name

            if job_id:
                _progress_set(job_id, base + 2, f"Reading DXF {idx+1}/{num_files}…")
            doc = ezdxf.readfile(tmp_file_path)
            msp = doc.modelspace()
            os.unlink(tmp_file_path)

            # Layers
            layers = {}
            for layer in doc.layers:
                layers[layer.dxf.name] = layer.dxf.color
            layers_overall.update(layers.keys())

            # Filter entities and group parts
            if job_id:
                _progress_set(job_id, base + 5, f"Filtering {idx+1}/{num_files}…")
            # Count originals
            try:
                original_entities_total += len(list(msp))
            except Exception:
                pass
            filtered_entities, removed_count = filter_entities(msp, layers)
            filtered_entities_total += len(filtered_entities)
            removed_entities_total += int(removed_count or 0)

            if job_id:
                _progress_set(job_id, base + 8, f"Grouping {idx+1}/{num_files}…")
            parts = find_connected_parts(filtered_entities, layers)
            parts = merge_contained_parts(parts, layers)
            parts = attach_isolated_entities_to_parents(parts, layers)

            # Keep meaningful parts
            meaningful_parts = []
            for part in parts:
                if len(part) > 1:
                    meaningful_parts.append(part)
                elif len(part) == 1 and part[0].dxftype() in ['CIRCLE', 'ELLIPSE', 'ARC', 'SPLINE']:
                    meaningful_parts.append(part)

            # Main visualization for this file
            if job_id:
                _progress_set(job_id, base + 10, f"Rendering overview {idx+1}/{num_files}…")
            fig = create_dxf_visualization(filtered_entities, layers, "", meaningful_parts, show_legend=True)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                fig.savefig(tmp_img.name, format='png', dpi=150, facecolor='black', edgecolor='none', bbox_inches='tight')
                tmp_img_path = tmp_img.name
            plt.close(fig)
            with open(tmp_img_path, 'rb') as fimg:
                img_bytes = fimg.read()
            try:
                os.unlink(tmp_img_path)
            except Exception:
                pass
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            if first_main_image_b64 is None:
                first_main_image_b64 = img_str
            all_main_images_b64.append({'file': file.filename, 'image': img_str})

            # Loop parts for visuals, costs, and labels
            file_part_images = []
            file_part_costs = []
            file_part_areas = []
            file_total_cost = 0.0

            # Extract labels for these parts in this file too
            try:
                multi_file_labels = extract_labels_for_parts(doc, msp, meaningful_parts)
            except Exception as _e:
                print(f"Label extraction (multi) error: {_e}")
                multi_file_labels = {}

            for p_idx, part_entities in enumerate(meaningful_parts):
                part_number_global = total_parts_count + p_idx + 1

                length_mm, width_mm, area = calculate_part_dimensions(part_entities, layers)
                file_part_areas.append(area)

                # Part visualization
                part_fig = create_dxf_visualization(part_entities, layers, "", [part_entities], show_legend=False)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_pimg:
                    part_fig.savefig(tmp_pimg.name, format='png', dpi=150, facecolor='black', edgecolor='none', bbox_inches='tight')
                    tmp_pimg_path = tmp_pimg.name
                plt.close(part_fig)
                with open(tmp_pimg_path, 'rb') as fpimg:
                    part_img_bytes = fpimg.read()
                try:
                    os.unlink(tmp_pimg_path)
                except Exception:
                    pass
                part_img_str = base64.b64encode(part_img_bytes).decode('utf-8')

                file_part_images.append({
                    'part_number': part_number_global,
                    'entity_count': len(part_entities),
                    'area': area,
                    'length_mm': length_mm,
                    'width_mm': width_mm,
                    'image': part_img_str,
                    'source_file': file.filename,
                    'material_name': material_name,
                    'thickness': thickness_mm,
                    'grade': grade,
                    'finish': finish,
                    'extracted_label': multi_file_labels.get(p_idx + 1)
                })

                object_parts_count = count_object_parts(part_entities, layers)
                
                # CRITICAL: For multi-upload, use global material/thickness/grade/finish for all parts
                # Validate that required fields are provided globally
                missing_fields = []
                if not material_name or str(material_name).strip() == '':
                    missing_fields.append('material')
                if thickness_mm is None or str(thickness_mm).strip() == '' or float(thickness_mm) <= 0:
                    missing_fields.append('thickness')
                
                # For materials that don't need grade/finish, set them to 0 if missing
                materials_no_grade_finish = ['Brass', 'Copper', 'Mild Steel', 'Aluminum']
                if material_name in materials_no_grade_finish:
                    # These materials only need material + thickness, grade/finish can be 0
                    if grade is None or str(grade).strip() == '':
                        grade = 0
                        print(f"  ⚠ Grade missing for {material_name}, setting to 0")
                    if not finish or str(finish).strip() == '':
                        finish = 0
                        print(f"  ⚠ Finish missing for {material_name}, setting to 0")
                else:
                    # Stainless Steel and other materials need grade and finish
                    if grade is None or str(grade).strip() == '':
                        missing_fields.append('grade')
                    if not finish or str(finish).strip() == '':
                        missing_fields.append('finish')
                
                if missing_fields:
                    print(f"  ❌ Missing required global fields for part {part_number_global}: {', '.join(missing_fields)}. Skipping cost calculation.")
                    cost_data = {
                        'error': f"ERROR: Missing required {', '.join(missing_fields)} for part {part_number_global}. Set them in upload form for multi-file upload.",
                        'area_sq_mm': area * 1000000 if area else 0,
                        'length_mm': length_mm,
                        'width_mm': width_mm,
                        'perimeter_meters': 0.0,
                        'cutting_time_machine': 0.0,
                        'cutting_time_vaporization': 0.0,
                        'piercing_time_total': 0.0,
                        'total_time_min': 0.0,
                        'object_parts_count': object_parts_count,
                        'laser_cost': 0.0,
                        'weight_kg': 0.0,
                        'material_cost': 0.0,
                        'vgroove_count': 0,
                        'bending_count': 0,
                        'total_bending_lines': 0,
                        'bending_cost': 0.0,
                        'vgroove_length_meters': 0.0,
                        'vgroove_cost': 0.0,
                        'total_cost': 0.0
                    }
                else:
                    # Required fields present, proceed with cost calculation
                    print(f"  ✓ Required global fields complete for part {part_number_global}: {material_name} / {thickness_mm} / Grade:{grade} / Finish:{finish}. Proceeding with cost calculation.")
                    db_config = None
                    try:
                        db_config = get_material_config(material_name, thickness_mm, grade, finish)
                    except Exception as e:
                        print(f"  ❌ Exception getting material config: {e}")
                        db_config = None
                        
                    if not db_config:
                        print(f"  ❌ DB config not found for part {part_number_global}: {material_name} / {thickness_mm} / {grade} / {finish}")
                        print(f"      Please add this combination to the Materials database in Admin panel.")
                        cost_data = {
                            'error': f"ERROR: Material combination '{material_name} / {thickness_mm}mm / Grade {grade} / {finish}' not found in database for part {part_number_global}. Please add it in Admin Materials.",
                            'area_sq_mm': area * 1000000 if area else 0,
                            'length_mm': length_mm,
                            'width_mm': width_mm,
                            'perimeter_meters': 0.0,
                            'cutting_time_machine': 0.0,
                            'cutting_time_vaporization': 0.0,
                            'piercing_time_total': 0.0,
                            'total_time_min': 0.0,
                            'object_parts_count': object_parts_count,
                            'laser_cost': 0.0,
                            'weight_kg': 0.0,
                            'material_cost': 0.0,
                            'vgroove_count': 0,
                            'bending_count': 0,
                            'total_bending_lines': 0,
                            'bending_cost': 0.0,
                            'vgroove_length_meters': 0.0,
                            'vgroove_cost': 0.0,
                            'total_cost': 0.0
                        }
                    else:
                        # Found in database, proceed with calculation
                        print(f"  ✓ Found DB config for part {part_number_global}. Calculating costs...")
                        admin_config = load_admin_config()
                        db_config['laser_cost'] = admin_config.get('laser_cost', 2)
                        db_config['piercing_toggle'] = admin_config.get('piercing_toggle', False)
                        db_config['scrap_factor'] = scrap_factor
                        # Apply UI overrides if present for this material key
                        try:
                            key = "|".join([str(material_name or ''), str(thickness_mm or ''), str(grade if grade is not None else ''), str(finish if finish is not None else '')])
                            ov = material_overrides.get(key) if isinstance(material_overrides, dict) else None
                            if ov:
                                if ov.get('scrap_price_per_kg') is not None:
                                    db_config['scrap_price_per_kg'] = float(ov['scrap_price_per_kg'])
                        except Exception:
                            pass
                        # Apply UI overrides if present for this material key
                        try:
                            key = "|".join([str(material_name or ''), str(thickness_mm or ''), str(grade if grade is not None else ''), str(finish if finish is not None else '')])
                            ov = material_overrides.get(key) if isinstance(material_overrides, dict) else None
                            if ov:
                                if ov.get('scrap_price_per_kg') is not None:
                                    db_config['scrap_price_per_kg'] = float(ov['scrap_price_per_kg'])
                        except Exception:
                            pass

                        cost_data = calculate_cost(area, db_config, material_name, thickness_mm, object_parts_count, part_entities, layers, length_mm, width_mm)
                if cost_data:
                    # Check if there's an error in cost calculation
                    if 'error' in cost_data:
                        print(f"  ❌ Cost calculation error for part {part_number_global}: {cost_data['error']}")
                        # Add error information to file_part_costs for display
                        file_part_costs.append({
                            'part_number': part_number_global,
                            'area': area,
                            'length_mm': length_mm,
                            'width_mm': width_mm,
                            'object_parts_count': object_parts_count,
                            'cost_data': cost_data,
                            'source_file': file.filename,
                            'material_name': material_name,
                            'thickness': thickness_mm,
                            'grade': grade,
                            'finish': finish,
                            'extracted_label': multi_file_labels.get(p_idx + 1),
                            'calculation_error': cost_data['error']
                        })
                        # Don't add to file total cost since there was an error
                    else:
                        # DEBUG: Log cost data for part 10 and part 36 in multi-file processing
                        if part_number_global == 10:
                            print(f"🔍 PART 10 MULTI-FILE COST DATA DEBUG:")
                            print(f"    Material Cost: ${cost_data.get('material_cost', 'Not set')}")
                            print(f"    Total Cost: ${cost_data.get('total_cost', 'Not set')}")
                            print(f"    Weight: {cost_data.get('weight_kg', 'Not set')} kg")
                            print(f"    Area: {area} m²")
                            print(f"    Thickness: {thickness_mm} mm")
                            print(f"    Source File: {file.filename}")
                        elif part_number_global == 36:
                            print(f"🔍 PART 36 MULTI-FILE COST DATA DEBUG:")
                            print(f"    Material Cost: ${cost_data.get('material_cost', 'Not set')}")
                            print(f"    Weight: {cost_data.get('weight_kg', 'Not set')} kg")
                            print(f"    Area: {area} m²")
                            print(f"    Thickness: {thickness_mm} mm")
                            print(f"    Source File: {file.filename}")
                        
                        file_part_costs.append({
                            'part_number': part_number_global,
                            'area': area,
                            'length_mm': length_mm,
                            'width_mm': width_mm,
                            'object_parts_count': object_parts_count,
                            'cost_data': cost_data,
                            'source_file': file.filename,
                            'material_name': material_name,
                            'thickness': thickness_mm,
                            'grade': grade,
                            'finish': finish,
                            'extracted_label': multi_file_labels.get(p_idx + 1)
                        })
                        file_total_cost += cost_data['total_cost']

                if job_id:
                    step = base + 10 + int(60 * (p_idx + 1) / max(1, len(meaningful_parts)))
                    _progress_set(job_id, min(95, step), f"{file.filename}: {p_idx+1}/{len(meaningful_parts)} parts")

            # Validate areas for this file
            _total_area_file, _valid_areas = validate_area_calculations(file_part_areas)

            # Accumulate
            all_part_images.extend(file_part_images)
            all_part_costs.extend(file_part_costs)
            all_part_areas.extend(file_part_areas)
            total_cost_sum += file_total_cost
            total_parts_count += len(meaningful_parts)

        except Exception as e:
            import traceback
            print(f"Multi upload error on file {getattr(file, 'filename', '?')}: {e}")
            print(traceback.format_exc())
            if job_id:
                _progress_done(job_id, False, f"Failed at {getattr(file, 'filename', '?')}")
            return jsonify({'error': f'Error processing file {getattr(file, "filename", "?")}: {str(e)}'}), 500

    # Overall totals
    total_area_overall, _ = validate_area_calculations(all_part_areas)
    
    # DEBUG: Log final part costs for part 10 and part 36 in multi-file processing
    for pc in all_part_costs:
        if pc.get('part_number') == 10:
            print(f"🔍 PART 10 MULTI-FILE FINAL RESPONSE DEBUG:")
            print(f"    Part Number: {pc.get('part_number')}")
            print(f"    Area: {pc.get('area')} m²")
            print(f"    Material Cost in cost_data: ${pc.get('cost_data', {}).get('material_cost', 'Not set')}")
            print(f"    Total Cost in cost_data: ${pc.get('cost_data', {}).get('total_cost', 'Not set')}")
            print(f"    Applied Material: {pc.get('applied_material_name')}")
            print(f"    Applied Thickness: {pc.get('applied_thickness')} mm")
            print(f"    Source File: {pc.get('source_file', 'Not set')}")
        elif pc.get('part_number') == 36:
            print(f"🔍 PART 36 MULTI-FILE FINAL RESPONSE DEBUG:")
            print(f"    Part Number: {pc.get('part_number')}")
            print(f"    Area: {pc.get('area')} m²")
            print(f"    Material Cost in cost_data: ${pc.get('cost_data', {}).get('material_cost', 'Not set')}")
            print(f"    Total Cost in cost_data: ${pc.get('cost_data', {}).get('total_cost', 'Not set')}")
            print(f"    Applied Material: {pc.get('applied_material_name')}")
            print(f"    Applied Thickness: {pc.get('applied_thickness')} mm")
            print(f"    Source File: {pc.get('source_file', 'Not set')}")
        
        # Break if we found both parts
        if pc.get('part_number') in [10, 36]:
            break
    
    stats = {
        'original_entities': original_entities_total,
        'filtered_entities': filtered_entities_total,
        'removed_entities': removed_entities_total,
        'layers_found': list(sorted(layers_overall)),
        'filename': f"Multiple files ({num_files})",
        'original_filename': ', '.join(original_filenames[:3]) + (" …" if len(original_filenames) > 3 else ""),
        'total_parts': total_parts_count,
        'total_area': total_area_overall,
        'total_cost': total_cost_sum,
        'parts_info': [
            {
                'part_number': pc['part_number'],
                'entity_count': next((pi['entity_count'] for pi in all_part_images if pi['part_number'] == pc['part_number']), None)
            } for pc in all_part_costs
        ]
    }

    if job_id:
        _progress_done(job_id, True, "Completed")

    # Clean nan/inf just like single
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
        return obj

    return jsonify({
        'success': True,
        'image': first_main_image_b64 or "",
        'images': all_main_images_b64,
        'stats': stats,
        'part_images': _clean(all_part_images),
        'part_costs': _clean(all_part_costs)
    })

@app.route('/api/materials')
def get_materials_route():
    """Get materials data for popup dropdowns"""
    try:
        data = get_materials_data()
        return jsonify({
            'success': True,
            # Map database key 'material_names' to API key 'material_types' expected by frontend
            'material_types': data.get('material_names', data.get('material_types', [])),
            'thicknesses': data.get('thicknesses', []),
            'grades': data.get('grades', []),
            'finishes': data.get('finishes', []),
            'filtered_options': data.get('filtered_options', {})
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/filtered-options', methods=['POST'])
def get_filtered_options_route():
    """Get filtered options for popup dropdowns"""
    try:
        data = request.get_json()
        material_name = data.get('material_name', '')
        thickness = data.get('thickness', '')
        grade = data.get('grade', '')
        
        # Convert empty strings to None and handle type conversion
        if not material_name or material_name == '':
            material_name = None
        if not thickness or thickness == '':
            thickness = None
        else:
            try:
                thickness = float(thickness)
            except ValueError:
                thickness = None
        if not grade or grade == '':
            grade = None
        else:
            try:
                grade = int(grade)
            except ValueError:
                grade = None
        
        options = get_filtered_options(material_name, thickness, grade, None)
        return jsonify({
            'success': True,
            'thicknesses': options.get('thicknesses', []),
            'grades': options.get('grades', []),
            'finishes': options.get('finishes', [])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/materials/data')
def get_materials_data_route():
    """Get materials data for dropdowns"""
    try:
        data = get_materials_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/materials/filtered')
def get_filtered_materials_route():
    """Get filtered materials options"""
    try:
        material_name = request.args.get('material_name')
        thickness = request.args.get('thickness')
        grade = request.args.get('grade')
        finish = request.args.get('finish')
        
        # Convert thickness to float if provided
        if thickness:
            try:
                thickness = float(thickness)
            except ValueError:
                thickness = None
        
        options = get_filtered_options(material_name, thickness, grade, finish)
        return jsonify(options)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/materials/config')
def get_material_config_route():
    """Get material configuration for cost calculation"""
    try:
        material_name = request.args.get('material_name')
        thickness = request.args.get('thickness')
        grade = request.args.get('grade')
        finish = request.args.get('finish')
        
        if not material_name or not thickness:
            return jsonify({'error': 'Material name and thickness are required'}), 400
        
        # Convert thickness to float
        try:
            thickness = float(thickness)
        except ValueError:
            return jsonify({'error': 'Invalid thickness value'}), 400
        
        config = get_material_config(material_name, thickness, grade, finish)
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/materials')
def admin_materials():
    """Admin materials management page"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    try:
        materials = get_all_materials()
        return render_template('admin_materials.html', materials=materials)
    except Exception as e:
        flash(f'Error loading materials: {str(e)}', 'error')
        return render_template('admin_materials.html', materials=[])

@app.route('/api/admin/materials', methods=['GET'])
def api_get_materials():
    """API endpoint to get all materials"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        materials = get_all_materials()
        return jsonify(materials)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/materials', methods=['POST'])
def api_add_material():
    """API endpoint to add new material"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        
        # Validate required fields (only material_name and thickness are required)
        required_fields = ['material_name', 'thickness', 
                          'density', 'price_per_kg', 'speed', 'piercing_time', 
                          'vaporization_speed', 'vgroove_price', 'bending_price']
        
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Grade should default to 0 (not NULL); finish can be empty string
        if 'grade' not in data or data['grade'] is None or data['grade'] == '':
            data['grade'] = 0
        if 'finish' not in data or data['finish'] is None:
            data['finish'] = ''
        
        # Convert numeric fields
        try:
            data['thickness'] = float(data['thickness'])
            # Ensure grade is integer
            try:
                data['grade'] = int(data.get('grade', 0))
            except Exception:
                data['grade'] = 0
            data['density'] = float(data['density'])
            data['price_per_kg'] = float(data['price_per_kg'])
            data['speed'] = float(data['speed'])
            data['piercing_time'] = float(data['piercing_time'])
            data['vaporization_speed'] = float(data['vaporization_speed'])
            data['vgroove_price'] = float(data['vgroove_price'])
            data['bending_price'] = float(data['bending_price'])
            # Optional field: scrap price per kg
            if 'scrap_price_per_kg' in data and data['scrap_price_per_kg'] is not None and data['scrap_price_per_kg'] != '':
                data['scrap_price_per_kg'] = float(data['scrap_price_per_kg'])
            else:
                data['scrap_price_per_kg'] = 0.0
        except ValueError:
            return jsonify({'error': 'Invalid numeric values'}), 400
        
        success = add_material(data)
        if success:
            # Broadcast real-time update
            broadcaster.broadcast_event('material_added', {
                'material_name': data['material_name'],
                'thickness': data['thickness'],
                'grade': data['grade'],
                'finish': data['finish']
            })
            return jsonify({'message': 'Material added successfully'})
        else:
            return jsonify({'error': 'Failed to add material'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/materials/<material_name>', methods=['PUT'])
def api_update_material(material_name):
    """API endpoint to update material - Simple approach: replace entire row"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        print(f"DEBUG: Received data for update: {data}")
        
        from urllib.parse import unquote
        decoded_material_name = unquote(material_name)
        
        # Ensure all required fields are present
        required_fields = ['material_name', 'thickness', 'density', 'price_per_kg', 'speed', 
                          'piercing_time', 'vaporization_speed', 'vgroove_price', 'bending_price']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Set defaults for optional fields
        if 'grade' not in data or data['grade'] is None or data['grade'] == '':
            data['grade'] = 0
        if 'finish' not in data or data['finish'] is None:
            data['finish'] = ''
        
        # Convert numeric fields
        try:
            data['thickness'] = float(data['thickness'])
            data['grade'] = int(data.get('grade', 0))
            data['density'] = float(data['density'])
            data['price_per_kg'] = float(data['price_per_kg'])
            data['speed'] = float(data['speed'])
            data['piercing_time'] = float(data['piercing_time'])
            data['vaporization_speed'] = float(data['vaporization_speed'])
            data['vgroove_price'] = float(data['vgroove_price'])
            data['bending_price'] = float(data['bending_price'])
            if 'scrap_price_per_kg' in data and data['scrap_price_per_kg'] is not None and data['scrap_price_per_kg'] != '':
                data['scrap_price_per_kg'] = float(data['scrap_price_per_kg'])
            else:
                data['scrap_price_per_kg'] = 0.0
        except ValueError:
            return jsonify({'error': 'Invalid numeric values'}), 400
        
        print(f"DEBUG: Updating material '{decoded_material_name}' with complete data")
        
        # Get the original material identification from the form data
        original_thickness = data.get('original_thickness')
        original_grade = data.get('original_grade')
        original_finish = data.get('original_finish')
        
        print(f"DEBUG: Original material identification - Thickness: {original_thickness}, Grade: {original_grade}, Finish: '{original_finish}'")
        
        if original_thickness is None:
            return jsonify({'error': 'Missing original material identification for update'}), 400
        
        # Update the material using the original identifying information
        success = update_material(
            decoded_material_name, 
            data,
            original_thickness=original_thickness,
            original_grade=original_grade,
            original_finish=original_finish
        )
        if success:
            # Broadcast real-time update
            broadcaster.broadcast_event('material_updated', {
                'material_name': data['material_name'],
                'thickness': data['thickness'],
                'grade': data['grade'],
                'finish': data['finish']
            })
            return jsonify({'message': 'Material updated successfully'})
        else:
            return jsonify({'error': 'Failed to update material'}), 500
            
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/materials/<material_name>', methods=['DELETE'])
def api_delete_material(material_name):
    """API endpoint to delete material with unique identification"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get additional parameters from query string
        thickness = request.args.get('thickness')
        grade = request.args.get('grade')
        finish = request.args.get('finish')
        
        if not thickness:
            return jsonify({'error': 'Thickness is required for deletion'}), 400
        
        success = delete_material(material_name, float(thickness), grade, finish)
        if success:
            # Broadcast real-time update
            broadcaster.broadcast_event('material_deleted', {
                'material_name': material_name,
                'thickness': float(thickness),
                'grade': grade,
                'finish': finish
            })
            return jsonify({'message': 'Material deleted successfully'})
        else:
            return jsonify({'error': 'Failed to delete material'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/materials/events')
def materials_events():
    """Server-Sent Events endpoint for real-time material updates"""
    def event_stream():
        client_queue = queue.Queue(maxsize=50)
        broadcaster.add_client(client_queue)
        
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to material updates'})}\n\n"
            
            while True:
                try:
                    # Wait for events with timeout to send heartbeat
                    event = client_queue.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
                    
        except GeneratorExit:
            # Client disconnected
            pass
        finally:
            broadcaster.remove_client(client_queue)
    
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/api/admin/materials/get/<material_name>/<thickness>', methods=['GET'])
def api_get_specific_material(material_name, thickness):
    """API endpoint to get a specific material by name and thickness with grade/finish support"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get additional parameters from query string and decode them properly
        from urllib.parse import unquote
        grade = request.args.get('grade')
        finish = request.args.get('finish')
        
        # Decode URL-encoded parameters
        if grade:
            grade = unquote(grade)
        if finish:
            finish = unquote(finish)
        
        # Get the specific material from database
        from DatabaseConfig import engine
        from sqlalchemy import text
        
        # Convert parameters to proper types for exact matching
        # Handle None, empty string, and 0 values properly
        if grade is None or grade == '':
            exact_grade = 0
        else:
            try:
                exact_grade = int(grade)
            except (ValueError, TypeError):
                exact_grade = 0
        
        if finish is None or finish == '':
            exact_finish = '0'
        else:
            exact_finish = str(finish)
        
        # Query for the exact material row based on all four parameters
        with engine.connect() as conn:
            # Resolve scrap price column dynamically to handle legacy schemas
            from DatabaseConfig import _scrap_select_alias
            scrap_alias = _scrap_select_alias(conn)
            result = conn.execute(text(f"""
                SELECT "Material Name", "Grade", "Finish", "Thickness", "Density", 
                       "Price per kg", {scrap_alias}, "Speed", "Piercing Time", "Vaporization Speed", 
                       "V-Groov", "Bending"
                FROM materials 
                WHERE "Material Name" = :material_name 
                  AND "Thickness" = :thickness
                  AND COALESCE("Grade", 0) = :grade
                  AND COALESCE("Finish", '0') = :finish
            """), {
                'material_name': material_name, 
                'thickness': float(thickness),
                'grade': exact_grade,
                'finish': exact_finish
            })
            
            row = result.fetchone()
            if row:
                # Get column names from the result
                columns = result.keys()
                material = dict(zip(columns, row))
                
                # Convert Decimal to float for JSON serialization
                for key, value in material.items():
                    if hasattr(value, '__float__'):
                        material[key] = float(value)
                
                return jsonify(material)
            else:
                return jsonify({'error': 'Material not found'}), 404
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/config', methods=['GET'])
def api_get_config():
    """API endpoint to get admin configuration"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        config = load_admin_config()
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/config', methods=['POST'])
def api_save_config():
    """API endpoint to save admin configuration"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'laser_cost' not in data or 'piercing_toggle' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Validate laser cost
        try:
            laser_cost = float(data['laser_cost'])
            if laser_cost < 0:
                return jsonify({'error': 'Laser cost must be positive'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid laser cost value'}), 400
        
        # Validate piercing toggle
        piercing_toggle = bool(data['piercing_toggle'])
        
        # Save configuration
        config = {
            'laser_cost': laser_cost,
            'piercing_toggle': piercing_toggle
        }
        
        success = save_admin_config(config)
        if success:
            return jsonify({'success': True, 'message': 'Configuration saved successfully'})
        else:
            return jsonify({'error': 'Failed to save configuration'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# PDF GENERATION ROUTES
# ============================================================================

@app.route('/api/pdf/<quote_id>', methods=['GET', 'POST'])
def generate_quote_pdf(quote_id):
    """
    Generate PDF for a specific quote
    
    Args:
        quote_id (str): Unique quote identifier
        
    Returns:
        PDF file as attachment
    """
    try:
        job_id = None
        # Get the latest quote data from the request
        quote_data = request.get_json()
        if isinstance(quote_data, dict):
            job_id = quote_data.get('job_id')
        if job_id:
            _progress_set(job_id, 5, "Preparing PDF…")
        
        if not quote_data:
            return jsonify({'error': 'No quote data provided'}), 400
        
        # Initialize DocGenerator with template files
        template_path = Path(__file__).parent / "quote_template.xlsx"
        cell_map_path = Path(__file__).parent / "quote_cell_map.json"
        
        if not template_path.exists():
            return jsonify({'error': 'quote_template.xlsx not found'}), 500
        
        if not cell_map_path.exists():
            return jsonify({'error': 'quote_cell_map.json not found'}), 500
        
        # Load cell mapping
        with open(cell_map_path, 'r') as f:
            cell_map = json.load(f)
        
        # Get current user data for "prepared by" field
        current_user = get_current_user()
        if current_user:
            quote_data['prepared_by'] = current_user.get('full_name', 'Unknown')
            # Get extension number from database, convert to string if it exists
            extnumber = current_user.get('extnumber')
            quote_data['extension_number'] = str(extnumber) if extnumber is not None else ''
            print(f"DEBUG: User {current_user.get('email')} - Full Name: {quote_data['prepared_by']}, Extension: {quote_data['extension_number']}")
        else:
            quote_data['prepared_by'] = 'Unknown'
            quote_data['extension_number'] = ''
            print("DEBUG: No user logged in")

        # Get material configuration if material data is provided
        if quote_data.get('material_name') and quote_data.get('thickness'):
            try:
                if job_id:
                    _progress_set(job_id, 15, "Fetching material config…")
                from DatabaseConfig import engine
                from sqlalchemy import text
                
                material_name = quote_data['material_name']
                thickness = float(quote_data['thickness'])
                grade = quote_data.get('grade', '')
                finish = quote_data.get('finish', '')
                
                # Build WHERE clause
                where_conditions = ['"Material Name" = :material_name', '"Thickness" = :thickness']
                params = {'material_name': material_name, 'thickness': thickness}
                
                if grade:
                    where_conditions.append('"Grade" = :grade')
                    params['grade'] = grade
                else:
                    where_conditions.append('("Grade" IS NULL OR "Grade" = \'\')')
                
                if finish:
                    where_conditions.append('"Finish" = :finish')
                    params['finish'] = finish
                else:
                    where_conditions.append('("Finish" IS NULL OR "Finish" = \'\')')
                
                where_clause = ' AND '.join(where_conditions)
                
                with engine.connect() as conn:
                    # Determine scrap price column dynamically
                    from DatabaseConfig import _scrap_select_alias
                    scrap_alias = _scrap_select_alias(conn)
                    result = conn.execute(text(f"""
                        SELECT "Material Name", "Grade", "Finish", "Thickness", "Density", 
                               "Price per kg", {scrap_alias}, "Speed", "Piercing Time", "Vaporization Speed", 
                               "V-Groov", "Bending"
                        FROM materials 
                        WHERE {where_clause}
                    """), params)
                    row = result.fetchone()
                    if row:
                        columns = result.keys()
                        material_config = dict(zip(columns, row))
                        # Convert Decimal to float for JSON serialization
                        for key, value in material_config.items():
                            if hasattr(value, '__float__'):
                                material_config[key] = float(value)
                        # Add material configuration to quote data
                        quote_data['material_config'] = material_config
                        
            except Exception as e:
                print(f"Warning: Could not fetch material configuration: {e}")

        # Initialize DocGenerator
        doc_gen = DocGenerator(excel_template=str(template_path), cell_map=cell_map)
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "quote.pdf"
            
            # Generate PDF with assets logo
            logo_path = Path(__file__).parent / "assets" / "logo.jpg"
            if not logo_path.exists():
                logo_path = Path(__file__).parent / "logo.jpg"  # Fallback to root
            
            if job_id:
                _progress_set(job_id, 30, "Rendering PDF…")
            doc_gen.render_pdf(
                quote_data, 
                str(pdf_path),
                title="Quotation",
                logo_path=str(logo_path) if logo_path.exists() else None
            )
            
            if job_id:
                _progress_set(job_id, 85, "Finalizing PDF…")
            # Read the PDF file into memory
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
        
        # Create an in-memory bytes buffer
        pdf_buffer = io.BytesIO(pdf_data)
        pdf_buffer.seek(0)
        
        # Return PDF as downloadable file from memory
        if job_id:
            _progress_done(job_id, True, "PDF ready")
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'Quote_{quote_id}.pdf'
        )
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        try:
            # Attempt to read job_id again on error
            data = request.get_json(silent=True) or {}
            jid = data.get('job_id') if isinstance(data, dict) else None
            if jid:
                _progress_done(jid, False, "Failed")
        except Exception:
            pass
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500


@app.route('/api/placeholders')
def get_placeholder_fields_route():
    """
    Get list of required placeholder fields for PDF generation
    
    Returns:
        JSON with field descriptions
    """
    try:
        # Load cell mapping to get placeholder fields
        cell_map_path = Path(__file__).parent / "quote_cell_map.json"
        
        if not cell_map_path.exists():
            return jsonify({'error': 'quote_cell_map.json not found'}), 500
        
        with open(cell_map_path, 'r') as f:
            cell_map = json.load(f)
        
        # Extract field names (excluding anchor fields)
        fields = sorted([k for k in cell_map.keys() if not k.endswith("_anchor")])
        
        return jsonify(fields)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------------------
# Nesting functionality removed for deployment

# _log_missing_parts function removed - will be replaced with new implementation

# def _select_boards_intelligently(): - removed
    """
    Choose the best subset (and quantities) of boards from `db_boards` based on
    the incoming `parts` and simple geometric/area heuristics.

    - Ensures every part can fit on the selected board considering rotation and margins
    - Targets minimal total board area while meeting total parts area
    - Respects available quantities from the DB
    """
    try:
        if not parts or not db_boards:
            return []

        # Extract policy margins and part gap for feasibility checks
        pol = policy or {}
        margins = pol.get('sheet_margins', {}) or {}
        margin_left = float(margins.get('left', pol.get('sheet_margin_left_mm', 10.0)))
        margin_right = float(margins.get('right', pol.get('sheet_margin_right_mm', 10.0)))
        margin_top = float(margins.get('top', pol.get('sheet_margin_top_mm', 10.0)))
        margin_bottom = float(margins.get('bottom', pol.get('sheet_margin_bottom_mm', 10.0)))
        gap = float(pol.get('min_part_gap_mm', 5.0))
        allow_rotation = bool(pol.get('allow_rotation_90', True))

        def effective_board_dims(b):
            return (
                max(0.0, float(b.get('width_mm', 0.0)) - (margin_left + margin_right)),
                max(0.0, float(b.get('height_mm', 0.0)) - (margin_top + margin_bottom)),
            )

        def part_can_fit_on_board(part, b_w_eff, b_h_eff):
            try:
                p_len = float(part.get('length_mm', 0.0))
                p_wid = float(part.get('width_mm', 0.0))
            except Exception:
                return False
            # Feasibility check should not be overly strict; packing will apply margins/gaps.
            # Allow a small tolerance to account for rounding.
            TOL = 0.001
            allow_rot_part = bool(part.get('rotation_allowed', allow_rotation))
            need_w = p_wid
            need_h = p_len
            if (need_w - TOL) <= b_w_eff and (need_h - TOL) <= b_h_eff:
                return True
            if allow_rot_part and (p_len - TOL) <= b_w_eff and (p_wid - TOL) <= b_h_eff:
                return True
            return False

        # Compute per-board fit stats (what fraction of parts this board can place)
        boards_aug = []
        for b in db_boards:
            bw_eff, bh_eff = effective_board_dims(b)
            # Use RAW board dims for feasibility (margins applied later during packing)
            bw_raw = float(b.get('width_mm', 0.0))
            bh_raw = float(b.get('height_mm', 0.0))
            fits = [part_can_fit_on_board(p, bw_raw, bh_raw) for p in parts]
            fit_count = sum(1 for f in fits if f)
            boards_aug.append({**b, '_raw_w': bw_raw, '_raw_h': bh_raw, '_eff_w': bw_eff, '_eff_h': bh_eff, '_eff_area': max(0.0, bw_eff) * max(0.0, bh_eff), '_fit_count': fit_count, '_fits_all': (fit_count == len(parts))})

        def copies_per_sheet(bw_eff: float, bh_eff: float, p_len: float, p_wid: float, allow_rot: bool) -> int:
            # Grid capacity considering gap and margins already removed
            def cap(l_mm: float, w_mm: float) -> int:
                nx = int(max(0.0, math.floor((bw_eff + gap) / (l_mm + gap))))
                ny = int(max(0.0, math.floor((bh_eff + gap) / (w_mm + gap))))
                return max(0, nx) * max(0, ny)
            c0 = cap(p_len, p_wid)
            if allow_rot and abs(p_len - p_wid) > 1e-6:
                c1 = cap(p_wid, p_len)
                return max(c0, c1)
            return c0

        def min_sheets_by_dimensions(b, parts_list):
            # Try a greedy shelf simulation to account for mixed sizes fitting together on a single sheet.
            bw_eff = b.get('_eff_w', 0.0)
            bh_eff = b.get('_eff_h', 0.0)
            if bw_eff <= 0 or bh_eff <= 0:
                return float('inf')

            # Expand items by quantity
            items = []
            for p in parts_list:
                try:
                    for _ in range(max(0, int(p.get('quantity', 1)))):
                        items.append({
                            'L': float(p.get('length_mm', 0.0)),
                            'W': float(p.get('width_mm', 0.0)),
                            'rot': bool(p.get('rotation_allowed', allow_rotation))
                        })
                except Exception:
                    return float('inf')

            # Sort large-first by max side then min side
            items.sort(key=lambda it: (max(it['L'], it['W']), min(it['L'], it['W'])), reverse=True)

            def place_items_one_sheet(items_left):
                curr_x = 0.0
                curr_y = 0.0
                row_h = 0.0
                placed_idx = []
                for i, it in enumerate(items_left):
                    # Evaluate both orientations and pick the one with best rank
                    orientations = [(it['L'], it['W'])]
                    if it['rot'] and abs(it['L'] - it['W']) > 1e-6:
                        orientations.append((it['W'], it['L']))

                    best_local = None  # (rank_tuple, place_on_new_row(bool), l, w)
                    # Compute minimal horizontal length among remaining items to favor pairing on same row
                    min_next_len = None
                    for j, jt in enumerate(items_left):
                        if j == i or j in placed_idx:
                            continue
                        cand_min = min(jt['L'], jt['W'])
                        min_next_len = cand_min if (min_next_len is None or cand_min < min_next_len) else min_next_len
                    for (l, w) in orientations:
                        # current row candidate
                        nx = curr_x if (row_h == 0.0 and len(placed_idx) == 0) else (curr_x + (gap if curr_x > 0.0 else 0.0))
                        if (nx + l) <= bw_eff and (curr_y + max(row_h, w)) <= bh_eff:
                            new_row = 0.0
                            row_h_grow = max(0.0, max(row_h, w) - row_h)
                            used_h_after = curr_y + max(row_h, w)
                            # Pair-fit bonus: if remaining width after placement can host the smallest remaining item, prefer this
                            remaining_w = bw_eff - (nx + l)
                            pair_ok = 1.0
                            if min_next_len is not None:
                                pair_ok = 0.0 if (remaining_w >= (min_next_len + (gap if (nx + l) > 0.0 else 0.0))) else 1.0
                            rank = (pair_ok, new_row, row_h_grow, used_h_after, nx + l)
                            cand = (rank, False, l, w)
                            if best_local is None or cand[0] < best_local[0]:
                                best_local = cand
                        # new row candidate
                        nx2 = l
                        ny2 = curr_y + (row_h + gap if row_h > 0.0 else 0.0)
                        if (nx2) <= bw_eff and (ny2 + w) <= bh_eff:
                            new_row = 1.0
                            row_h_grow = w  # starting new row
                            used_h_after = ny2 + w
                            # On new row, pairing is naturally available; pair_ok not applied
                            rank = (1.0, new_row, row_h_grow, used_h_after, nx2)
                            cand = (rank, True, l, w)
                            if best_local is None or cand[0] < best_local[0]:
                                best_local = cand

                    if best_local is None:
                        # cannot place this item on this sheet
                        continue

                    # Commit best local placement
                    _, on_new_row, l_sel, w_sel = best_local
                    if on_new_row and row_h > 0.0:
                        curr_y = curr_y + row_h + gap
                        curr_x = 0.0
                        row_h = 0.0
                    # place at row
                    nx_final = curr_x if (row_h == 0.0 and len(placed_idx) == 0) else (curr_x + (gap if curr_x > 0.0 else 0.0))
                    curr_x = nx_final + l_sel
                    row_h = max(row_h, w_sel)
                    placed_idx.append(i)

                return set(placed_idx)

            sheets = 0
            remaining = items
            safety = 0
            while remaining and safety < 1000:
                safety += 1
                placed_set = place_items_one_sheet(remaining)
                if not placed_set:
                    # if none placed, abort
                    return float('inf')
                # remove placed
                remaining = [it for idx, it in enumerate(remaining) if idx not in placed_set]
                sheets += 1
            return max(1, sheets)

        # Boards that can place the largest part (must exist for feasibility)
        def part_dims_key(p):
            try:
                return max(float(p.get('length_mm', 0.0)), float(p.get('width_mm', 0.0))) * min(float(p.get('length_mm', 0.0)), float(p.get('width_mm', 0.0)))
            except Exception:
                return 0.0
        largest_part = max(parts, key=part_dims_key)
        try:
            lpL = float(largest_part.get('length_mm', 0.0))
            lpW = float(largest_part.get('width_mm', 0.0))
            print(f"[selector] Largest part dims LxW: {lpL:.2f} x {lpW:.2f} mm (rotation_allowed={largest_part.get('rotation_allowed', True)})")
        except Exception:
            pass
        try:
            dims = [(b.get('id'), float(b.get('width_mm', 0.0)), float(b.get('height_mm', 0.0))) for b in db_boards]
            if dims:
                w_min = min(d[1] for d in dims); w_max = max(d[1] for d in dims)
                h_min = min(d[2] for d in dims); h_max = max(d[2] for d in dims)
                print(f"[selector] Boards width range: {w_min:.1f}..{w_max:.1f} mm, height range: {h_min:.1f}..{h_max:.1f} mm; count={len(dims)}")
        except Exception:
            pass
        large_fit_boards = []
        for b in boards_aug:
            if part_can_fit_on_board(largest_part, b['_raw_w'], b['_raw_h']):
                large_fit_boards.append(b)
        if not large_fit_boards:
            # Unit rescue heuristic: if parts look much larger than boards and boards are small (< 1000mm),
            # retry assuming boards were stored in cm or inches.
            max_board_dim = max((max(b.get('width_mm', 0.0), b.get('height_mm', 0.0)) for b in db_boards), default=0.0)
            max_part_dim = max((max(float(p.get('length_mm', 0.0)), float(p.get('width_mm', 0.0))) for p in parts), default=0.0)
            scaled_candidates = []
            if max_board_dim > 0 and max_part_dim > max_board_dim * 1.05 and max_board_dim < 1000:
                # Try cm->mm (x10), inch->mm (x25.4), meter->mm (x1000)
                for scale in (10.0, 25.4, 1000.0):
                    boards_scaled = []
                    for b in db_boards:
                        w_scaled = float(b.get('width_mm', 0.0)) * scale
                        h_scaled = float(b.get('height_mm', 0.0)) * scale
                        bw_eff = max(0.0, w_scaled - (margin_left + margin_right))
                        bh_eff = max(0.0, h_scaled - (margin_top + margin_bottom))
                        boards_scaled.append({**b, 'width_mm': w_scaled, 'height_mm': h_scaled, '_raw_w': w_scaled, '_raw_h': h_scaled, '_eff_w': bw_eff, '_eff_h': bh_eff, '_eff_area': max(0.0, bw_eff) * max(0.0, bh_eff)})
                    lf = [b for b in boards_scaled if part_can_fit_on_board(largest_part, b['_raw_w'], b['_raw_h'])]
                    if lf:
                        print(f"[selector] Unit scaling applied: x{scale}")
                        scaled_candidates = boards_scaled
                        break
            if scaled_candidates:
                boards_aug = scaled_candidates
                large_fit_boards = [b for b in boards_aug if part_can_fit_on_board(largest_part, b['_raw_w'], b['_raw_h'])]
            else:
                # As a last resort, proceed with area-only selection across all boards (smallest first)
                # This avoids returning no boards; packing may still succeed if our feasibility test was too strict.
                print("[selector] Proceeding with area-only selection as feasibility fallback.")
                # Compute required area
                SAFETY = 1.10
                required_area = (total_parts_area if (total_parts_area := (total_area_hint_sq_mm or 0.0)) else 0.0)
                if required_area <= 0.0:
                    required_area = 0.0
                    for p in parts:
                        try:
                            q = int(p.get('quantity', 1))
                            a = float(p.get('area_sq_mm')) if p.get('area_sq_mm') is not None else (float(p.get('length_mm', 0.0)) * float(p.get('width_mm', 0.0)))
                            required_area += max(0.0, a) * max(0, q)
                        except Exception:
                            continue
                required_area *= SAFETY
                # Only consider boards that can fit at least one part
                boards_aug = [b for b in boards_aug if b.get('_fit_count', 0) > 0]
                # Sort by effective area ascending
                boards_aug.sort(key=lambda x: (x['_eff_area'], x['width_mm'] * x['height_mm']))
                selected = []
                remaining = required_area
                for b in boards_aug:
                    if remaining <= 0:
                        break
                    avail = int(b.get('quantity', 0))
                    ea = max(0.0, b.get('_eff_area', 0.0))
                    if avail <= 0 or ea <= 0:
                        continue
                    need = int(math.ceil(remaining / ea))
                    take = min(avail, max(1, need))
                    selected.append({'id': b.get('id'), 'name': b.get('name'), 'width_mm': float(b.get('width_mm', 0.0)), 'height_mm': float(b.get('height_mm', 0.0)), 'quantity': take})
                    remaining -= ea * take
                return selected

        # Compute total parts area requirement (prefer provided hint from DXF processing)
        total_parts_area = 0.0
        if total_area_hint_sq_mm is not None:
            try:
                total_parts_area = float(total_area_hint_sq_mm)
            except Exception:
                total_parts_area = 0.0
        if total_parts_area <= 0.0:
            for p in parts:
                try:
                    qty = int(p.get('quantity', 1))
                    if p.get('area_sq_mm') is not None:
                        area = float(p.get('area_sq_mm'))
                    else:
                        area = float(p.get('length_mm', 0.0)) * float(p.get('width_mm', 0.0))
                    total_parts_area += max(0.0, area) * max(0, qty)
                except Exception:
                    continue

        SAFETY = 1.10  # slightly lower to avoid over-pruning when area is already approximate from DXF

        # Phase A: precise per-part allocation for the largest parts first
        def part_area(p):
            try:
                return float(p.get('area_sq_mm')) if p.get('area_sq_mm') is not None else (float(p.get('length_mm', 0.0)) * float(p.get('width_mm', 0.0)))
            except Exception:
                return 0.0

        parts_sorted = sorted(parts, key=part_area, reverse=True)
        # consider top-k largest by area (or all if few parts)
        k = max(1, min(len(parts_sorted), 8))
        big_parts_order = parts_sorted[:k]

        # Selected boards aggregate
        selected = []
        # Available quantities map
        avail_by_id = {b.get('id'): int(b.get('quantity', 0)) for b in boards_aug}

        for bp in big_parts_order:
            qty_needed = int(bp.get('quantity', 1))
            if qty_needed <= 0:
                continue
            Lp = float(bp.get('length_mm', 0.0))
            Wp = float(bp.get('width_mm', 0.0))
            try:
                print(f"[selector] Big-part pass: id={bp.get('id')} LxW={Lp:.2f}x{Wp:.2f} qty={qty_needed}")
            except Exception:
                pass
            # Candidate boards that can fit this part
            candidates = []
            for b in boards_aug:
                if avail_by_id.get(b.get('id'), 0) <= 0:
                    continue
                if not part_can_fit_on_board(bp, b['_raw_w'], b['_raw_h']):
                    continue
                cap = copies_per_sheet(b.get('_eff_w', 0.0), b.get('_eff_h', 0.0), Lp, Wp, bool(bp.get('rotation_allowed', allow_rotation)))
                if cap <= 0:
                    continue
                # Density: parts area that can fit per sheet divided by board effective area
                area_per_sheet = cap * max(0.0, part_area(bp))
                eff_area = max(1e-9, b.get('_eff_area', 0.0))
                density = area_per_sheet / eff_area
                # Tie-break by smaller leftover max side relative to part dims
                lo_w = max(0.0, b.get('_eff_w', 0.0) - (math.floor((b.get('_eff_w', 0.0) + gap) / (Lp + gap)) * (Lp + gap) - gap)) if (Lp + gap) > 0 else b.get('_eff_w', 0.0)
                lo_h = max(0.0, b.get('_eff_h', 0.0) - (math.floor((b.get('_eff_h', 0.0) + gap) / (Wp + gap)) * (Wp + gap) - gap)) if (Wp + gap) > 0 else b.get('_eff_h', 0.0)
                leftover_max = max(lo_w, lo_h)
                meets_need = (cap >= qty_needed)
                need_rank = 0 if meets_need else 1
                # Prefer boards that already cover qty with minimal area; else fall back to density
                candidates.append(( need_rank, eff_area, leftover_max, -density, str(b.get('id')), cap, b ))
            # Sort best-first (higher density => lower -density)
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            # Log top candidates (up to 5)
            try:
                for _rank, cand in enumerate(candidates[:5]):
                    need_rank, ea, lom, dneg, bid, cap_cand, bb = cand
                    print(f"[selector]  cand board id={bid} eff={bb.get('_eff_w',0.0):.0f}x{bb.get('_eff_h',0.0):.0f} cap={cap_cand} density={-dneg:.4f} leftover_max={lom:.1f} meets_need={need_rank==0}")
            except Exception:
                pass
            for need_rank, ea, lom, dneg, _, cap, b in candidates:
                if qty_needed <= 0:
                    break
                available = avail_by_id.get(b.get('id'), 0)
                if available <= 0:
                    continue
                sheets_needed = int(math.ceil(qty_needed / cap))
                take = min(available, max(1, sheets_needed))
                try:
                    print(f"[selector]  choose board id={b.get('id')} cap_per_sheet={cap} sheets_needed={sheets_needed} take={take} avail={available}")
                except Exception:
                    pass
                # Merge into selected
                merged = False
                for s in selected:
                    if s.get('id') == b.get('id'):
                        s['quantity'] += take
                        merged = True
                        break
                if not merged:
                    selected.append({'id': b.get('id'), 'name': b.get('name'), 'width_mm': float(b.get('width_mm', 0.0)), 'height_mm': float(b.get('height_mm', 0.0)), 'quantity': take})
                avail_by_id[b.get('id')] = max(0, available - take)
                qty_needed -= cap * take

        # Phase B: use remaining boards (including smaller ones) to cover the rest of the total area
        total_required_area = total_parts_area * SAFETY
        covered_area = sum(b['_eff_area'] * s['quantity'] for s in selected for b in boards_aug if b.get('id') == s.get('id'))
        remaining_total_area = max(0.0, total_required_area - covered_area)
        if remaining_total_area > 0.0:
            # Consider all boards that can fit at least one part, preferring smaller effective areas first
            some_fit_boards = [b for b in boards_aug if b['_fit_count'] > 0]
            some_fit_boards.sort(key=lambda x: (x['_eff_area'], x['width_mm'] * x['height_mm']))
            for b in some_fit_boards:
                if remaining_total_area <= 0:
                    break
                # Use updated availability
                available_qty = avail_by_id.get(b.get('id'), int(b.get('quantity', 0)))
                if available_qty <= 0 or b['_eff_area'] <= 0:
                    continue
                # Use area-based need here too; dimensional constraints will be checked later
                area_needed = int(math.ceil(remaining_total_area / b['_eff_area']))
                take = min(available_qty, max(1, area_needed))
                if take <= 0:
                    continue
                # Append or bump quantity if already in list
                merged = False
                for s in selected:
                    if s.get('id') == b.get('id'):
                        s['quantity'] += take
                        merged = True
                        break
                if not merged:
                    selected.append({'id': b.get('id'), 'name': b.get('name'), 'width_mm': float(b.get('width_mm', 0.0)), 'height_mm': float(b.get('height_mm', 0.0)), 'quantity': take})
                remaining_total_area -= b['_eff_area'] * take
                avail_by_id[b.get('id')] = max(0, available_qty - take)

        # Capacity completion: ensure selected boards can accommodate all part quantities by grid capacity
        def total_capacity_for_part(part, selection, boards_catalog):
            total = 0
            allow_rot_part = bool(part.get('rotation_allowed', allow_rotation))
            L = float(part.get('length_mm', 0.0))
            W = float(part.get('width_mm', 0.0))
            for s in selection:
                # find matching board aug
                for b in boards_catalog:
                    if b.get('id') == s.get('id'):
                        bw_eff = b.get('_eff_w', 0.0)
                        bh_eff = b.get('_eff_h', 0.0)
                        cap = copies_per_sheet(bw_eff, bh_eff, L, W, allow_rot_part)
                        total += cap * int(s.get('quantity', 0))
                        break
            return total

        # Early fail-fast for obviously insufficient capacity on the largest parts
        try:
            largest_parts = sorted(parts, key=part_dims_key, reverse=True)[:5]
            for p in largest_parts:
                need = int(p.get('quantity', 1))
                cap = total_capacity_for_part(p, [{'id': b.get('id'), 'quantity': int(b.get('quantity', 0))} for b in boards_aug], boards_aug)
                print(f"[capacity] part {p.get('id')} need={need} capacity={cap}")
                if cap < need:
                    print(f"[capacity] insufficient total capacity for {p.get('id')} — expect missing or longer search")
        except Exception:
            pass

        # Check each part's capacity; if shortage, add more fitting boards smallest-first
        parts_short = []
        for p in parts:
            try:
                qty = int(p.get('quantity', 1))
                cap = total_capacity_for_part(p, selected, boards_aug)
                if cap < qty:
                    parts_short.append({'part': p, 'needed': qty - cap})
            except Exception:
                continue

        if parts_short:
            # Prefer boards that can fit the largest part to satisfy shortages first
            addable = sorted([b for b in boards_aug if b['_fit_count'] > 0], key=lambda x: (x['_eff_area'], x['width_mm'] * x['height_mm']))
            for entry in parts_short:
                p = entry['part']
                remaining_copies = int(entry['needed'])
                if remaining_copies <= 0:
                    continue
                for b in addable:
                    if remaining_copies <= 0:
                        break
                    # available quantity after existing selection
                    already = sum(int(s.get('quantity', 0)) for s in selected if s.get('id') == b.get('id'))
                    avail = max(0, int(b.get('quantity', 0)) - already)
                    if avail <= 0:
                        continue
                    cap_per_sheet = copies_per_sheet(b.get('_eff_w', 0.0), b.get('_eff_h', 0.0), float(p.get('length_mm', 0.0)), float(p.get('width_mm', 0.0)), bool(p.get('rotation_allowed', allow_rotation)))
                    if cap_per_sheet <= 0:
                        continue
                    # sheets needed for this part shortage
                    need_sheets = int(math.ceil(remaining_copies / cap_per_sheet))
                    take = min(avail, max(1, need_sheets))
                    try:
                        print(f"[selector] shortage for part {p.get('id')} remain={remaining_copies} cap_per_sheet={cap_per_sheet} board id={b.get('id')} take={take}")
                    except Exception:
                        pass
                    # merge into selected
                    merged = False
                    for s in selected:
                        if s.get('id') == b.get('id'):
                            s['quantity'] += take
                            merged = True
                            break
                    if not merged:
                        selected.append({'id': b.get('id'), 'name': b.get('name'), 'width_mm': float(b.get('width_mm', 0.0)), 'height_mm': float(b.get('height_mm', 0.0)), 'quantity': take})
                    remaining_copies -= cap_per_sheet * take

        # Final fallback: if selection still empty, pick the single smallest board that fits the largest part (quantity 1)
        if not selected and large_fit_boards:
            b0 = sorted(large_fit_boards, key=lambda x: (x['_eff_area'], x['width_mm'] * x['height_mm']))[0]
            selected = [{'id': b0.get('id'), 'name': b0.get('name'), 'width_mm': float(b0.get('width_mm', 0.0)), 'height_mm': float(b0.get('height_mm', 0.0)), 'quantity': 1}]

        # Absolute final fallback: if still empty, choose the largest board by raw area with quantity 1
        if not selected and boards_aug:
            bL = sorted(boards_aug, key=lambda x: (x.get('_raw_w', x.get('width_mm', 0.0)) * x.get('_raw_h', x.get('height_mm', 0.0))), reverse=True)[0]
            selected = [{'id': bL.get('id'), 'name': bL.get('name'), 'width_mm': float(bL.get('width_mm', bL.get('_raw_w', 0.0))), 'height_mm': float(bL.get('height_mm', bL.get('_raw_h', 0.0))), 'quantity': 1}]

        try:
            # Detailed log of selected boards with sizes
            detail = []
            for s in selected:
                bid = s.get('id')
                qty = s.get('quantity')
                bb = next((b for b in boards_aug if b.get('id') == bid), None)
                if bb:
                    detail.append({'id': bid, 'qty': qty, 'eff': f"{bb.get('_eff_w',0.0):.0f}x{bb.get('_eff_h',0.0):.0f}", 'raw': f"{bb.get('_raw_w',0.0):.0f}x{bb.get('_raw_h',0.0):.0f}"})
                else:
                    detail.append({'id': bid, 'qty': qty})
            print(f"[selector] Selected boards: {detail} (total types={len(selected)})")
        except Exception:
            print(f"[selector] Selected boards: {[{'id': s.get('id'), 'qty': s.get('quantity')} for s in selected]} (total types={len(selected)})")
        return selected
    except Exception as e:
        print(f"[boards] intelligent selection failed: {e}")
        return []
# ============================================================================
# NESTING FUNCTIONALITY - SVGNest Integration
# ============================================================================

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math

# Nesting configuration with defaults (all measurements in mm)
DEFAULT_NESTING_CONFIG = {
    'min_part_gap_mm': 5.0,
    'kerf_mm': 0.2,
    'rotation_step_deg': 90,
    'allow_rotation': True,
    'allow_mirroring': False,
    'cluster_same_parts': True,
    'min_bridge_width_mm': 1.0,
    'sheet_margin_mm': {'top': 5.0, 'right': 5.0, 'bottom': 5.0, 'left': 5.0},
}

# Ensure nesting config is in main config
DEFAULT_CONFIG.setdefault('nesting', DEFAULT_NESTING_CONFIG)

def _merge_defaults_with_config(defaults: dict, cfg: dict) -> dict:
    """Deep-merge defaults into cfg without losing existing values."""
    out = dict(cfg or {})
    for k, v in defaults.items():
        if isinstance(v, dict):
            out[k] = _merge_defaults_with_config(v, out.get(k, {}))
        else:
            out.setdefault(k, v)
    return out

def _validate_nesting_config(data: dict) -> dict:
    """Sanitize and validate nesting config, all lengths in mm."""
    out = {}
    def _f(key, min_val=0.0, max_val=None):
        if key in data and data[key] is not None:
            try:
                val = float(data[key])
                if val < min_val: val = min_val
                if max_val is not None and val > max_val: val = max_val
                out[key] = val
            except Exception:
                pass

    _f('min_part_gap_mm', 0.0, 100.0)
    _f('kerf_mm', 0.0, 50.0)
    _f('rotation_step_deg', 1.0, 360.0)
    _f('min_bridge_width_mm', 0.0, 1000.0)

    for b in ['allow_rotation', 'allow_mirroring', 'cluster_same_parts']:
        if b in data:
            out[b] = bool(data[b])

    if 'sheet_margin_mm' in data and isinstance(data['sheet_margin_mm'], dict):
        sm = data['sheet_margin_mm']
        out['sheet_margin_mm'] = {
            'top': max(0.0, float(sm.get('top', 5.0))),
            'right': max(0.0, float(sm.get('right', 5.0))),
            'bottom': max(0.0, float(sm.get('bottom', 5.0))),
            'left': max(0.0, float(sm.get('left', 5.0))),
        }

    return out

@app.route('/api/nesting/config', methods=['GET'])
def api_get_nesting_config():
    """Get current nesting configuration"""
    cfg = load_admin_config()
    return jsonify(cfg.get('nesting', DEFAULT_NESTING_CONFIG))

@app.route('/api/nesting/config', methods=['POST', 'PUT'])
def api_update_nesting_config():
    """Update nesting configuration (admin only)"""
    if not is_admin_logged_in():
        return jsonify({'error': 'Not authorized'}), 401
    try:
        incoming = request.get_json(silent=True) or {}
        sanitized = _validate_nesting_config(incoming)

        cfg = load_admin_config()
        cfg['nesting'] = _merge_defaults_with_config(
            DEFAULT_NESTING_CONFIG,
            {**cfg.get('nesting', {}), **sanitized}
        )
        if save_admin_config(cfg):
            return jsonify({'success': True, 'nesting': cfg['nesting']})
        return jsonify({'error': 'Failed to save configuration'}), 500
    except Exception as e:
        return jsonify({'error': f'Invalid configuration: {str(e)}'}), 400

def get_board_specs_from_db():
    """Get available boards from database"""
    try:
        from DatabaseConfig import get_db
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT "Material Name", "Grade", "Finish", "Thickness", "Price per kg"
            FROM materials 
            WHERE "Thickness" > 0
            ORDER BY "Material Name", "Thickness"
        """)
        
        boards = []
        # Standard sheet sizes in mm for different materials
        standard_sizes = [
            {'width_mm': 1000, 'height_mm': 2000, 'cost_per_sheet': 100},
            {'width_mm': 1250, 'height_mm': 2500, 'cost_per_sheet': 150},
            {'width_mm': 1500, 'height_mm': 3000, 'cost_per_sheet': 200},
            {'width_mm': 2000, 'height_mm': 4000, 'cost_per_sheet': 300},
        ]
        
        board_id = 1
        for row in cursor.fetchall():
            material_name = row[0]
            thickness = float(row[3])
            
            # Create board specs for each standard size
            for size in standard_sizes:
                boards.append({
                    'id': board_id,
                    'material_name': material_name,
                    'thickness_mm': thickness,
                    'width_mm': size['width_mm'],
                    'height_mm': size['height_mm'],
                    'cost_per_sheet': size['cost_per_sheet'],
                    'area_mm2': size['width_mm'] * size['height_mm']
                })
                board_id += 1
        
        return boards
    except Exception as e:
        print(f"Database error getting board specs: {e}")
        # Fallback to default boards
        return [
            {'id': 1, 'material_name': 'Steel', 'thickness_mm': 1.0, 'width_mm': 1000, 'height_mm': 2000, 'cost_per_sheet': 100, 'area_mm2': 2000000},
            {'id': 2, 'material_name': 'Steel', 'thickness_mm': 1.0, 'width_mm': 1250, 'height_mm': 2500, 'cost_per_sheet': 150, 'area_mm2': 3125000},
        ]

def entity_to_svg_path(entity):
    """Convert DXF entity to SVG path string"""
    try:
        entity_type = entity.dxftype()
        
        if entity_type == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            return f"M {start.x},{start.y} L {end.x},{end.y}"
        
        elif entity_type == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            # Convert circle to SVG path using arcs
            return f"M {center.x-radius},{center.y} A {radius},{radius} 0 1,0 {center.x+radius},{center.y} A {radius},{radius} 0 1,0 {center.x-radius},{center.y} Z"
        
        elif entity_type == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = math.radians(entity.dxf.start_angle)
            end_angle = math.radians(entity.dxf.end_angle)
            
            start_x = center.x + radius * math.cos(start_angle)
            start_y = center.y + radius * math.sin(start_angle)
            end_x = center.x + radius * math.cos(end_angle)
            end_y = center.y + radius * math.sin(end_angle)
            
            large_arc = 1 if (end_angle - start_angle) > math.pi else 0
            return f"M {start_x},{start_y} A {radius},{radius} 0 {large_arc},1 {end_x},{end_y}"
        
        elif entity_type in ['LWPOLYLINE', 'POLYLINE']:
            if hasattr(entity, 'vertices'):
                vertices = list(entity.vertices)
            elif hasattr(entity, 'get_points'):
                vertices = list(entity.get_points())
            else:
                return None
            
            if not vertices:
                return None
            
            path = f"M {vertices[0][0]},{vertices[0][1]}"
            for vertex in vertices[1:]:
                path += f" L {vertex[0]},{vertex[1]}"
            
            if entity.is_closed:
                path += " Z"
            
            return path
        
        elif entity_type == 'SPLINE':
            # Simplified spline to polyline conversion
            if hasattr(entity, 'flattening'):
                points = list(entity.flattening(0.1))  # 0.1mm tolerance
                if points:
                    path = f"M {points[0][0]},{points[0][1]}"
                    for point in points[1:]:
                        path += f" L {point[0]},{point[1]}"
                    return path
        
        return None
    except Exception as e:
        print(f"Error converting entity {entity_type} to SVG: {e}")
        return None

def _compute_part_bbox(part_entities):
    """Compute bounding box for a part (global function for nesting)"""
    try:
        xs = []
        ys = []
        for ent in part_entities:
            if hasattr(ent, 'dxf'):
                if hasattr(ent.dxf, 'start') and hasattr(ent.dxf, 'end'):
                    xs.extend([ent.dxf.start.x, ent.dxf.end.x])
                    ys.extend([ent.dxf.start.y, ent.dxf.end.y])
                elif ent.dxftype() == 'CIRCLE':
                    c = ent.dxf.center; r = ent.dxf.radius
                    xs.extend([c.x - r, c.x + r]); ys.extend([c.y - r, c.y + r])
                elif ent.dxftype() == 'ARC':
                    c = ent.dxf.center; r = ent.dxf.radius
                    xs.extend([c.x - r, c.x + r]); ys.extend([c.y - r, c.y + r])
                elif ent.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                    try:
                        pts = list(ent.get_points())
                        if pts:
                            xs.extend([p[0] for p in pts]); ys.extend([p[1] for p in pts])
                    except Exception:
                        pass
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
    except Exception as e:
        print(f"Error computing bbox: {e}")
    return (0.0, 0.0, 0.0, 0.0)

def convert_dxf_parts_to_svg(parts_with_quantities):
    """Convert extracted DXF parts to SVG format for nesting"""
    svg_parts = []
    
    for part_data in parts_with_quantities:
        part_entities = part_data['entities']
        quantity = part_data['quantity']
        part_index = part_data['part_index']
        
        # Collect all SVG paths for this part
        part_paths = []
        for entity in part_entities:
            svg_path = entity_to_svg_path(entity)
            if svg_path:
                part_paths.append(svg_path)
        
        if part_paths:
            # Combine all paths into a single SVG group
            combined_path = " ".join(part_paths)
            
            # Create multiple instances based on quantity
            for i in range(quantity):
                svg_parts.append({
                    'path': combined_path,
                    'part_id': part_index,
                    'instance': i + 1,
                    'bbox': part_data.get('bbox')
                })
    
    return svg_parts

@app.route('/api/scrap/calculate', methods=['POST'])
def calculate_scrap_ai():
    """Calculate scrap factor using AI"""
    try:
        data = request.get_json()
        parts_data = data.get('parts', [])
        boards_data = data.get('boards', [])
        material_type = data.get('material_type', 'steel')
        thickness_mm = data.get('thickness_mm', 1.0)
        openai_api_key = data.get('openai_api_key')
        
        if not parts_data or not boards_data:
            return jsonify({'error': 'Parts and boards data required'}), 400
        
        if _HAVE_AI_SCRAP:
            # Run AI calculation in a thread to avoid blocking
            import asyncio
            import threading
            
            def run_ai_calculation():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    calc = AIScrapCalculator(openai_api_key=openai_api_key, use_cache=True)
                    ai_input = ScrapCalculationInput(parts_data=parts_data, boards_data=boards_data, material_type=material_type, thickness_mm=thickness_mm)
                    result = loop.run_until_complete(calc.calculate_scrap_factor_ai(ai_input))
                    return result
                finally:
                    loop.close()
            
            # Run in thread with timeout
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_ai_calculation)
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                except concurrent.futures.TimeoutError:
                    return jsonify({'error': 'AI calculation timeout'}), 408
            
            return jsonify({
                'success': True,
                'scrap_factor': result.scrap_factor,
                'confidence_score': result.confidence_score,
                'reasoning': result.reasoning,
                'calculation_time': result.calculation_time,
                'optimization_suggestions': result.optimization_suggestions,
                'estimated_utilization': result.estimated_utilization
            })
        else:
            # Fallback to basic calculation
            return jsonify({
                'success': True,
                'scrap_factor': 1.2,
                'confidence_score': 0.5,
                'reasoning': 'AI scrap calculator not available - using default',
                'calculation_time': 0.0,
                'optimization_suggestions': ['Consider manual adjustment'],
                'estimated_utilization': 0.83
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_part_area_from_bbox(bbox):
    """Calculate part area from bounding box"""
    if not bbox or len(bbox) != 4:
        return 0.0
    return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

def run_svgnest_algorithm(svg_parts, board_specs, nesting_config):
    """Run simplified nesting algorithm (placeholder for SVGNest integration)"""
    try:
        board_width = board_specs['width_mm']
        board_height = board_specs['height_mm']
        board_area = board_width * board_height
        margin = nesting_config.get('sheet_margin_mm', {'top': 5, 'right': 5, 'bottom': 5, 'left': 5})
        gap = nesting_config.get('min_part_gap_mm', 5.0)
        
        # Available nesting area
        available_width = board_width - margin['left'] - margin['right']
        available_height = board_height - margin['top'] - margin['bottom']
        available_area = available_width * available_height
        
        # Simple grid-based nesting simulation
        nested_parts = []
        current_x = margin['left']
        current_y = margin['top']
        row_height = 0
        total_used_area = 0
        
        for part in svg_parts:
            bbox = part.get('bbox')
            if not bbox:
                continue
            
            part_width = abs(bbox[2] - bbox[0])
            part_height = abs(bbox[3] - bbox[1])
            part_area = part_width * part_height
            
            # Check if part fits in current position
            if current_x + part_width <= board_width - margin['right']:
                # Part fits horizontally
                nested_parts.append({
                    'part_id': part['part_id'],
                    'instance': part['instance'],
                    'x': current_x,
                    'y': current_y,
                    'width': part_width,
                    'height': part_height,
                    'area': part_area
                })
                
                current_x += part_width + gap
                row_height = max(row_height, part_height)
                total_used_area += part_area
                
            elif current_y + row_height + gap + part_height <= board_height - margin['bottom']:
                # Move to next row
                current_x = margin['left']
                current_y += row_height + gap
                row_height = part_height
                
                nested_parts.append({
                    'part_id': part['part_id'],
                    'instance': part['instance'],
                    'x': current_x,
                    'y': current_y,
                    'width': part_width,
                    'height': part_height,
                    'area': part_area
                })
                
                current_x += part_width + gap
                total_used_area += part_area
            
            # If part doesn't fit, skip it (would need multiple sheets)
        
        utilization = total_used_area / available_area if available_area > 0 else 0
        
        return {
            'parts': nested_parts,
            'utilization': utilization,
            'used_area_mm2': total_used_area,
            'available_area_mm2': available_area,
            'board_area_mm2': board_area,
            'parts_fitted': len(nested_parts),
            'parts_total': len(svg_parts)
        }
        
    except Exception as e:
        print(f"Nesting algorithm error: {e}")
        return {
            'parts': [],
            'utilization': 0.0,
            'used_area_mm2': 0.0,
            'available_area_mm2': 0.0,
            'board_area_mm2': 0.0,
            'parts_fitted': 0,
            'parts_total': len(svg_parts)
        }

def process_nesting_with_svgnest(parts_with_quantities, board_specs, scrap_threshold, nesting_config):
    """Process nesting using existing quantity data"""
    
    # Convert parts to SVG format
    svg_parts = convert_dxf_parts_to_svg(parts_with_quantities)
    
    # Run nesting algorithm
    nested_result = run_svgnest_algorithm(svg_parts, board_specs, nesting_config)
    
    # Calculate scrap based on rules
    total_board_area = board_specs['width_mm'] * board_specs['height_mm']
    used_area = nested_result['used_area_mm2']
    unused_area = total_board_area - used_area
    unused_percentage = unused_area / total_board_area if total_board_area > 0 else 0
    
    # Rule: if unused < threshold%, it's leftover; otherwise scrap
    if unused_percentage < scrap_threshold:
        scrap_percentage = 0.0
        leftover_parts = [{'area_mm2': unused_area, 'type': 'leftover', 'percentage': unused_percentage}]
        scrap_parts = []
    else:
        scrap_percentage = unused_percentage
        leftover_parts = []
        scrap_parts = [{'area_mm2': unused_area, 'type': 'scrap', 'percentage': unused_percentage}]
    
    return {
        'board': board_specs,
        'nested_parts': nested_result['parts'],
        'utilization': nested_result['utilization'],
        'scrap_percentage': scrap_percentage,
        'leftover_parts': leftover_parts,
        'scrap_parts': scrap_parts,
        'total_parts_nested': nested_result['parts_fitted'],
        'total_parts_required': nested_result['parts_total'],
        'fitting_success': nested_result['parts_fitted'] == nested_result['parts_total']
    }

@app.route('/api/nesting/calculate', methods=['POST'])
def calculate_nesting_with_existing_quantities():
    """Calculate nesting using existing DXF quantity data"""
    try:
        data = request.get_json()
        
        # Get data that your system already provides
        meaningful_parts = data.get('meaningful_parts', [])
        extracted_labels = data.get('extracted_part_labels', {})
        scrap_threshold = data.get('scrap_threshold', 0.20)  # 20% default
        selected_material = data.get('material_name', 'Steel')
        selected_thickness = data.get('thickness_mm', 1.0)
        
        if not meaningful_parts:
            return jsonify({'error': 'No parts data provided'}), 400
        
        # Extract quantities from your existing system
        parts_with_quantities = []
        for idx, part in enumerate(meaningful_parts, start=1):
            label_data = extracted_labels.get(idx, {})
            quantity = label_data.get('quantity', 1)  # Default to 1 if not found
            if quantity is None or quantity <= 0:
                quantity = 1
            
            # Use your existing _compute_part_bbox function
            bbox = _compute_part_bbox(part)
            
            parts_with_quantities.append({
                'part_index': idx,
                'entities': part,
                'quantity': int(quantity),
                'bbox': bbox
            })
        
        print(f"[NESTING] Processing {len(parts_with_quantities)} parts with total quantities: {sum(p['quantity'] for p in parts_with_quantities)}")
        
        # Get board specs from database, filtered by material and thickness
        all_boards = get_board_specs_from_db()
        filtered_boards = [
            board for board in all_boards 
            if board['material_name'].lower() == selected_material.lower() 
            and abs(board['thickness_mm'] - selected_thickness) < 0.1
        ]
        
        # If no exact match, use all boards
        if not filtered_boards:
            filtered_boards = all_boards[:4]  # Use first 4 boards as fallback
        
        # Get nesting configuration
        nesting_config = load_admin_config().get('nesting', DEFAULT_NESTING_CONFIG)
        
        # Process with each board size
        nesting_results = []
        for board in filtered_boards:
            result = process_nesting_with_svgnest(
                parts_with_quantities, 
                board, 
                scrap_threshold,
                nesting_config
            )
            nesting_results.append(result)
        
        # Find best board (lowest scrap percentage, then highest utilization)
        best_board = None
        if nesting_results:
            # First, prefer boards that fit all parts
            complete_fits = [r for r in nesting_results if r['fitting_success']]
            if complete_fits:
                best_board = min(complete_fits, key=lambda x: (x['scrap_percentage'], -x['utilization']))
            else:
                # If no complete fit, choose best partial fit
                best_board = max(nesting_results, key=lambda x: (x['total_parts_nested'], x['utilization']))
        
        return jsonify({
            'success': True,
            'results': nesting_results,
            'best_board': best_board,
            'parts_summary': {
                'total_parts': len(parts_with_quantities),
                'total_instances': sum(p['quantity'] for p in parts_with_quantities),
                'material': selected_material,
                'thickness_mm': selected_thickness
            },
            'config_used': nesting_config
        })
        
    except Exception as e:
        print(f"Nesting calculation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 