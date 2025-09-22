"""
DXF entity flattening module.

This module provides functionality to convert complex DXF entities (curves, polylines, etc.)
into simple line segments for cost estimation purposes.
"""

from typing import List, Tuple, Optional, Union
import math
import numpy as np
import ezdxf
from ezdxf.math import Vec2, BSpline, ConstructionRay
from ezdxf.entities import DXFGraphic


class GeometryError(Exception):
    """Custom exception for geometry processing errors."""
    pass


def to_segments(entity: DXFGraphic, tol: float = 0.05) -> List[Vec2]:
    """
    Convert a DXF entity to a list of line segments.
    
    Args:
        entity: The DXF entity to flatten
        tol: Maximum chord-height error tolerance in drawing units (default: 0.05mm)
    
    Returns:
        List of Vec2 points representing line segments
        
    Raises:
        GeometryError: If the entity cannot be processed
    """
    try:
        dxftype = entity.dxftype()
        
        if dxftype == 'LINE':
            return _line_to_segments(entity)
        elif dxftype == 'LWPOLYLINE':
            return _lwpolyline_to_segments(entity, tol)
        elif dxftype == 'POLYLINE':
            return _polyline_to_segments(entity, tol)
        elif dxftype == 'ARC':
            return _arc_to_segments(entity, tol)
        elif dxftype == 'CIRCLE':
            return _circle_to_segments(entity, tol)
        elif dxftype == 'ELLIPSE':
            return _ellipse_to_segments(entity, tol)
        elif dxftype == 'SPLINE':
            return _spline_to_segments(entity, tol)
        elif dxftype == 'HATCH':
            return _hatch_to_segments(entity, tol)
        else:
            # For unsupported entities, return empty list
            return []
            
    except Exception as e:
        raise GeometryError(f"Failed to process {entity.dxftype()}: {str(e)}")


def _line_to_segments(entity: DXFGraphic) -> List[Vec2]:
    """Convert LINE entity to segments."""
    start = entity.dxf.start
    end = entity.dxf.end
    return [Vec2(start.x, start.y), Vec2(end.x, end.y)]


def _lwpolyline_to_segments(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """Convert LWPOLYLINE entity to segments, handling bulges."""
    points = list(entity.get_points())
    if len(points) < 2:
        return []
    
    segments = []
    for i in range(len(points) - 1):
        start_point = Vec2(points[i][0], points[i][1])
        end_point = Vec2(points[i + 1][0], points[i + 1][1])
        
        # Check if this segment has a bulge (curved)
        if hasattr(entity, 'bulge') and entity.bulge(i) != 0:
            # Convert bulge to arc segments
            arc_segments = _bulge_to_arc_segments(start_point, end_point, entity.bulge(i), tol)
            segments.extend(arc_segments)
        else:
            # Straight line segment
            segments.extend([start_point, end_point])
    
    # Handle closed polyline
    if entity.closed:
        start_point = Vec2(points[-1][0], points[-1][1])
        end_point = Vec2(points[0][0], points[0][1])
        
        if hasattr(entity, 'bulge') and entity.bulge(len(points) - 1) != 0:
            arc_segments = _bulge_to_arc_segments(start_point, end_point, entity.bulge(len(points) - 1), tol)
            segments.extend(arc_segments)
        else:
            segments.extend([start_point, end_point])
    
    return segments


def _polyline_to_segments(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """Convert POLYLINE entity to segments."""
    segments = []
    vertices = list(entity.vertices)
    
    for i in range(len(vertices) - 1):
        start_vertex = vertices[i]
        end_vertex = vertices[i + 1]
        
        start_point = Vec2(start_vertex.dxf.location.x, start_vertex.dxf.location.y)
        end_point = Vec2(end_vertex.dxf.location.x, end_vertex.dxf.location.y)
        
        # Check for bulge
        if hasattr(start_vertex.dxf, 'bulge') and start_vertex.dxf.bulge != 0:
            arc_segments = _bulge_to_arc_segments(start_point, end_point, start_vertex.dxf.bulge, tol)
            segments.extend(arc_segments)
        else:
            segments.extend([start_point, end_point])
    
    # Handle closed polyline
    if entity.closed and len(vertices) > 2:
        start_vertex = vertices[-1]
        end_vertex = vertices[0]
        
        start_point = Vec2(start_vertex.dxf.location.x, start_vertex.dxf.location.y)
        end_point = Vec2(end_vertex.dxf.location.x, end_vertex.dxf.location.y)
        
        if hasattr(start_vertex.dxf, 'bulge') and start_vertex.dxf.bulge != 0:
            arc_segments = _bulge_to_arc_segments(start_point, end_point, start_vertex.dxf.bulge, tol)
            segments.extend(arc_segments)
        else:
            segments.extend([start_point, end_point])
    
    return segments


def _arc_to_segments(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """Convert ARC entity to segments."""
    center = entity.dxf.center
    radius = entity.dxf.radius
    start_angle = math.radians(entity.dxf.start_angle)
    end_angle = math.radians(entity.dxf.end_angle)
    
    # Ensure end_angle > start_angle
    if end_angle <= start_angle:
        end_angle += 2 * math.pi
    
    # Calculate number of segments based on tolerance
    arc_length = radius * (end_angle - start_angle)
    num_segments = max(8, int(arc_length / (2 * math.sqrt(2 * radius * tol - tol * tol))))
    
    angles = np.linspace(start_angle, end_angle, num_segments + 1)
    segments = []
    
    for angle in angles:
        x = center.x + radius * np.cos(angle)
        y = center.y + radius * np.sin(angle)
        segments.append(Vec2(x, y))
    
    return segments


def _circle_to_segments(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """Convert CIRCLE entity to segments."""
    center = entity.dxf.center
    radius = entity.dxf.radius
    
    # Calculate number of segments based on tolerance
    circumference = 2 * math.pi * radius
    num_segments = max(16, int(circumference / (2 * math.sqrt(2 * radius * tol - tol * tol))))
    
    angles = np.linspace(0, 2 * math.pi, num_segments + 1)
    segments = []
    
    for angle in angles:
        x = center.x + radius * np.cos(angle)
        y = center.y + radius * np.sin(angle)
        segments.append(Vec2(x, y))
    
    return segments


def _ellipse_to_segments(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """Convert ELLIPSE entity to segments using ezdxf's built-in flattening."""
    try:
        # Use ezdxf's built-in flattening method
        points = list(entity.flattening(tol))
        if points:
            return [Vec2(p.x, p.y) for p in points]
        else:
            # Fallback: manual ellipse approximation
            return _ellipse_manual_approximation(entity, tol)
    except Exception as e:
        print(f"Error flattening ellipse: {e}")
        # Fallback: manual ellipse approximation
        return _ellipse_manual_approximation(entity, tol)


def _spline_to_segments(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """Convert SPLINE entity to segments using ezdxf's built-in flattening."""
    try:
        # Use ezdxf's built-in flattening method
        points = list(entity.flattening(tol))
        if points:
            return [Vec2(p.x, p.y) for p in points]
        else:
            # Fallback: use control points as approximation
            return _spline_control_points_approximation(entity)
    except Exception as e:
        print(f"Error flattening spline: {e}")
        # Fallback: use control points as approximation
        return _spline_control_points_approximation(entity)


def _ellipse_manual_approximation(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """Manual ellipse approximation using parametric equations."""
    try:
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio
        start_param = entity.dxf.start_param
        end_param = entity.dxf.end_param
        
        # Ensure end_param > start_param
        if end_param <= start_param:
            end_param += 2 * math.pi
        
        # Calculate number of segments based on tolerance
        arc_length = abs(end_param - start_param) * max(major_axis.magnitude(), major_axis.magnitude() * ratio)
        num_segments = max(16, int(arc_length / (2 * math.sqrt(2 * tol))))
        
        params = np.linspace(start_param, end_param, num_segments + 1)
        segments = []
        
        for param in params:
            # Parametric ellipse equation
            x = center.x + major_axis.x * math.cos(param) + major_axis.y * ratio * math.sin(param)
            y = center.y + major_axis.y * math.cos(param) - major_axis.x * ratio * math.sin(param)
            segments.append(Vec2(x, y))
        
        return segments
    except Exception as e:
        print(f"Error in manual ellipse approximation: {e}")
        return []


def _spline_control_points_approximation(entity: DXFGraphic) -> List[Vec2]:
    """Approximate spline using control points."""
    try:
        control_points = entity.control_points
        if len(control_points) >= 2:
            segments = []
            for point in control_points:
                segments.append(Vec2(point.x, point.y))
            return segments
        else:
            return []
    except Exception as e:
        print(f"Error in spline control points approximation: {e}")
        return []

def _hatch_to_segments(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """Convert HATCH entity to segments (outline only)."""
    segments = []
    
    try:
        # Get hatch boundaries
        for path in entity.paths:
            if hasattr(path, 'vertices'):
                vertices = list(path.vertices)
                if len(vertices) >= 2:
                    for i in range(len(vertices) - 1):
                        start_point = Vec2(vertices[i].x, vertices[i].y)
                        end_point = Vec2(vertices[i + 1].x, vertices[i + 1].y)
                        segments.extend([start_point, end_point])
                    
                    # Close the path if needed
                    if path.is_closed:
                        start_point = Vec2(vertices[-1].x, vertices[-1].y)
                        end_point = Vec2(vertices[0].x, vertices[0].y)
                        segments.extend([start_point, end_point])
        
        return segments
        
    except Exception as e:
        # If hatch processing fails, return empty list
        return []


def _bulge_to_arc_segments(start_point: Vec2, end_point: Vec2, bulge: float, tol: float) -> List[Vec2]:
    """
    Convert a bulge (curved segment) to arc segments.
    
    The bulge value represents the tangent of 1/4 of the included angle.
    bulge = tan(angle/4)
    """
    if abs(bulge) < 1e-6:  # Nearly straight line
        return [start_point, end_point]
    
    # Calculate arc parameters
    chord_vector = end_point - start_point
    chord_length = chord_vector.magnitude()
    
    if chord_length < 1e-6:  # Zero length chord
        return [start_point]
    
    # Calculate included angle from bulge
    included_angle = 4 * math.atan(abs(bulge))
    
    # Calculate radius
    radius = chord_length / (2 * np.sin(included_angle / 2))
    
    # Calculate center
    chord_midpoint = (start_point + end_point) / 2
    perpendicular = Vec2(-chord_vector.y, chord_vector.x).normalize()
    
    # Distance from chord midpoint to center
    center_distance = radius * np.cos(included_angle / 2)
    
    if bulge > 0:
        center = chord_midpoint + perpendicular * center_distance
    else:
        center = chord_midpoint - perpendicular * center_distance
    
    # Calculate start and end angles
    start_angle = math.atan2(start_point.y - center.y, start_point.x - center.x)
    end_angle = math.atan2(end_point.y - center.y, end_point.x - center.x)
    
    # Ensure proper angle direction
    if bulge > 0:
        if end_angle <= start_angle:
            end_angle += 2 * math.pi
    else:
        if start_angle <= end_angle:
            start_angle += 2 * math.pi
    
    # Calculate number of segments
    arc_length = radius * abs(end_angle - start_angle)
    num_segments = max(4, int(arc_length / (2 * math.sqrt(2 * radius * tol - tol * tol))))
    
    angles = np.linspace(start_angle, end_angle, num_segments + 1)
    segments = []
    
    for angle in angles:
        x = center.x + radius * np.cos(angle)
        y = center.y + radius * np.sin(angle)
        segments.append(Vec2(x, y))
    
    return segments 