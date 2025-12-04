"""
DXF entity flattening module.

This module provides functionality to convert complex DXF entities (curves, polylines, etc.)
into simple line segments for cost estimation purposes.

IMPORTANT: This module uses ezdxf.path.make_path() + .flattening() as the primary method
for converting entities to points. This correctly handles:
  - LWPOLYLINE bulges (curved segments between vertices)
  - ARC, CIRCLE, ELLIPSE entities
  - SPLINE curves
  - Any other curved geometry

Previously, using entity.get_points() would ONLY return the vertices, completely
ignoring bulge values. This caused:
  - Incorrect perimeter calculations (straight line between arc endpoints)
  - Incorrect area calculations (missing the curved region)
  - Visual rendering issues

The Path.flattening(max_sagitta) method approximates curves with line segments
where max_sagitta is the maximum perpendicular deviation from the true curve.
"""

from typing import List, Tuple, Optional, Union
import math
import numpy as np
import ezdxf
from ezdxf.math import Vec2, BSpline, ConstructionRay
from ezdxf.entities import DXFGraphic
from ezdxf import path as ezdxf_path


class GeometryError(Exception):
    """Custom exception for geometry processing errors."""
    pass


def entity_to_polyline_points(entity: DXFGraphic, max_sagitta: float = 0.1) -> List[Tuple[float, float]]:
    """
    Convert a DXF entity into a list of (x, y) points approximating its geometry.
    
    This function correctly handles LWPOLYLINE bulges and other curved entities
    by using ezdxf's path module instead of raw get_points().
    
    Args:
        entity: The DXF entity to convert
        max_sagitta: Maximum deviation from true curve in drawing units (default: 0.1mm)
                    Smaller values = more points = higher accuracy
    
    Returns:
        List of (x, y) tuples representing the polyline approximation of the entity.
        For closed shapes, the last point will equal the first point.
    
    Supported entity types:
        - LWPOLYLINE (with bulges/arcs)
        - POLYLINE
        - LINE
        - ARC
        - CIRCLE
        - ELLIPSE
        - SPLINE
        - HATCH (boundary paths)
    
    Example:
        >>> points = entity_to_polyline_points(lwpolyline_entity, max_sagitta=0.05)
        >>> perimeter = sum(dist(points[i], points[i+1]) for i in range(len(points)-1))
    """
    dxftype = entity.dxftype()
    
    # Primary method: use ezdxf.path.make_path() + flattening() for curved entities
    # This correctly handles bulges, arcs, splines, etc.
    if dxftype in {'LWPOLYLINE', 'POLYLINE', 'ARC', 'CIRCLE', 'ELLIPSE', 'SPLINE'}:
        try:
            path = ezdxf_path.make_path(entity)
            # flattening() returns an iterator of Vec3 points
            flattened = list(path.flattening(max_sagitta))
            if flattened:
                return [(float(v.x), float(v.y)) for v in flattened]
        except Exception as e:
            # Fall through to fallback methods if path conversion fails
            print(f"Path flattening failed for {dxftype}: {e}, using fallback")
    
    # Handle LINE directly (no curves)
    if dxftype == 'LINE':
        start = entity.dxf.start
        end = entity.dxf.end
        return [(float(start.x), float(start.y)), (float(end.x), float(end.y))]
    
    # Handle HATCH boundaries
    if dxftype == 'HATCH':
        return _hatch_to_points(entity, max_sagitta)
    
    # Fallback for entities with vertices attribute
    if hasattr(entity, 'vertices'):
        try:
            vertices = list(entity.vertices)
            return [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in vertices]
        except Exception:
            pass
    
    # Fallback for entities with get_points() - note: this loses bulge info!
    if hasattr(entity, 'get_points'):
        try:
            points = list(entity.get_points())
            return [(float(p[0]), float(p[1])) for p in points]
        except Exception:
            pass
    
    # Fallback for entities with start/end attributes
    if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
        s = entity.dxf.start
        e = entity.dxf.end
        return [(float(s.x), float(s.y)), (float(e.x), float(e.y))]
    
    return []


def _hatch_to_points(entity: DXFGraphic, max_sagitta: float) -> List[Tuple[float, float]]:
    """Extract boundary points from HATCH entity."""
    all_points = []
    try:
        for boundary_path in entity.paths:
            # Try to convert the boundary path using ezdxf path tools
            try:
                path = ezdxf_path.make_path(boundary_path)
                flattened = list(path.flattening(max_sagitta))
                if flattened:
                    all_points.extend([(float(v.x), float(v.y)) for v in flattened])
            except Exception:
                # Fallback: direct vertex extraction
                if hasattr(boundary_path, 'vertices'):
                    for v in boundary_path.vertices:
                        if hasattr(v, 'x') and hasattr(v, 'y'):
                            all_points.append((float(v.x), float(v.y)))
                        elif isinstance(v, (list, tuple)) and len(v) >= 2:
                            all_points.append((float(v[0]), float(v[1])))
    except Exception as e:
        print(f"Error extracting HATCH boundary: {e}")
    return all_points


def to_segments(entity: DXFGraphic, tol: float = 0.05) -> List[Vec2]:
    """
    Convert a DXF entity to a list of line segments (Vec2 points).
    
    This function uses ezdxf.path.make_path() + flattening() for curved entities,
    which correctly handles LWPOLYLINE bulges and other curved geometry.
    
    Args:
        entity: The DXF entity to flatten
        tol: Maximum chord-height error tolerance in drawing units (default: 0.05mm)
             This is the max_sagitta value for curve flattening.
    
    Returns:
        List of Vec2 points representing line segments
        
    Raises:
        GeometryError: If the entity cannot be processed
    """
    try:
        dxftype = entity.dxftype()
        
        # For curved entities, use the path-based approach which correctly handles bulges
        if dxftype in {'LWPOLYLINE', 'POLYLINE', 'ARC', 'CIRCLE', 'ELLIPSE', 'SPLINE'}:
            return _path_based_to_segments(entity, tol)
        elif dxftype == 'LINE':
            return _line_to_segments(entity)
        elif dxftype == 'HATCH':
            return _hatch_to_segments(entity, tol)
        else:
            # For unsupported entities, return empty list
            return []
            
    except Exception as e:
        raise GeometryError(f"Failed to process {entity.dxftype()}: {str(e)}")


def _path_based_to_segments(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """
    Convert entity to segments using ezdxf.path.make_path() + flattening().
    
    This is the CORRECT way to handle curved entities like LWPOLYLINE with bulges.
    The old approach using entity.get_points() only returned vertices, completely
    ignoring bulge values and creating incorrect geometry.
    """
    try:
        path = ezdxf_path.make_path(entity)
        flattened = list(path.flattening(tol))
        if flattened:
            return [Vec2(v.x, v.y) for v in flattened]
    except Exception as e:
        # Fallback to entity-specific methods if path conversion fails
        dxftype = entity.dxftype()
        print(f"Path flattening failed for {dxftype}: {e}, using fallback")
        
        if dxftype == 'LWPOLYLINE':
            return _lwpolyline_to_segments_fallback(entity, tol)
        elif dxftype == 'POLYLINE':
            return _polyline_to_segments_fallback(entity, tol)
        elif dxftype == 'ARC':
            return _arc_to_segments(entity, tol)
        elif dxftype == 'CIRCLE':
            return _circle_to_segments(entity, tol)
        elif dxftype == 'ELLIPSE':
            return _ellipse_to_segments(entity, tol)
        elif dxftype == 'SPLINE':
            return _spline_to_segments(entity, tol)
    
    return []


def _line_to_segments(entity: DXFGraphic) -> List[Vec2]:
    """Convert LINE entity to segments."""
    start = entity.dxf.start
    end = entity.dxf.end
    return [Vec2(start.x, start.y), Vec2(end.x, end.y)]


def _lwpolyline_to_segments_fallback(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """
    FALLBACK: Convert LWPOLYLINE entity to segments with manual bulge handling.
    
    NOTE: This is only used when ezdxf.path.make_path() fails.
    The path-based approach in _path_based_to_segments() is preferred as it
    correctly handles all curve types using ezdxf's internal geometry.
    """
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


def _polyline_to_segments_fallback(entity: DXFGraphic, tol: float) -> List[Vec2]:
    """
    FALLBACK: Convert POLYLINE entity to segments with manual bulge handling.
    
    NOTE: This is only used when ezdxf.path.make_path() fails.
    The path-based approach in _path_based_to_segments() is preferred.
    """
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