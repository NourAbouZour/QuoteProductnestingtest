"""
Geometry processing module for DXF entity flattening.

This module provides functions for converting DXF entities (including curves) into
polyline approximations for area/perimeter calculations.

Key function:
- entity_to_polyline_points(entity, max_sagitta): Convert any DXF entity to a list
  of (x, y) points, correctly handling bulges (arcs) in LWPOLYLINE entities.
"""

from .flatten import to_segments, GeometryError, entity_to_polyline_points

__all__ = ["to_segments", "GeometryError", "entity_to_polyline_points"] 