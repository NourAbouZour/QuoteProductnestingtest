"""
Geometry processing module for DXF entity flattening.
"""

from .flatten import to_segments, GeometryError

__all__ = ["to_segments", "GeometryError"] 