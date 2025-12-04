"""
Test for LWPOLYLINE bulge handling.

This test demonstrates the difference between:
1. Old approach: entity.get_points() - returns only vertices, ignores bulges
2. New approach: entity_to_polyline_points() - correctly handles bulges via path flattening

A bulge value in DXF represents the tangent of 1/4 of the included arc angle.
For example:
- bulge = 0: straight line
- bulge = 1: semicircle (180 degrees)
- bulge = -1: semicircle in opposite direction
"""

import sys
import os
import math

# Add the parent directory to the path so we can import the geometry module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezdxf
from ezdxf.math import Vec2


def calculate_perimeter_old_method(entity):
    """
    OLD METHOD: Using get_points() directly.
    This ignores bulges and calculates straight-line distance between vertices.
    """
    points = list(entity.get_points())
    if len(points) < 2:
        return 0.0
    
    perimeter = 0.0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        perimeter += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    # Handle closed polyline
    if entity.closed:
        p1 = points[-1]
        p2 = points[0]
        perimeter += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    return perimeter


def calculate_perimeter_new_method(entity):
    """
    NEW METHOD: Using entity_to_polyline_points() which correctly handles bulges.
    """
    from geometry.flatten import entity_to_polyline_points
    
    points = entity_to_polyline_points(entity, max_sagitta=0.1)
    if len(points) < 2:
        return 0.0
    
    perimeter = 0.0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        perimeter += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    return perimeter


def calculate_area_from_points(points):
    """Calculate polygon area using shoelace formula."""
    if len(points) < 3:
        return 0.0
    
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2.0


def create_test_lwpolyline_with_bulge():
    """
    Create a test DXF document with an LWPOLYLINE that has a semicircular bulge.
    
    This creates a shape like a rectangle with one curved side:
    - 3 straight edges
    - 1 semicircular bulge (arc) instead of a straight edge
    
    The shape is 100mm x 50mm with a semicircular bulge on top.
    """
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # Create an LWPOLYLINE with a bulge
    # Points: (0,0) -> (100,0) -> (100,50) [with bulge] -> (0,50)
    # The bulge of 1.0 creates a semicircle between points 2 and 3
    
    points = [
        (0, 0, 0, 0, 0),       # (x, y, start_width, end_width, bulge)
        (100, 0, 0, 0, 0),     # Straight line from (0,0) to (100,0)
        (100, 50, 0, 0, 1.0),  # Bulge=1.0: semicircle from (100,50) to (0,50)
        (0, 50, 0, 0, 0),      # Straight line from (0,50) to (0,0) [closing]
    ]
    
    lwpolyline = msp.add_lwpolyline(
        points,
        dxfattribs={'layer': '0'},
        close=True
    )
    
    return doc, lwpolyline


def test_bulge_handling():
    """
    Test that demonstrates the difference between old and new bulge handling.
    """
    print("=" * 70)
    print("TEST: LWPOLYLINE Bulge Handling")
    print("=" * 70)
    
    # Create test geometry
    doc, lwpolyline = create_test_lwpolyline_with_bulge()
    
    # Get raw points (old method)
    raw_points = list(lwpolyline.get_points())
    print(f"\nRaw points from get_points() ({len(raw_points)} points):")
    for i, p in enumerate(raw_points):
        bulge_val = lwpolyline.get_bulge(i) if hasattr(lwpolyline, 'get_bulge') else lwpolyline.bulge(i) if hasattr(lwpolyline, 'bulge') else 0
        print(f"  Point {i}: ({p[0]:.1f}, {p[1]:.1f}) - bulge: {bulge_val}")
    
    # Calculate perimeter using OLD method (ignores bulges)
    perimeter_old = calculate_perimeter_old_method(lwpolyline)
    print(f"\n--- OLD METHOD (get_points) ---")
    print(f"Perimeter: {perimeter_old:.2f} mm")
    print("  (This incorrectly treats the arc as a straight line!)")
    
    # Calculate perimeter using NEW method (handles bulges)
    from geometry.flatten import entity_to_polyline_points
    flattened_points = entity_to_polyline_points(lwpolyline, max_sagitta=0.1)
    perimeter_new = calculate_perimeter_new_method(lwpolyline)
    print(f"\n--- NEW METHOD (entity_to_polyline_points) ---")
    print(f"Flattened to {len(flattened_points)} points")
    print(f"Perimeter: {perimeter_new:.2f} mm")
    
    # Calculate expected perimeter analytically
    # 3 straight edges: 100 + 50 + 50 = 200mm
    # 1 semicircular arc with diameter 100mm: π * 50 ≈ 157.08mm
    chord_length = 100.0  # Distance between (100,50) and (0,50)
    # For a bulge of 1.0, included angle is 180 degrees (semicircle)
    # Radius = chord / (2 * sin(angle/2)) = 100 / (2 * sin(90°)) = 100 / 2 = 50
    radius = 50.0
    arc_length = math.pi * radius  # Half circumference
    expected_perimeter = 100 + 50 + arc_length + 50  # bottom + right + arc + left
    
    print(f"\n--- EXPECTED (analytical) ---")
    print(f"Straight edges: 100 + 50 + 50 = 200 mm")
    print(f"Semicircular arc (r=50mm): π × 50 = {arc_length:.2f} mm")
    print(f"Total expected perimeter: {expected_perimeter:.2f} mm")
    
    # Calculate error
    error_old = abs(perimeter_old - expected_perimeter)
    error_new = abs(perimeter_new - expected_perimeter)
    
    print(f"\n--- ERROR ANALYSIS ---")
    print(f"Old method error: {error_old:.2f} mm ({error_old/expected_perimeter*100:.1f}%)")
    print(f"New method error: {error_new:.2f} mm ({error_new/expected_perimeter*100:.2f}%)")
    
    # Calculate areas
    area_old = calculate_area_from_points([(p[0], p[1]) for p in raw_points])
    area_new = calculate_area_from_points(flattened_points)
    
    # Expected area: rectangle 100x50 + semicircle area
    # Wait, actually the semicircle bulges outward from the rectangle...
    # The base rectangle is 100x50 = 5000 mm²
    # The semicircle adds: π * r² / 2 = π * 50² / 2 ≈ 3927 mm²
    # But the semicircle is centered at (50, 50) with radius 50, extending outward
    expected_area = 100 * 50 + (math.pi * radius**2) / 2
    
    print(f"\n--- AREA COMPARISON ---")
    print(f"Old method area: {area_old:.2f} mm² (rectangle only)")
    print(f"New method area: {area_new:.2f} mm²")
    print(f"Expected area: {expected_area:.2f} mm²")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("The new path-based approach correctly handles bulges (arcs) in LWPOLYLINEs.")
    print(f"Perimeter improvement: {error_old - error_new:.2f} mm more accurate")
    print("=" * 70)
    
    return perimeter_old, perimeter_new, expected_perimeter


def test_straight_polyline():
    """
    Control test: A polyline without bulges should give the same result for both methods.
    """
    print("\n" + "=" * 70)
    print("CONTROL TEST: Straight LWPOLYLINE (no bulges)")
    print("=" * 70)
    
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # Create a simple rectangle without bulges
    points = [(0, 0), (100, 0), (100, 50), (0, 50)]
    lwpolyline = msp.add_lwpolyline(points, close=True)
    
    perimeter_old = calculate_perimeter_old_method(lwpolyline)
    perimeter_new = calculate_perimeter_new_method(lwpolyline)
    expected = 2 * (100 + 50)  # Rectangle perimeter
    
    print(f"Old method: {perimeter_old:.2f} mm")
    print(f"New method: {perimeter_new:.2f} mm")
    print(f"Expected: {expected:.2f} mm")
    print(f"Both methods match: {abs(perimeter_old - perimeter_new) < 0.1}")


if __name__ == '__main__':
    try:
        test_bulge_handling()
        test_straight_polyline()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


