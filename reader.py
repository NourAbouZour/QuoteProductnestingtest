#!/usr/bin/env python3
"""
DXF File Visualizer - Standalone Debug Tool

This script provides a simple way to visualize DXF files for debugging purposes.
It reads DXF files and displays them using matplotlib, showing all entities
and their properties.

Usage:
    python reader.py [dxf_file_path]
    
If no file path is provided, it will try to read 'abdo_70.dxf' in the current directory.
"""

import sys
import os
import ezdxf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Tuple, Optional


class DXFVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
    def read_dxf(self, file_path: str) -> Optional[ezdxf.document.Drawing]:
        """Read a DXF file and return the document object."""
        try:
            doc = ezdxf.readfile(file_path)
            print(f"✓ Successfully loaded DXF file: {file_path}")
            print(f"  - AutoCAD version: {doc.dxfversion}")
            print(f"  - File encoding: {doc.encoding}")
            return doc
        except Exception as e:
            print(f"✗ Error reading DXF file: {e}")
            return None
    
    def get_drawing_bounds(self, doc: ezdxf.document.Drawing) -> Tuple[float, float, float, float]:
        """Get the drawing bounds from the DXF file."""
        try:
            # Try to get bounds from header
            extmin = doc.header.get('$EXTMIN', (0, 0, 0))
            extmax = doc.header.get('$EXTMAX', (100, 100, 0))
            
            x_min, y_min = extmin[0], extmin[1]
            x_max, y_max = extmax[0], extmax[1]
            
            print(f"  - Drawing bounds: ({x_min:.2f}, {y_min:.2f}) to ({x_max:.2f}, {y_max:.2f})")
            return x_min, y_min, x_max, y_max
            
        except Exception as e:
            print(f"  - Warning: Could not get bounds from header: {e}")
            return 0, 0, 100, 100
    
    def plot_entity(self, entity, color='blue', linewidth=1):
        """Plot a single DXF entity."""
        try:
            dxftype = entity.dxftype()
            
            if dxftype == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                           color=color, linewidth=linewidth)
                
            elif dxftype == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                circle = patches.Circle((center[0], center[1]), radius, 
                                      fill=False, color=color, linewidth=linewidth)
                self.ax.add_patch(circle)
                
            elif dxftype == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                
                # Convert angles to radians and ensure proper direction
                start_rad = np.radians(start_angle)
                end_rad = np.radians(end_angle)
                
                # Create arc points
                theta = np.linspace(start_rad, end_rad, 100)
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                self.ax.plot(x, y, color=color, linewidth=linewidth)
                
            elif dxftype == 'POLYLINE' or dxftype == 'LWPOLYLINE':
                points = list(entity.get_points())
                if points:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    self.ax.plot(x_coords, y_coords, color=color, linewidth=linewidth)
                    
            elif dxftype == 'TEXT':
                insert = entity.dxf.insert
                text = entity.dxf.text
                height = getattr(entity.dxf, 'height', 1.0)
                self.ax.text(insert[0], insert[1], text, 
                           fontsize=height*10, color=color, ha='center', va='center')
                
            elif dxftype == 'MTEXT':
                insert = entity.dxf.insert
                text = entity.text
                height = getattr(entity.dxf, 'char_height', 1.0)
                self.ax.text(insert[0], insert[1], text, 
                           fontsize=height*10, color=color, ha='left', va='bottom')
                
            elif dxftype == 'DIMENSION':
                # Plot dimension lines
                deftext = entity.dxf.defpoint
                text_midpoint = entity.dxf.text_midpoint
                if hasattr(entity.dxf, 'defpoint2'):
                    defpoint2 = entity.dxf.defpoint2
                    self.ax.plot([deftext[0], defpoint2[0]], [deftext[1], defpoint2[1]], 
                               color='red', linewidth=1, alpha=0.7)
                
            elif dxftype == 'INSERT':
                # Handle block insertions
                insert_point = entity.dxf.insert
                block_name = entity.dxf.name
                self.ax.plot(insert_point[0], insert_point[1], 'ro', markersize=5)
                self.ax.text(insert_point[0], insert_point[1], f"BLOCK:{block_name}", 
                           fontsize=8, color='red', ha='center', va='bottom')
                
            else:
                # For other entity types, try to get any points
                if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
                    start = entity.dxf.start
                    end = entity.dxf.end
                    self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                               color='gray', linewidth=1, alpha=0.5)
                    
        except Exception as e:
            print(f"    - Warning: Could not plot {dxftype}: {e}")
    
    def visualize_dxf(self, file_path: str):
        """Main function to visualize a DXF file."""
        print(f"Reading DXF file: {file_path}")
        
        # Read the DXF file
        doc = self.read_dxf(file_path)
        if not doc:
            return
        
        # Get modelspace
        msp = doc.modelspace()
        
        # Get drawing bounds
        x_min, y_min, x_max, y_max = self.get_drawing_bounds(doc)
        
        # Set plot limits with some padding
        padding = max((x_max - x_min), (y_max - y_min)) * 0.1
        self.ax.set_xlim(x_min - padding, x_max + padding)
        self.ax.set_ylim(y_min - padding, y_max + padding)
        
        # Count entities by type
        entity_counts = {}
        
        print("\nProcessing entities...")
        for entity in msp:
            dxftype = entity.dxftype()
            entity_counts[dxftype] = entity_counts.get(dxftype, 0) + 1
            
            # Plot the entity
            self.plot_entity(entity)
        
        # Print entity statistics
        print(f"\nEntity Statistics:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"  - {entity_type}: {count}")
        
        # Set up the plot
        self.ax.set_title(f'DXF Visualization: {os.path.basename(file_path)}')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', label='Lines/Polylines'),
            plt.Line2D([0], [0], color='green', label='Circles/Arcs'),
            plt.Line2D([0], [0], color='red', label='Dimensions/Blocks'),
            plt.Line2D([0], [0], color='gray', label='Other Entities')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        print(f"\n✓ Visualization complete!")
        print(f"  - Total entities: {sum(entity_counts.values())}")
        print(f"  - Drawing dimensions: {x_max-x_min:.2f} x {y_max-y_min:.2f}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the DXF visualizer."""
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "abdo_70.dxf"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        print("Usage: python reader.py [dxf_file_path]")
        return
    
    # Create visualizer and show the DXF
    visualizer = DXFVisualizer()
    visualizer.visualize_dxf(file_path)


if __name__ == "__main__":
    main() 