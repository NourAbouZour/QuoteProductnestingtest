#!/usr/bin/env python3
"""
Quick test for Nesting Center - minimal test to verify it works.
Run this first to quickly check if nesting is functional.
"""

import sys
import os

# Add the .cursor/nesting folder to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.cursor'))

def quick_test():
    """Quick test of nesting functionality."""
    print("Quick Nesting Test")
    print("=" * 50)
    
    # Test 1: Import check
    print("\n1. Checking imports...")
    try:
        from nesting_center_api import run_nesting_sync
        print("   [OK] Imports successful")
    except ImportError as e:
        print(f"   [FAIL] Import failed: {e}")
        print("\n   Install missing packages:")
        print("   pip install msal aiohttp geomdl")
        return False
    
    # Test 2: Credentials check
    print("\n2. Checking credentials...")
    try:
        from nesting.NestingCredentials import NestingCredentials
        token = NestingCredentials.acquire_token()
        if token:
            print(f"   [OK] Credentials working (token length: {len(token)})")
        else:
            print("   [FAIL] Failed to get token - check NestingCredentials.py")
            return False
    except Exception as e:
        print(f"   [FAIL] Credentials error: {e}")
        return False
    
    # Test 3: Simple nesting computation
    print("\n3. Running simple nesting test...")
    print("   (This will take 30-60 seconds)")
    
    try:
        parts = [
            {"id": "test1", "length_mm": 200, "width_mm": 100, "quantity": 5}
        ]
        boards = [
            {"id": "board1", "width_mm": 3000, "height_mm": 1500, "quantity": 1}
        ]
        
        result = run_nesting_sync(parts, boards, timeout=60)
        
        if result.get('success'):
            print(f"   [OK] Nesting successful!")
            print(f"   [OK] Nested {result.get('total_parts_nested', 0)} parts")
            print(f"   [OK] Used {result.get('total_plates_used', 0)} plates")
            print("\n[SUCCESS] Nesting is working correctly!")
            return True
        else:
            print(f"   [FAIL] Nesting failed: {result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
