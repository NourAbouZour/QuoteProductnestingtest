#!/usr/bin/env python3
"""
Test script for Nesting Center API functionality.

This script tests:
1. API availability and status
2. Credentials authentication
3. Nesting computation with sample data
4. Result parsing and SVG generation
"""

import json
import sys
import os
import requests
from typing import Dict, Any

# Add the .cursor/nesting folder to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.cursor'))

def test_credentials():
    """Test if credentials are configured and working."""
    print("\n" + "="*60)
    print("TEST 1: Credentials Check")
    print("="*60)
    
    try:
        from nesting.NestingCredentials import NestingCredentials
        
        print("✓ NestingCredentials module imported successfully")
        
        # Try to acquire a token
        token = NestingCredentials.acquire_token()
        
        if token:
            print(f"✓ Token acquired successfully (length: {len(token)} chars)")
            print(f"  Token preview: {token[:50]}...")
            return True
        else:
            print("✗ Failed to acquire token")
            print("  Check your credentials in NestingCredentials.py")
            return False
            
    except Exception as e:
        print(f"✗ Error testing credentials: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_status(base_url="http://localhost:5000"):
    """Test if the API endpoints are available."""
    print("\n" + "="*60)
    print("TEST 2: API Status Check")
    print("="*60)
    
    try:
        status_url = f"{base_url}/api/nesting-center/status"
        print(f"Checking: {status_url}")
        
        response = requests.get(status_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is available")
            print(f"  Service: {data.get('service', 'Unknown')}")
            print(f"  Endpoints: {', '.join(data.get('endpoints', []))}")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Flask server")
        print("  Make sure the Flask app is running: python app.py")
        return False
    except Exception as e:
        print(f"✗ Error checking API status: {e}")
        return False


def test_direct_nesting():
    """Test nesting computation directly (without Flask)."""
    print("\n" + "="*60)
    print("TEST 3: Direct Nesting Computation")
    print("="*60)
    
    try:
        from nesting_center_api import run_nesting_sync
        
        # Simple test case: 10 small parts on a large board
        test_parts = [
            {
                "id": "part1",
                "length_mm": 200,
                "width_mm": 100,
                "quantity": 10
            },
            {
                "id": "part2",
                "length_mm": 150,
                "width_mm": 75,
                "quantity": 5
            }
        ]
        
        test_boards = [
            {
                "id": "board1",
                "width_mm": 3000,
                "height_mm": 1500,
                "quantity": 2
            }
        ]
        
        print(f"Testing with {len(test_parts)} part types on {len(test_boards)} board types")
        print(f"  Parts: {sum(p['quantity'] for p in test_parts)} total pieces")
        print(f"  Boards: {sum(b['quantity'] for b in test_boards)} available")
        
        print("\nStarting nesting computation (this may take 30-60 seconds)...")
        
        result = run_nesting_sync(
            parts=test_parts,
            boards=test_boards,
            gap_mm=5.0,
            margin_mm=5.0,
            rotation="Fixed90",
            timeout=60
        )
        
        if result.get('success'):
            print("\n✓ Nesting computation completed successfully!")
            print(f"  Parts nested: {result.get('total_parts_nested', 0)}")
            print(f"  Plates used: {result.get('total_plates_used', 0)}")
            print(f"  Layouts generated: {len(result.get('layouts', []))}")
            print(f"  SVG layouts: {len(result.get('svg_layouts', []))}")
            
            # Show statistics
            stats = result.get('statistics', {})
            if stats:
                print(f"\n  Statistics:")
                print(f"    Average parts per plate: {stats.get('average_parts_per_plate', 0):.2f}")
            
            # Show layout details
            for i, layout in enumerate(result.get('layouts', [])):
                print(f"\n  Layout {i+1}:")
                print(f"    Plate index: {layout.get('plate_index', 'N/A')}")
                print(f"    Parts nested: {layout.get('parts_nested', 0)}")
            
            return True
        else:
            print("\n✗ Nesting computation failed")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            return False
            
    except ImportError as e:
        print(f"✗ Cannot import nesting modules: {e}")
        print("  Make sure all dependencies are installed:")
        print("    pip install msal aiohttp geomdl")
        return False
    except Exception as e:
        print(f"✗ Error during nesting computation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoint(base_url="http://localhost:5000"):
    """Test the optimize endpoint via HTTP."""
    print("\n" + "="*60)
    print("TEST 4: API Endpoint Test (via HTTP)")
    print("="*60)
    
    try:
        optimize_url = f"{base_url}/api/nesting-center/optimize"
        
        # Prepare test data
        test_data = {
            "parts": [
                {
                    "id": "test_part_1",
                    "length_mm": 200,
                    "width_mm": 100,
                    "quantity": 5
                }
            ],
            "boards": [
                {
                    "id": "test_board_1",
                    "width_mm": 3000,
                    "height_mm": 1500,
                    "quantity": 1
                }
            ],
            "settings": {
                "gap_mm": 5.0,
                "margin_mm": 5.0,
                "rotation": "Fixed90",
                "timeout": 60
            }
        }
        
        print(f"POST to: {optimize_url}")
        print(f"Payload: {json.dumps(test_data, indent=2)}")
        print("\nSending request (this may take 30-60 seconds)...")
        
        response = requests.post(
            optimize_url,
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print("\n✓ API endpoint test successful!")
                print(f"  Parts nested: {result.get('total_parts_nested', 0)}")
                print(f"  Plates used: {result.get('total_plates_used', 0)}")
                print(f"  SVG layouts: {len(result.get('svg_layouts', []))}")
                
                # Save result to file for inspection
                output_file = "test_nesting_result.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n  Full result saved to: {output_file}")
                
                return True
            else:
                print(f"\n✗ Nesting failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"\n✗ API returned status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Flask server")
        print("  Make sure the Flask app is running: python app.py")
        return False
    except Exception as e:
        print(f"✗ Error testing API endpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("NESTING CENTER API TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Credentials
    results.append(("Credentials", test_credentials()))
    
    # Test 2: API Status (optional - only if Flask is running)
    try:
        results.append(("API Status", test_api_status()))
    except:
        print("\n⚠ Skipping API status test (Flask may not be running)")
        results.append(("API Status", None))
    
    # Test 3: Direct nesting computation
    results.append(("Direct Nesting", test_direct_nesting()))
    
    # Test 4: API endpoint (optional - only if Flask is running)
    try:
        results.append(("API Endpoint", test_api_endpoint()))
    except:
        print("\n⚠ Skipping API endpoint test (Flask may not be running)")
        results.append(("API Endpoint", None))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"{test_name:20} {status}")
    
    passed = sum(1 for _, r in results if r is True)
    total = sum(1 for _, r in results if r is not None)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total and total > 0:
        print("\n🎉 All tests passed! Nesting is working correctly.")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
