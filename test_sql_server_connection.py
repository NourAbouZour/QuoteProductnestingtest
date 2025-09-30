#!/usr/bin/env python3
"""
Test script for SQL Server customer search functionality
Run this to verify your SQL Server connection and search functionality
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_connection():
    """Test SQL Server connection"""
    try:
        import pyodbc
        
        print("Testing SQL Server connection...")
        
        # Test direct connection first
        server = os.environ.get('SQL_SERVER_HOST', '192.168.1.6')
        database = os.environ.get('SQL_SERVER_DB', 'SOCDB_EDMOND_BARADEI')
        username = os.environ.get('SQL_SERVER_USER', 'bot')
        password = os.environ.get('SQL_SERVER_PASSWORD', 'sqlbot@naggiar1')
        
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=no;TrustServerCertificate=yes"
        
        conn = pyodbc.connect(conn_str, timeout=10)
        print("Direct connection successful!")
        
        # Test customer search
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 3 FileName FROM CustomerFile")
        customers = cursor.fetchall()
        print(f"Found {len(customers)} sample customers:")
        for customer in customers:
            print(f"  - {customer[0]}")
        
        conn.close()
        is_connected = True
        
        if is_connected:
            print("SQL Server connection successful!")
            
            # Test search functionality
            print("\nTesting customer search...")
            test_search_term = "test"  # Change this to a term that might exist in your database
            
            # Test search with direct connection
            conn = pyodbc.connect(conn_str, timeout=10)
            cursor = conn.cursor()
            cursor.execute("SELECT TOP 5 FileName, FileNumber FROM CustomerFile WHERE FileName LIKE ?", f"%{test_search_term}%")
            search_results = cursor.fetchall()
            conn.close()
            
            print(f"Found {len(search_results)} customers matching '{test_search_term}'")
            
            if search_results:
                print("\nSample customer data:")
                for i, (name, number) in enumerate(search_results[:3]):  # Show first 3 results
                    print(f"  {i+1}. {name} (File: {number})")
            else:
                print("No customers found. This might be normal if the database is empty or the search term doesn't match.")
                
        else:
            print("SQL Server connection failed!")
            print("\nTroubleshooting tips:")
            print("1. Check your environment variables (SQL_SERVER_HOST, SQL_SERVER_DB, etc.)")
            print("2. Verify SQL Server is running and accessible")
            print("3. Ensure ODBC Driver 17 for SQL Server is installed")
            print("4. Check network connectivity and firewall settings")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install pyodbc>=4.0.0")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nCheck your configuration and try again.")

def show_config():
    """Show current configuration"""
    print("Current SQL Server Configuration:")
    print(f"  Host: {os.environ.get('SQL_SERVER_HOST', 'Not set')}")
    print(f"  Database: {os.environ.get('SQL_SERVER_DB', 'Not set')}")
    print(f"  User: {os.environ.get('SQL_SERVER_USER', 'Not set')}")
    print(f"  Password: {'*' * len(os.environ.get('SQL_SERVER_PASSWORD', '')) if os.environ.get('SQL_SERVER_PASSWORD') else 'Not set'}")
    print(f"  Trusted Connection: {os.environ.get('SQL_SERVER_TRUSTED', 'Not set')}")

if __name__ == "__main__":
    print("SQL Server Customer Search Test")
    print("=" * 50)
    
    show_config()
    print()
    
    test_connection()
    
    print("\n" + "=" * 50)
    print("To configure your connection, set these environment variables:")
    print("   SQL_SERVER_HOST=your_server_name")
    print("   SQL_SERVER_DB=SOCDB_EDMOND_BARADEI")
    print("   SQL_SERVER_USER=your_username")
    print("   SQL_SERVER_PASSWORD=your_password")
    print("   SQL_SERVER_TRUSTED=no")
