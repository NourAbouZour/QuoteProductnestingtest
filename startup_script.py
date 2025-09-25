#!/usr/bin/env python3
"""
Startup Script for DXF Quotation Application
============================================

This script handles the proper startup of the Flask application with all necessary
initialization, error handling, and production-ready configuration.

Features:
- Database connection testing
- Dependency validation
- Graceful error handling
- Production and development modes
- Windows Server 2022 compatibility

Author: ChatGPT for Nour
Python: 3.8+
"""

import os
import sys
import time
import traceback
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'flask', 'ezdxf', 'numpy', 'matplotlib', 'shapely', 
        'werkzeug', 'jinja2', 'openpyxl', 'pydantic'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} - MISSING")
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    
    return True

def check_database_connection():
    """Test database connection."""
    try:
        from DatabaseConfig import test_connection
        print("🔍 Testing database connection...")
        if test_connection():
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def create_required_directories():
    """Create required directories if they don't exist."""
    directories = ['uploads', 'temp', 'logs', 'cache', 'build']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory: {directory}")

def check_file_permissions():
    """Check if we have proper file permissions."""
    test_file = "temp/startup_test.tmp"
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✅ File permissions OK")
        return True
    except Exception as e:
        print(f"❌ File permission error: {e}")
        return False

def start_application():
    """Start the Flask application."""
    try:
        print("\n" + "="*50)
        print("🚀 Starting DXF Quotation Application")
        print("="*50)
        
        # Import and start the app
        from app import app
        
        print("✅ Flask application loaded successfully")
        print("🌐 Server starting on http://0.0.0.0:5000")
        print("📱 Access the application at: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the Flask development server
        app.run(
            debug=False,  # Set to False for production
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main startup function."""
    print("="*60)
    print("🔧 DXF Quotation Application Startup")
    print("="*60)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python executable: {sys.executable}")
    print()
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Step 3: Create required directories
    print("\n📁 Creating required directories...")
    create_required_directories()
    
    # Step 4: Check file permissions
    print("\n🔐 Checking file permissions...")
    if not check_file_permissions():
        print("❌ File permission check failed")
        sys.exit(1)
    
    # Step 5: Test database connection
    print("\n🗄️ Testing database connection...")
    if not check_database_connection():
        print("⚠️  Database connection failed, but continuing...")
        print("   The application may not work properly without database access")
    
    # Step 6: Start the application
    print("\n🚀 All checks passed! Starting application...")
    time.sleep(1)  # Brief pause for readability
    start_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        traceback.print_exc()
        sys.exit(1)
