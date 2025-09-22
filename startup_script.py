"""
Windows Server 2022 Startup Script
Production-ready startup with monitoring and health checks
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from deployment_config import DeploymentConfig
from memory_manager import memory_manager
from concurrency_manager import request_limiter, session_manager, cleanup_background
from enhanced_error_handler import error_handler

# Configure logging
logging.basicConfig(
    level=getattr(logging, DeploymentConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DeploymentConfig.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def validate_environment():
    """Validate environment before startup"""
    logger.info("Validating environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        return False
    
    # Check required modules
    required_modules = [
        'flask', 'ezdxf', 'matplotlib', 'numpy', 'shapely',
        'sqlalchemy', 'psutil', 'requests'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        return False
    
    # Test database connection using existing DatabaseConfig
    try:
        from DatabaseConfig import test_connection
        if not test_connection():
            logger.error("Database connection test failed")
            return False
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False
    
    logger.info("Environment validation passed")
    return True

def setup_monitoring():
    """Setup monitoring and background tasks"""
    logger.info("Setting up monitoring...")
    
    # Start memory monitoring
    memory_manager.start_monitoring()
    
    # Start background cleanup
    cleanup_thread = threading.Thread(target=cleanup_background, daemon=True)
    cleanup_thread.start()
    
    # Register cleanup callbacks
    def cleanup_temp_files():
        """Clean up temporary files"""
        try:
            temp_dir = DeploymentConfig.TEMP_FOLDER
            if temp_dir.exists():
                for file_path in temp_dir.iterdir():
                    if file_path.is_file():
                        file_age = time.time() - file_path.stat().st_mtime
                        if file_age > DeploymentConfig.MAX_TEMP_FILE_AGE:
                            file_path.unlink()
                            logger.info(f"Cleaned up old temp file: {file_path}")
        except Exception as e:
            logger.error(f"Temp file cleanup error: {e}")
    
    memory_manager.register_cleanup_callback(cleanup_temp_files)
    
    # Register error alert callback
    def error_alert_callback(error_key, error_details):
        """Handle error alerts"""
        logger.critical(f"ALERT: {error_key} - {error_details['count']} occurrences")
        # Here you could add email notifications, etc.
    
    error_handler.register_alert_callback(error_alert_callback)
    
    logger.info("Monitoring setup complete")

def create_health_check_endpoint(app):
    """Add health check endpoint to Flask app"""
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        try:
            # Check database connection
            from DatabaseConfig import test_connection
            db_healthy = test_connection()
            
            # Check memory usage
            memory_usage = memory_manager.get_memory_usage()
            memory_healthy = memory_usage['rss_mb'] < DeploymentConfig.MAX_MEMORY_MB * 0.9
            
            # Check request limiter status
            limiter_status = request_limiter.get_status()
            limiter_healthy = limiter_status['active_requests'] < DeploymentConfig.MAX_CONCURRENT_REQUESTS
            
            # Check session manager
            session_stats = session_manager.get_session_stats()
            session_healthy = session_stats['total_sessions'] < 100  # Reasonable limit
            
            overall_healthy = db_healthy and memory_healthy and limiter_healthy and session_healthy
            
            status_code = 200 if overall_healthy else 503
            
            return {
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'checks': {
                    'database': db_healthy,
                    'memory': memory_healthy,
                    'request_limiter': limiter_healthy,
                    'session_manager': session_healthy
                },
                'stats': {
                    'memory_usage_mb': memory_usage['rss_mb'],
                    'active_requests': limiter_status['active_requests'],
                    'total_sessions': session_stats['total_sessions']
                }
            }, status_code
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }, 500
    
    @app.route('/metrics')
    def metrics():
        """Metrics endpoint"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'memory': memory_manager.get_memory_usage(),
                'requests': request_limiter.get_status(),
                'sessions': session_manager.get_session_stats(),
                'errors': error_handler.get_error_stats()
            }
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return {'error': str(e)}, 500

def start_application():
    """Start the Flask application"""
    logger.info("Starting DXF Quotation Application...")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed, exiting")
        sys.exit(1)
    
    # Setup monitoring
    setup_monitoring()
    
    # Import and configure Flask app
    from app import app
    
    # Apply production configuration with performance optimizations
    app.config.update({
        'SECRET_KEY': DeploymentConfig.SECRET_KEY,
        'MAX_CONTENT_LENGTH': DeploymentConfig.MAX_CONTENT_LENGTH,
        'UPLOAD_FOLDER': str(DeploymentConfig.UPLOAD_FOLDER),
        'SESSION_COOKIE_SECURE': False,  # Set to False for HTTP, True for HTTPS
        'SESSION_COOKIE_HTTPONLY': DeploymentConfig.SESSION_COOKIE_HTTPONLY,
        'SESSION_COOKIE_SAMESITE': DeploymentConfig.SESSION_COOKIE_SAMESITE,
        # Performance optimizations
        'SEND_FILE_MAX_AGE_DEFAULT': 3600,  # Cache static files for 1 hour
        'PERMANENT_SESSION_LIFETIME': 1800,  # 30 minutes session timeout
        'JSON_SORT_KEYS': False,  # Don't sort JSON keys for better performance
        'JSONIFY_PRETTYPRINT_REGULAR': False,  # Disable pretty printing
    })
    
    # Add health check endpoints
    create_health_check_endpoint(app)
    
    # Log startup information
    logger.info(f"Application starting on {DeploymentConfig.HOST}:{DeploymentConfig.PORT}")
    logger.info(f"Max concurrent requests: {DeploymentConfig.MAX_CONCURRENT_REQUESTS}")
    logger.info(f"Max memory limit: {DeploymentConfig.MAX_MEMORY_MB}MB")
    logger.info(f"Upload folder: {DeploymentConfig.UPLOAD_FOLDER}")
    logger.info(f"Temp folder: {DeploymentConfig.TEMP_FOLDER}")
    
    try:
        # Start the application
        app.run(
            host=DeploymentConfig.HOST,
            port=DeploymentConfig.PORT,
            debug=DeploymentConfig.DEBUG,
            threaded=True,
            use_reloader=False  # Disable reloader in production
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        logger.info("Shutting down...")
        memory_manager.stop_monitoring()
        logger.info("Shutdown complete")

if __name__ == '__main__':
    start_application()
