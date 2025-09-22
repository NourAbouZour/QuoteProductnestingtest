"""
Deployment Configuration for Windows Server 2022
Production-ready configuration for 6 concurrent users
"""

import os
from pathlib import Path

class DeploymentConfig:
    """Production deployment configuration"""
    
    # Server Configuration
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = False
    
    # Security Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-production-secret-key-change-this')
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = Path('uploads')
    TEMP_FOLDER = Path('temp')
    ALLOWED_EXTENSIONS = {'dxf'}
    
    # Database Configuration - Uses existing DatabaseConfig.py settings
    # These will be overridden by the existing DatabaseConfig.py file
    DATABASE_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'QuotationDB',
        'user': 'dxf_user',
        'password': 'DXFanalyzer2024!',
        'pool_size': 15,  # Increased for 6 users
        'max_overflow': 25,
        'pool_pre_ping': True,
        'pool_recycle': 3600,
        'echo': False
    }
    
    # Memory Management
    MAX_MEMORY_MB = 4096  # 4GB limit
    MEMORY_CLEANUP_THRESHOLD = 0.8
    GARBAGE_COLLECTION_INTERVAL = 300  # 5 minutes
    
    # Concurrency Configuration
    MAX_CONCURRENT_REQUESTS = 6
    MAX_QUEUE_SIZE = 15
    SESSION_TIMEOUT_MINUTES = 30
    REQUEST_TIMEOUT_SECONDS = 300  # 5 minutes
    
    # File Processing Configuration
    MAX_PARTS_PER_FILE = 100
    MAX_FILES_PER_BATCH = 10
    IMAGE_DPI = 150
    IMAGE_FORMAT = 'png'
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'app.log'
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Performance Configuration
    ENABLE_CACHING = True
    CACHE_TIMEOUT = 3600  # 1 hour
    ENABLE_COMPRESSION = True
    
    # Monitoring Configuration
    ENABLE_HEALTH_CHECKS = True
    HEALTH_CHECK_INTERVAL = 60  # 1 minute
    ENABLE_METRICS = True
    METRICS_INTERVAL = 300  # 5 minutes
    
    # Cleanup Configuration
    TEMP_FILE_CLEANUP_INTERVAL = 1800  # 30 minutes
    MAX_TEMP_FILE_AGE = 3600  # 1 hour
    ENABLE_AUTO_CLEANUP = True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.TEMP_FOLDER,
            Path('logs'),
            Path('cache')
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            # Set appropriate permissions on Windows
            try:
                os.chmod(directory, 0o755)
            except:
                pass  # Windows doesn't support chmod
    
    @classmethod
    def get_database_url(cls):
        """Get database URL from configuration"""
        config = cls.DATABASE_CONFIG
        return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check directory permissions
        try:
            cls.create_directories()
        except Exception as e:
            errors.append(f"Directory creation failed: {e}")
        
        # Check database connection using existing DatabaseConfig
        try:
            from DatabaseConfig import test_connection
            if not test_connection():
                errors.append("Database connection test failed")
        except Exception as e:
            errors.append(f"Database connection error: {e}")
        
        return errors

# Production environment variables template
PRODUCTION_ENV_TEMPLATE = """
# Windows Server 2022 Production Environment Variables
# Copy these to your environment or .env file

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=QuotationDB
DB_USER=dxf_user
DB_PASSWORD=your_secure_password_here

# Security
SECRET_KEY=your_very_secure_secret_key_here

# Server Configuration
PORT=5000
HOST=0.0.0.0

# Optional: Custom paths
UPLOAD_FOLDER=uploads
TEMP_FOLDER=temp
LOG_FOLDER=logs
"""
