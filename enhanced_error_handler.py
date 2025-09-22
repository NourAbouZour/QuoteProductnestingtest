"""
Enhanced Error Handling Module for Windows Server 2022 Deployment
Provides comprehensive error handling, logging, and graceful degradation
"""

import logging
import traceback
import sys
import os
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_thresholds = {
            'memory_error': 5,
            'database_error': 10,
            'file_error': 15,
            'processing_error': 20
        }
        self.alert_callbacks = []
    
    def log_error(self, error_type: str, error: Exception, context: Dict[str, Any] = None):
        """Log error with context and check thresholds"""
        error_key = f"{error_type}_{type(error).__name__}"
        
        # Increment error count
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error details
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_class': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'count': self.error_counts[error_key]
        }
        
        logger.error(f"Error {error_key}: {error_details}")
        
        # Check if threshold exceeded
        threshold = self.error_thresholds.get(error_type, 10)
        if self.error_counts[error_key] >= threshold:
            self._trigger_alert(error_key, error_details)
    
    def _trigger_alert(self, error_key: str, error_details: Dict[str, Any]):
        """Trigger alert for repeated errors"""
        alert_message = f"Error threshold exceeded for {error_key}: {error_details['count']} occurrences"
        logger.critical(alert_message)
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(error_key, error_details)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'error_counts': dict(self.error_counts),
            'total_errors': sum(self.error_counts.values()),
            'error_thresholds': dict(self.error_thresholds)
        }
    
    def reset_error_counts(self):
        """Reset error counts"""
        self.error_counts.clear()
        logger.info("Error counts reset")

# Global error handler instance
error_handler = ErrorHandler()

def handle_errors(error_type: str, fallback_response: Any = None):
    """Decorator for comprehensive error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error with context
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                error_handler.log_error(error_type, e, context)
                
                # Return fallback response or re-raise
                if fallback_response is not None:
                    return fallback_response
                else:
                    raise
        return wrapper
    return decorator

def handle_file_errors(func):
    """Specialized error handler for file operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            error_handler.log_error('file_error', e, {'operation': 'file_not_found'})
            return {'error': 'File not found'}, 404
        except PermissionError as e:
            error_handler.log_error('file_error', e, {'operation': 'permission_denied'})
            return {'error': 'Permission denied'}, 403
        except OSError as e:
            error_handler.log_error('file_error', e, {'operation': 'os_error'})
            return {'error': 'File system error'}, 500
        except Exception as e:
            error_handler.log_error('file_error', e, {'operation': 'unknown'})
            return {'error': 'File processing error'}, 500
    return wrapper

def handle_database_errors(func):
    """Specialized error handler for database operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler.log_error('database_error', e, {'operation': 'database_query'})
            return {'error': 'Database error'}, 500
    return wrapper

def handle_memory_errors(func):
    """Specialized error handler for memory-intensive operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError as e:
            error_handler.log_error('memory_error', e, {'operation': 'memory_exhaustion'})
            return {'error': 'Insufficient memory'}, 507
        except Exception as e:
            error_handler.log_error('memory_error', e, {'operation': 'memory_operation'})
            return {'error': 'Memory processing error'}, 500
    return wrapper

def handle_processing_errors(func):
    """Specialized error handler for DXF processing operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler.log_error('processing_error', e, {'operation': 'dxf_processing'})
            return {'error': 'Processing error'}, 500
    return wrapper

def create_error_response(error_message: str, error_code: int = 500, details: Dict[str, Any] = None) -> tuple:
    """Create standardized error response"""
    response = {
        'error': error_message,
        'timestamp': datetime.now().isoformat(),
        'details': details or {}
    }
    return response, error_code

def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Function {func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Function {func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper
