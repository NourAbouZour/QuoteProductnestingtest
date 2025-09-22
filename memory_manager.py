"""
Memory Management Module for Windows Server 2022 Deployment
Handles memory monitoring, cleanup, and optimization for 6 concurrent users
"""

import psutil
import gc
import os
import time
import threading
from typing import Dict, Any, Optional
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory usage and cleanup for the application"""
    
    def __init__(self, max_memory_mb: int = 2048, cleanup_threshold: float = 0.8):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.cleanup_callbacks = []
        
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self.monitor_thread.start()
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self):
        """Background memory monitoring"""
        while self.monitoring:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_percent = memory_mb / self.max_memory_mb
                
                if memory_percent > self.cleanup_threshold:
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB ({memory_percent:.1%})")
                    self._trigger_cleanup()
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(30)
    
    def _trigger_cleanup(self):
        """Trigger cleanup procedures"""
        logger.info("Triggering memory cleanup")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Run registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback error: {e}")
    
    def register_cleanup_callback(self, callback):
        """Register a cleanup callback function"""
        self.cleanup_callbacks.append(callback)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }

# Global memory manager instance
memory_manager = MemoryManager()

def memory_cleanup(func):
    """Decorator to ensure memory cleanup after function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Force cleanup after function execution
            gc.collect()
    return wrapper

def check_memory_limit(func):
    """Decorator to check memory limits before function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        usage = memory_manager.get_memory_usage()
        if usage['rss_mb'] > memory_manager.max_memory_mb * 0.9:
            logger.warning("Memory limit approaching, triggering cleanup")
            memory_manager._trigger_cleanup()
        
        return func(*args, **kwargs)
    return wrapper
