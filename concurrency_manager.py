"""
Concurrency Management Module for Windows Server 2022 Deployment
Handles request limiting, session management, and resource protection
"""

import threading
import time
import queue
from typing import Dict, Any, Optional
import logging
from functools import wraps
from datetime import datetime, timedelta
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestLimiter:
    """Limits concurrent requests to prevent resource exhaustion"""
    
    def __init__(self, max_concurrent: int = 6, max_queue_size: int = 10):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.active_requests = 0
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.lock = threading.Lock()
        self.request_times = {}
        
    def acquire_request_slot(self, request_id: str) -> bool:
        """Acquire a request slot, returns True if successful"""
        with self.lock:
            if self.active_requests < self.max_concurrent:
                self.active_requests += 1
                self.request_times[request_id] = time.time()
                logger.info(f"Request {request_id} acquired slot ({self.active_requests}/{self.max_concurrent})")
                return True
            else:
                logger.warning(f"Request {request_id} queued (active: {self.active_requests})")
                return False
    
    def release_request_slot(self, request_id: str):
        """Release a request slot"""
        with self.lock:
            if request_id in self.request_times:
                duration = time.time() - self.request_times[request_id]
                del self.request_times[request_id]
                logger.info(f"Request {request_id} completed in {duration:.2f}s")
            
            if self.active_requests > 0:
                self.active_requests -= 1
                
            # Process queued requests
            if not self.request_queue.empty() and self.active_requests < self.max_concurrent:
                try:
                    queued_request = self.request_queue.get_nowait()
                    queued_request.set()
                except queue.Empty:
                    pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current request limiter status"""
        with self.lock:
            return {
                'active_requests': self.active_requests,
                'max_concurrent': self.max_concurrent,
                'queue_size': self.request_queue.qsize(),
                'max_queue_size': self.max_queue_size,
                'request_times': dict(self.request_times)
            }

class SessionManager:
    """Manages user sessions and prevents session conflicts"""
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.sessions = {}
        self.lock = threading.Lock()
        
    def create_session(self, user_id: str) -> str:
        """Create a new session for a user"""
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'active_requests': 0
            }
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate and update session"""
        with self.lock:
            if session_id not in self.sessions:
                return False
                
            session = self.sessions[session_id]
            if datetime.now() - session['last_activity'] > self.session_timeout:
                del self.sessions[session_id]
                logger.info(f"Session {session_id} expired")
                return False
                
            session['last_activity'] = datetime.now()
            return True
    
    def increment_request_count(self, session_id: str):
        """Increment active request count for session"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['active_requests'] += 1
    
    def decrement_request_count(self, session_id: str):
        """Decrement active request count for session"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['active_requests'] = max(0, 
                    self.sessions[session_id]['active_requests'] - 1)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        with self.lock:
            now = datetime.now()
            expired_sessions = [
                sid for sid, session in self.sessions.items()
                if now - session['last_activity'] > self.session_timeout
            ]
            
            for sid in expired_sessions:
                del self.sessions[sid]
                logger.info(f"Cleaned up expired session {sid}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self.lock:
            return {
                'total_sessions': len(self.sessions),
                'active_sessions': len([s for s in self.sessions.values() 
                                      if s['active_requests'] > 0]),
                'sessions': dict(self.sessions)
            }

# Global instances
request_limiter = RequestLimiter(max_concurrent=6, max_queue_size=10)
session_manager = SessionManager(session_timeout_minutes=30)

def limit_concurrent_requests(func):
    """Decorator to limit concurrent requests"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())
        
        if not request_limiter.acquire_request_slot(request_id):
            return {'error': 'Server busy, please try again later'}, 503
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            request_limiter.release_request_slot(request_id)
    
    return wrapper

def manage_session(func):
    """Decorator to manage user sessions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        from flask import session
        
        session_id = session.get('session_id')
        if not session_id or not session_manager.validate_session(session_id):
            return {'error': 'Invalid or expired session'}, 401
        
        session_manager.increment_request_count(session_id)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            session_manager.decrement_request_count(session_id)
    
    return wrapper

def cleanup_background():
    """Background cleanup task"""
    while True:
        try:
            session_manager.cleanup_expired_sessions()
            time.sleep(300)  # Clean up every 5 minutes
        except Exception as e:
            logger.error(f"Background cleanup error: {e}")
            time.sleep(60)
