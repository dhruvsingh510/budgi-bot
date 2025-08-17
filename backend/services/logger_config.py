#!/usr/bin/env python3
"""
Centralized logging configuration for Personal Finance Bot services.

This module sets up file-based logging with IST timestamps for all services.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
import pytz

class ISTFormatter(logging.Formatter):
    """Custom formatter to display timestamps in IST."""
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.ist_tz = pytz.timezone('Asia/Kolkata')
    
    def formatTime(self, record, datefmt=None):
        """Convert UTC time to IST."""
        dt = datetime.fromtimestamp(record.created, tz=self.ist_tz)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S IST')

class ServiceLogger:
    """Centralized logger setup for all services."""
    
    def __init__(self):
        self.logs_dir = self._create_logs_directory()
        self.loggers = {}
        
    def _create_logs_directory(self) -> Path:
        """Create logs directory if it doesn't exist."""
        logs_dir = Path(__file__).parent.parent / "data" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    def get_logger(self, service_name: str) -> logging.Logger:
        """Get or create a logger for a specific service."""
        if service_name in self.loggers:
            return self.loggers[service_name]
        
        # Create logger
        logger = logging.getLogger(f"financebot.{service_name}")
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create file handler for service-specific logs
        service_log_file = self.logs_dir / f"{service_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            service_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Create combined log handler (all services in one file)
        combined_log_file = self.logs_dir / "combined.log"
        combined_handler = logging.handlers.RotatingFileHandler(
            combined_log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=10,
            encoding='utf-8'
        )
        combined_handler.setLevel(logging.INFO)
        
        # Create formatters with IST timestamps
        detailed_formatter = ISTFormatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        simple_formatter = ISTFormatter(
            '%(asctime)s | %(levelname)s | [%(name)s] %(message)s'
        )
        
        # Set formatters
        file_handler.setFormatter(detailed_formatter)
        combined_handler.setFormatter(simple_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(combined_handler)
        
        # Store logger
        self.loggers[service_name] = logger
        
        logger.info(f"=== {service_name.upper()} SERVICE LOGGER INITIALIZED ===")
        
        return logger

# Global logger instance
service_logger = ServiceLogger()

def get_service_logger(service_name: str) -> logging.Logger:
    """Convenience function to get a service logger."""
    return service_logger.get_logger(service_name)