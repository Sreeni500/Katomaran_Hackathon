import os
import logging
from logging.handlers import RotatingFileHandler
import time

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup a logger with the given name
    
    Args:
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file is None:
        log_file = f"logs/{name}_{time.strftime('%Y%m%d')}.log"
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Default application logger
app_logger = setup_logger('katomaran_face_platform')

def log_function_call(logger=None):
    """
    Decorator to log function calls
    
    Args:
        logger: Logger to use (optional)
        
    Returns:
        function: Decorated function
    """
    if logger is None:
        logger = app_logger
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator 