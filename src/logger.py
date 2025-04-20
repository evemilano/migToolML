"""
Logging configuration and utilities for the URL Migration Tool.
"""
import logging
from logging.handlers import RotatingFileHandler
from config.config import LOG_CONFIG

def setup_logger(name='migTool'):
    """
    Set up logging configuration with both file and console handlers.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Se il logger ha già degli handler, non aggiungerne di nuovi
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, LOG_CONFIG['level']))
    
    # Create formatters
    formatter = logging.Formatter(LOG_CONFIG['format'])
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_CONFIG['log_file'],
        maxBytes=LOG_CONFIG['max_bytes'],
        backupCount=LOG_CONFIG['backup_count']
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_loading(library_name, logger):
    """
    Log library loading information.
    
    Args:
        library_name (str): Name of the library being loaded
        logger (logging.Logger): Logger instance
    """
    logger.info(f"Loading {library_name}")

def log_error(error, logger, context=None):
    """
    Log error information with context.
    
    Args:
        error (Exception): The error that occurred
        logger (logging.Logger): Logger instance
        context (str, optional): Additional context about the error
    """
    if context:
        logger.error(f"{context}: {str(error)}")
    else:
        logger.error(str(error))

def log_warning(message, logger, context=None):
    """
    Log warning information with context.
    
    Args:
        message (str): Warning message
        logger (logging.Logger): Logger instance
        context (str, optional): Additional context about the warning
    """
    if context:
        logger.warning(f"{context}: {message}")
    else:
        logger.warning(message)

def log_info(message, logger, context=None):
    """
    Log information with context.
    
    Args:
        message (str): Information message
        logger (logging.Logger): Logger instance
        context (str, optional): Additional context about the information
    """
    if context:
        logger.info(f"{context}: {message}")
    else:
        logger.info(message) 