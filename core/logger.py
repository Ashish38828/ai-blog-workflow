import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger with standard enterprise formatting."""
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't already have handlers (prevents duplicate logs)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Standard console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Format: [2026-04-25 20:28:41] [INFO] [agents.researcher] - Message here
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
