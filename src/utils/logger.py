# Module holds the logger configuration for the application

import logging
import os
import sys
from typing import Optional
from .constants import APP_NAME

# Logger configuration for the application.
class LoggerConfig:
    def __init__(
            self, 
            name: str = APP_NAME,
            log_level: str = logging.DEBUG,
            enable_file_logging: bool = False,
            log_dir: Optional[str] = "logs",
    ):
        self.name = name
        self.log_level = log_level
        self.enable_file_logging = enable_file_logging
        self.log_dir = log_dir

# Setup the logger with consistent formatting for the application.
def setup_logger(config: LoggerConfig) -> logging.Logger:
    
    logger = logging.getLogger(config.name)
    logger.setLevel(config.log_level)

    # Create formatter and add it to the handlers
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Clear any existing handlers
    logger.handlers = []

    # Create console handler with a specific log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # For file logging, create a file handler
    if config.enable_file_logging:
        try:
            # Check if the log directory exists, if not create it
            if not os.path.exists(config.log_dir):
                os.makedirs(config.log_dir) 

            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S'
            )
            log_file = os.path.join(config.log_dir, f"{config.name}.log")
            file_handler = logging.FileHandler(log_file)            
            file_handler.setLevel(config.log_level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler) 
            logger.debug(f"File logging enabled. Log file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {str(e)}")
            config.enable_file_logging = False
            logger.debug("File logging disabled due to error.")
    return logger


 # Create a logger instance with default configuration
default_logger_config = LoggerConfig(
    name=APP_NAME,
    log_level=logging.DEBUG,
    enable_file_logging=False,  # Set to True to enable file logging
    log_dir="logs"  # Directory for log files
)
logger = setup_logger(default_logger_config)
