import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.logger import logger
i = 0 
logger.info(f"Initialize i = {i}")

import time
while i<10:
    time.sleep(2) # wait for 2 seconds
    logger.info(f"i = {i}")
    i += 1

logger.info(f"Finished i = {i}")
