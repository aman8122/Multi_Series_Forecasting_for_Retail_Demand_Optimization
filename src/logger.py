import logging
import os
from datetime import datetime

# Log file name with timestamp
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Logs folder path define
logs_path=os.path.join(os.getcwd(), "artifacts", "logs")

# create logs folder exits or not
os.makedirs(logs_path, exist_ok=True)

# Final log file path
LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

# Logger configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)