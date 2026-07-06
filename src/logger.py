import logging
import os
from datetime import datetime

# Log file ka naam (date aur time ke saath)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Logs folder ka path
logs_path = os.path.join(os.getcwd(), "artifacts", "logs")

# Folder create karega agar pehle se nahi hai
os.makedirs(logs_path, exist_ok=True)

# Final log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Logger configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)