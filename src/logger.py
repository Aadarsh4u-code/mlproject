import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"  # Format to How log file is created
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)  # path for a log file
os.makedirs(logs_path,exist_ok=True) # even there is file keep on appending 

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

# Override the logging default pattern of logs 
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__ == "__main__":
#     logging.info("Logging has started")