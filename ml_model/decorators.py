import time
import logging
from functools import wraps

logging.basicConfig(
    filename = "ml_logs.log",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

from ml_model.exceptions import MLException

def log_time(func):
    """
    Logs the execution time of any function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info(f" starting '{func.__name__}'...")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logging.info(f"Finished '{func.__name__}' in {elapsed:.3f}s")
            print(f"[TIMER] '{func.__name__}' executed in {elapsed:.3f}s")
            return result
        
        except Exception as e:
            logging.error(f"Error in '{func.__name__}': {str(e)}")
            raise e
        
    return wrapper



def handle_errors(func):
    """
    handles exceptions and logs them
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        try:
            return func(*args, **kwargs)
        
        except MLException as mle:
            print(f"{mle.message}")
            logging.warning(f"MLException in '{func.__name__}': {mle.message}")
            logging.exception(mle)
            raise

        except Exception as e:      # for the generic/unknown error
            print(f"Unexpected error in '{func.__name__}': {str(e)}")
            logging.error(f"Unexpected error in '{func.__name__}': {e}")
            logging.exception(e)
            raise

    return wrapper




def log_action(func):
    """
    Logs method entry and exit
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Entering: {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Exiting: {func.__name__}")
        return result
    
    return wrapper