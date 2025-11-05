from abc import ABC, abstractmethod
import logging 
from datetime import datetime
from ml_model.decorators import log_time, handle_errors

# setting up simple logging to track 
logging.basicConfig(
    filename = "ml_logs.log",
    level = logging.INFO, #sirf abhi ke liye
    format = "%(asctime)s - %(levelname)s - %(message)s"
)
 

class BaseModel(ABC):
    """
    Abstract base class for all ML models
    Defines the std interface: fit, predict, evaluate nd save
    """

    def __init__(self, model_name = "BaseModel"):
        self.model_name = model_name
        self.created_at = datetime.now()
        logging.info(f"initialised model: {self.model_name}")


    @abstractmethod
    @log_time
    @handle_errors
    def fit(self, X, y):
        pass


    @abstractmethod
    @log_time
    @handle_errors
    def predict(self, X):
        pass


    @abstractmethod
    @log_time
    @handle_errors
    def evaluate(self, X, y):
        pass


    def save(self, path):
        import joblib
        joblib.dump(self, path)
        logging.info(f"Saved model {self.mode_name} to {path}")

    
    def info(self):
        print(f"Model name: {self.model_name}")
        print(f"created at: {self.created_at}")

        
