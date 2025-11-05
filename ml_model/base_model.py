from abc import ABC, abstractmethod
import logging 
from datetime import datetime
from ml_model.decorators import log_time, handle_errors
from ml_model.exceptions import ModelNotTrainedError, DataNotLoadedError


# setting up simple logging to track 
logging.basicConfig(
    filename = "ml_logs.log",
    level = logging.INFO, #sirf abhi ke liye we can expand later
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
        self._is_trained = False
        logging.info(f"initialised model: {self.model_name}")


    @abstractmethod
    def fit(self, X, y):
        """ train the model"""
        raise NotImplementedError


    @abstractmethod
    def predict(self, X):
        raise NotImplementedError


    @abstractmethod
    def evaluate(self, X, y):
        raise NotImplementedError


    def ensure_trained(self):
        """Call at the beginning of predict/evaluate to guarantee the model was trained"""
        if not self._is_trained:
            logging.error(f"ModelNotTrainedError: tried to use {self.model_name} before training")
            raise ModelNotTrainedError()

    def mark_trained(self):
        """Mark model as trained"""
        self._is_trained = True

    def info(self):
        print(f"Model Name: {self.model_name}")
        print(f"Created At: {self.created_at}")
        print(f"Trained: {self._is_trained}")

    def save(self, path):
        import joblib
        joblib.dump(self, path)
        logging.info(f"Saved model {self.model_name} to {path}")


        
