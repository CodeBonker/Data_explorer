
class MLException(Exception):
    """Base class for all custom ML exceptions in the project."""

    def __init__(self, message = None):
        if message is None:
            message = "An error occured in the ML pipeline"

        super().__init__(message)
        self.message = message

class DataNotLoadedError(MLException):
    """Raised when data is expected but not loaded"""

    def __init__(self, message=None):
        if message is None:
            message = "Required dataset not loaded. Call load_data() first or check file path"
        super().__init__(message)


class ModelNotTrainedError(MLException):
    """Raised when prediction or evaluation is attempted before training"""

    def __init__(self, message = None):
        if message is None:
            message = "Model has not been trained. Call fit() before predict() or evaluate()"
        super().__init__(message)

class InvalidInputError(MLException):
    """Raised when input data is invalid"""
    def __init__(self, message=None):
        if message is None:
            message = "Invalid input data supplied"
        super().__init__(message)

class PredictionError(MLException):
    """Raised when something goes wrong during prediction"""
    def __init__(self, message=None):
        if message is None:
            message = "An error occurred during prediction"
        super().__init__(message)
