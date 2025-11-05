import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from ml_model.base_model import BaseModel
from ml_model.decorators import handle_errors, log_time, log_action
from ml_model.exceptions import DataNotLoadedError, InvalidInputError


class AttritionModel(BaseModel):
    """Inherits from BaseModel and implements:
    - Data loading
    - Preprocessing and then splitting
    - Training (fit)
    - Prediction
    - Evaluation + Visualization
    also custom exceptions and decorators
    """

    def __init__(self, file_path):
        super().__init__(model_name="AttritionModel")
        self.file_path = file_path
        self.data = None
        self.model = None
        self.model = LogisticRegression(max_iter=500)
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

    @handle_errors
    @log_time
    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully.. Shape: {self.data.shape}")
            return self.data
        
        except FileNotFoundError as e:
            raise DataNotLoadedError(f"File not found at path: {self.file_path}") from e



    @handle_errors
    @log_time
    def preprocess_data(self):
        if self.data is None:
            raise DataNotLoadedError("Call load_data() before preprocess_data()")

        if "LeaveOrNot" not in self.data.columns:
            raise InvalidInputError("Target column 'LeaveOrNot' not found in dataset")

        categorical_cols = ["Education", "City", "Gender", "EverBenched"]
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[col] = self.encoder.fit_transform(self.data[col].astype(str))

        print("Preprocessing completed! Encoded categorical cols")
        return self.data
    
    
    @handle_errors
    def split_data(self, test_size=0.2, random_state=42):
        if self.data is None:
            raise DataNotLoadedError("Call load_data() before split_data().")

        X = self.data.drop("LeaveOrNot", axis=1)
        y = self.data["LeaveOrNot"]

        # simple scaling
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        print(f"Split complete: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test

    
    
    # model train karenge by implementing from BaseModel.fit
    @handle_errors
    @log_time
    def fit(self, X=None, y=None):
        """Train the model on preprocessed data"""
        if X is None or y is None:
            X_train, X_test, y_train, y_test = self.split_data()
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        self.model.fit(X_train, y_train)
        self.mark_trained()
        print("Model training completed")
        y_pred = self.model.predict(X_test)
        return self.evaluate(X_test, y_test, y_pred)


    @handle_errors
    @log_time
    def predict(self, X):
        self.ensure_trained()
        try:
            preds = self.model.predict(X)
            return preds
        
        except Exception as e:

            raise


    @handle_errors
    @log_time
    def evaluate(self, X_test, y_test, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("Evaluation Results:")
        print(f"Accuracy: {acc:.3f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)

        
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Labels")
        plt.tight_layout()
        plt.show()

        return acc