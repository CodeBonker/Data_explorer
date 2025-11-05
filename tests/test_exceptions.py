import os
import pytest
import pandas as pd
from ml_model.model_builder import AttritionModel
from ml_model.exceptions import (
    DataNotLoadedError,
    ModelNotTrainedError,
    InvalidInputError
)


@pytest.fixture
def temp_data(tmp_path):
    """Fixture to create a temporary valid dataset"""
    data = pd.DataFrame({
        "Education": ["Bachelors", "Masters", "PhD"],
        "City": ["Delhi", "Mumbai", "Kolkata"],
        "Gender": ["Male", "Female", "Male"],
        "EverBenched": ["No", "Yes", "No"],
        "ExperienceInCurrentDomain": [2, 5, 3],
        "JoiningYear": [2018, 2016, 2019],
        "PaymentTier": [2, 3, 2],
        "Age": [25, 29, 32],
        "LeaveOrNot": [0, 1, 0]
    })
    file_path = tmp_path / "sample_data.csv"
    data.to_csv(file_path, index=False)
    return file_path


def test_preprocess_without_loading_raises_error():
    """Should raise DataNotLoadedError if preprocess called before load_data()"""
    model = AttritionModel("fake_path.csv")
    with pytest.raises(DataNotLoadedError):
        model.preprocess_data()


def test_split_without_loading_raises_error():
    """Should raise DataNotLoadedError if split_data called before load_data()"""
    model = AttritionModel("fake_path.csv")
    with pytest.raises(DataNotLoadedError):
        model.split_data()


def test_predict_before_training_raises_error(temp_data):
    """Should raise ModelNotTrainedError if predict called before fit()"""
    model = AttritionModel(temp_data)
    model.load_data()
    model.preprocess_data()
    X_train, X_test, y_train, y_test = model.split_data()

    with pytest.raises(ModelNotTrainedError):
        model.predict(X_test)


def test_missing_target_column_raises_error(tmp_path):

    df = pd.DataFrame({
        "Education": ["Bachelors", "Masters"],
        "City": ["Delhi", "Mumbai"]
    })
    file_path = tmp_path / "invalid_data.csv"
    df.to_csv(file_path, index=False)

    model = AttritionModel(file_path)
    model.load_data()

    with pytest.raises(InvalidInputError):
        model.preprocess_data()


def test_fit_and_predict_work_correctly(temp_data):
 
    model = AttritionModel(temp_data)
    model.load_data()
    model.preprocess_data()
    acc = model.fit()
    assert isinstance(acc, float)

    X_train, X_test, y_train, y_test = model.split_data()
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
