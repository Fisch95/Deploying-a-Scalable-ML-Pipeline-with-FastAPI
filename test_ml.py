import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, inference
from train_model import train, test, X_train, y_train

# TODO: implement the first test. Change the function name and input as needed
def test_dataset_sizes():
    """
    Tests the train and test sets for the expected sample sizes.
    """
    expected_train_size = 26049
    expected_test_size = 6512

    assert len(train) == expected_train_size, f"Expected training dataset size: {expected_train_size}, but got: {len(train)}"
    assert len(test) == expected_test_size, f"Expected test dataset size: {expected_test_size}, but got: {len(test)}"


# TODO: implement the second test. Change the function name and input as needed
def test_model_uses_expected_algorithm():
    """
    Tests the train_model function to ensure it uses the proper algorithm
    """
    model = train_model(X_train, y_train)
    
    assert isinstance(model, LogisticRegression), "Expected model to be a LogisticRegression"


# TODO: implement the third test. Change the function name and input as needed
def test_inference_returns_expected_result():
    """
    Tests the inference function to ensure it returns the expected results
    """
    model = LogisticRegression()
    model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))  # Dummy training for example

    # Sample input data
    X = np.array([[1, 2], [3, 4]])  # Adjust shape as necessary
    
    # Call the inference function
    result = inference(model, X)
    
    # Check if the result is a NumPy array (or the expected type)
    assert isinstance(result, np.ndarray), "Expected result to be a NumPy array"
