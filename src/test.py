# Test training and evaluation of the model
# Load the model and evaluate it on the test set
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

def load_model():
    model = tf.keras.models.load_model("build/latest.keras")
    return model

def predict(model, input, preprocessor):
    input = pd.DataFrame([input], columns=['Balance', 'Employment', 'Sex', 'Age', 'Credit Score', 'Education'])
    input = preprocessor.transform(input)
    return model.predict(input)[0][0] > 0.5

def load_test_data():
    data = pd.read_csv("dataset/test.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def main():
    model = load_model()
    preprocessor = joblib.load("build/preprocessor.joblib")
    predictions = []
    X, y = load_test_data()
    test_data_count = len(X)
    print(f"Testing with {test_data_count} samples")
    for i, input_d in enumerate(X):
        prediction = predict(model, input_d, preprocessor)
        predictions.append(prediction)
        print(f"Test {i + 1}: Predicted: {prediction}, Actual: {y[i]}")
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
