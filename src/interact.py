import tensorflow as tf
import pandas as pd
import joblib

def load_model():
    model = tf.keras.models.load_model("build/latest.keras")
    return model

def predict(model, input, preprocessor):
    input = pd.DataFrame([input], columns=['Balance', 'Employment', 'Sex', 'Age', 'Credit Score', 'Education'])
    input = preprocessor.transform(input)
    return model.predict(input)[0][0] > 0.5

def main():
    model = load_model()
    preprocessor = joblib.load("build/preprocessor.joblib")
    should_continue = True
    while should_continue:
        user_input = input("Enter a comma-separated list of features (Balance, Employment, Sex, Age, Credit Score, Education): ")
        input_data = [float(x) for x in user_input.split(",")]
        target = ((input_data[0] > 5000) & (input_data[4] > 600) & (input_data[3] > 25))
        prediction = predict(model, input_data, preprocessor)
        print(f"Predicted: {prediction}, Actual: {target}")
        user_input = input("Do you want to continue? (y/n): ")
        should_continue = user_input.lower() == "y"

if __name__ == "__main__":
    main()
