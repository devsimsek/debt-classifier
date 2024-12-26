import tensorflow as tf
import pandas as pd
import joblib

def load_model():
    # Load the latest Keras model
    model = tf.keras.models.load_model("build/latest.keras")
    return model

def predict(model, input, preprocessor):
    # Preprocess the input data and make a prediction
    input = pd.DataFrame([input], columns=['Balance', 'Employment', 'Sex', 'Age', 'Credit Score', 'Education'])
    input = preprocessor.transform(input)
    # Return True if the model predicts a positive outcome
    return model.predict(input)[0][0] > 0.5

def main():
    # Load the model and preprocessor
    model = load_model()
    preprocessor = joblib.load("build/preprocessor.joblib")
    # Enter a loop to interact with the model
    should_continue = True
    while should_continue:
        # Get user input and make a prediction
        user_input = input("Enter a comma-separated list of features (Balance, Employment, Sex, Age, Credit Score, Education): ")
        # Convert the input to a list of floats (I know, I need to convert some of the values to int but i'm lazy)
        input_data = [float(x) for x in user_input.split(",")]
        # Check if the target should be positive based on the input data (just for auditing purposes)
        target = ((input_data[0] > 5000) & (input_data[4] > 600) & (input_data[3] > 25))
        # Make a prediction and print the result
        prediction = predict(model, input_data, preprocessor)
        print(f"Predicted: {prediction}, Actual: {target}")
        # Ask the user if they want to continue
        user_input = input("Do you want to continue? (y/n): ")
        should_continue = user_input.lower() == "y"

if __name__ == "__main__":
    # Aww, you are here? Dude, respect my man. You are a real one. I appreciate you.
    main()
