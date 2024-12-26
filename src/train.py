import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

def load_dataset(path):
    """Loads the dataset from the specified CSV path."""
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """Preprocesses the dataset by separating features and target and applying transformations."""
    X = df.drop('Target', axis=1)
    y = df['Target']

    # Define the column transformer for scaling numerical features and one-hot encoding categorical ones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Balance', 'Age', 'Credit Score']), # Normalize numerical features
            ('cat', OneHotEncoder(), ['Employment', 'Sex', 'Education'])  # One-hot encode categorical features
        ]
    )

    # Apply transformation to X and return
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor

def train_model(X, y, preprocessor):
    """Trains the model using the provided features and target labels."""
    model = Sequential([
        Dense(16, input_dim=X.shape[1], activation='relu'),  # The input_dim should match the transformed feature count
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%, Loss: {test_loss}")

    # Save the Keras model
    model.save("build/latest.keras")

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('build/latest.tflite', 'wb') as f:
        f.write(tflite_model)

    # Save the preprocessor
    joblib.dump(preprocessor, 'build/preprocessor.joblib')

if __name__ == '__main__':
    # Load and preprocess the dataset
    df = load_dataset('dataset/latest.csv')
    X_transformed, y, preprocessor = preprocess_data(df)

    # Train the model
    train_model(X_transformed, y, preprocessor)
