# Debt Classifier

## Description

I was bored and wanted to create a synthesized dataset for a binary classification algorithm to learn.
One thing led to another and I ended up creating a whole model to classify whether a person is in debt or not.
As a result, I created a dataset with 100000 samples and 6 features. The model is a simple feedforward neural network with 2 hidden layers.

## Installation

I mean, you can clone the repository and run the project. But why would you want to do that? This is just a fun project I created in my free time. You can use the code as a reference for your own projects, though.

```bash
# Clone the repository
git clone https://github.com/devsimsek/debt-classifier.git

# Navigate to the project directory
cd debt-classifier

# Install dependencies
pip install -r requirements.txt
```

## Usage

Instructions on how to use the project.

```bash
# Navigate to the project directory
cd src
# Generate the dataset
python synthetize_data.py

# Train the model
python train.py

# Run simulated tests
python test.py

# Interact with the model
python interact.py
```

## Contributing

Contributions are always welcome! Please create a pull request if you would like to contribute to the project.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

(You know the drill. You've done this a million times before. I assume, right?)

## License

This project is licensed under the MIT License - see the [LICENSE](https://devsimsek.mit-license.org) file for details.
(Ah, the good old MIT License. Wanna break from the ads?)

## Acknowledgements

- Do whatever you want with the code. I don't care. Just don't use it for evil purposes, or do. I'm not your dad.
- If you have any questions, chatgpt is always there for you. (I'm not responsible for any advice given by chatgpt. It's a bot, for God's sake.)
- If you want to chat with me, you can find me on Twitter. (I'm not responsible for any advice given by me. I'm a human, for God's sake.)
- Used OneHotEncoder from sklearn.
- Used Keras for the neural network.
- Using tflite for model conversion. (That's better right?)
- Don't forget to drink water. (I'm not responsible for any advice given by me. I'm a human, for God's sake.)
