# Copyright (C) devsimsek, smskSoft AIWorks 2025
import numpy as np
import pandas as pd
import os
import time

np.random.seed(61)

n_samples = 1000000 # 1 million samples

# Features
balance = np.random.uniform(0, 10000, n_samples) # Balance between 0 to 10K
employment = np.random.randint(0, 2, n_samples) # 0: unemployed, 1: employed
sex = np.random.randint(0, 2, n_samples) # 0: female, 1: male
age = np.random.randint(18, 70, n_samples) # Age between 18 and 70
credit_score = np.random.uniform(300, 850, n_samples) # Credit score between 300 and 850
education = np.random.randint(0, 3, n_samples) # 0: low, 1: medium, 2: high

# Target
# Can user pay the debt? (1: for yes, 0: for no)
# If the balance is high, credit score is within good range, age is moderate, they can pay the debt.
target = ((balance > 5000) & (credit_score > 600) & (age > 25)).astype(int)

# Create a Dataframe
df = pd.DataFrame({
    'Balance': balance,
    'Employment': employment,
    'Sex': sex,
    'Age': age,
    'Credit Score': credit_score,
    'Education': education,
    'Target': target
})

# Save the data
# Check the dataset folder, if it has latest.csv, rename it to dataset_{timestamp}.csv
if 'latest.csv' in os.listdir('dataset'):
    os.rename('dataset/latest.csv', f'dataset/dataset_{time.time()}.csv')
df.to_csv('dataset/latest.csv', index=False)
