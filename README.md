# Experiment 3: Logistic Regression for Classification


# Aim

To implement Logistic Regression for classification and evaluate the model performance using accuracy, confusion matrix, and classification report.


# Algorithm

1. Start the program.
2. Import the required libraries such as NumPy, Pandas, Matplotlib, and Scikit-learn.
3. Load the dataset and separate the input features X and target variable y.
4. Split the dataset into training and testing sets using an 80:20 ratio.
5. Standardize the features using StandardScaler.
6. Train the Logistic Regression model using LogisticRegression.
7. Predict the output labels using the test data.
8. Evaluate the model using accuracy, confusion matrix, and classification report.
9. Display the evaluation results.
10. Stop the program.
Formula / Theory
Sigmoid Function:
\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Where,
\[
z=\beta_0+\beta_1x_1+\cdots+\beta_px_p
\]

The sigmoid function converts predicted values into probabilities between 0 and 1.

# Program

# Step 1: Import the required libraries

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

# Step 2: Load the dataset
data = load_breast_cancer()

X = data.data

y = data.target

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Step 5: Train the Logistic Regression model
model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# Step 6: Predict the output
y_pred = model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Step 8: Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 9: Generate classification report
report = classification_report(y_test, y_pred)

# Step 10: Display the results
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# Output

<img width="547" height="339" alt="image" src="https://github.com/user-attachments/assets/7a907c68-a62c-4f9a-a8ca-43ddcbf2a13b" />

# Result

Thus, Logistic Regression was successfully implemented using Python and Scikit-learn. The model was trained and tested using the Breast Cancer dataset. Its performance was evaluated using accuracy, confusion matrix, and classification report, achieving approximately 97.37% accuracy.
