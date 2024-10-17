# R-and-D-SUBMISSION
# Linear Regression from Scratch

This project implements a simple linear regression model from scratch using Python, demonstrating the gradient descent algorithm.

## Dataset

We use a dataset saved in a CSV file named `salary_data.csv` with the following format:

YearsExperience,Salary 
1,40000 2,50000 3,60000 4,80000 5,110000


## Implementation

### Required Libraries

Make sure you have the following libraries installed:

``bash
pip install pandas numpy matplotlib
Code
Here’s the complete implementation of linear regression using gradient descent:

python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('salary_data.csv')
X = data['YearsExperience'].values
y = data['Salary'].values

# Add bias (intercept) term
X_b = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for the intercept

# Hyperparameters
learning_rate = 0.01
n_iterations = 1000

# Initialize parameters (weights)
theta = np.random.randn(2)  # Random initialization

# Gradient Descent
for iteration in range(n_iterations):
    predictions = X_b.dot(theta)
    errors = predictions - y
    gradients = 2 / len(y) * X_b.T.dot(errors)  # Gradient calculation
    theta -= learning_rate * gradients  # Update parameters

# Model parameters
intercept, slope = theta

# Make predictions
y_pred = X_b.dot(theta)

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_pred - y) ** 2)

# Print parameters and MSE
print(f"Intercept: {intercept}, Slope: {slope}")
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Years of Experience vs Salary')
plt.legend()
plt.show()

# Loading the Dataset: 
The dataset is loaded using pandas.
# Adding Bias Term: 
A column of ones is added to account for the intercept in the regression equation.
# Gradient Descent:
Parameters are initialized randomly. The parameters are updated iteratively based on the gradient of the cost function (Mean Squared Error).
# Calculating MSE: 
The Mean Squared Error is computed to evaluate model performance.
# Plotting: 
The original data points and the fitted regression line are visualized using matplotlib.
Expected Outcome
When you run the code with the provided dataset, you should see:

The model parameters (intercept and slope) printed out.
The Mean Squared Error (MSE) displayed.
A plot showing the data points and the fitted regression line.
