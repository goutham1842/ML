# Body Weight Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Create dataset
data = {
    'Weight': [79, 69, 73, 95, 82, 55, 69, 71, 64, 69],
    'Height': [1.80, 1.68, 1.82, 1.70, 1.87, 1.55, 1.50, 1.78, 1.67, 1.64],
    'Age': [35, 39, 25, 60, 27, 18, 89, 42, 16, 52],
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Male',
               'Female', 'Female', 'Female', 'Female', 'Female']
}

# Create DataFrame
df = pd.DataFrame(data)

print("Dataset")
print(df)

# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Define independent and dependent variables
X = df[['Height', 'Age', 'Gender']]
y = df['Weight']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nActual Weights")
print(y_test.values)

print("\nPredicted Weights")
print(y_pred)

# Evaluation
print("\nModel Evaluation")
print("----------------")
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE      :", mean_absolute_error(y_test, y_pred))
print("RMSE     :", mean_squared_error(y_test, y_pred) ** 0.5)

# Regression coefficients
print("\nIntercept :", model.intercept_)

print("\nCoefficients")
print("Height :", model.coef_[0])
print("Age    :", model.coef_[1])
print("Gender :", model.coef_[2])

# Predict a new person's weight
new_person = pd.DataFrame({
    'Height': [1.75],
    'Age': [30],
    'Gender': [1]   # Male = 1, Female = 0
})

predicted_weight = model.predict(new_person)

print("\nPredicted Weight =", round(predicted_weight[0], 2), "kg")