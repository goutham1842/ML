# Taxi Fare Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Chicago_Taxi_Trips.csv")

# Display first 5 rows
print("First Five Rows:")
print(df.head())

# Display column names
print("\nColumns in Dataset:")
print(df.columns)

# Select features and target
# Modify these names if they differ in your dataset
features = ['Trip Miles', 'Trip Seconds']
target = 'Fare'

# Remove missing values
df = df[features + [target]].dropna()

# Define X and y
X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation")
print("----------------")
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE      :", mean_absolute_error(y_test, y_pred))
print("RMSE     :", mean_squared_error(y_test, y_pred) ** 0.5)

# Regression equation
print("\nIntercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(feature, ":", coef)

sample = pd.DataFrame({
    'Trip Miles': [5.0],
    'Trip Seconds': [900]
})

prediction = model.predict(sample)
print("\nPredicted Fare for Sample Trip = $", prediction[0])

