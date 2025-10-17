house_price_predictor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset (built-in)
housing = fetch_california_housing(as_frame=True)
df_house = housing.frame

# Add the target variable (Price) to the DataFrame
df_house['Price'] = housing.target

print("✅ California Housing data loaded successfully.")
print(df_house.head())
# Identify Features (X) and Target (y)
# X contains all columns except 'Price'
X = df_house.drop('Price', axis=1)
# y is the target price
y = df_house['Price']

# Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("✅ Data split completed.")# Instantiate the Linear Regression model
model_reg = LinearRegression()

# Train the model using the training data
model_reg.fit(X_train, y_train)

print("✅ Linear Regression model trained successfully.")
# Make predictions on the test set
y_pred = model_reg.predict(X_test)

# 1. Calculate the R-squared (R2): Explains the variance in the target variable (closer to 1 is better)
r2 = r2_score(y_test, y_pred)

# 2. Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# 3. Calculate the Root Mean Squared Error (RMSE) manually using NumPy
rmse = np.sqrt(mse)

print("\n✅ Model evaluation complete.")
print(f"\n⭐ R-squared (Model Fit): {r2:.4f}")
print(f"⭐ Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"(RMSE measures the average prediction error, unit is 100k USD)")

# Display the first 5 predictions vs. actual prices
results = pd.DataFrame({'Actual Price': y_test.head(), 'Predicted Price': y_pred[:5]})
print("\nFirst 5 Predictions vs. Actual Prices:")
print(results)
