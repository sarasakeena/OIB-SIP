# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
# Replace 'sales_data.csv' with your actual file name
data = pd.read_csv(r'C:\Users\HP\Desktop\OIB-SIP\salesprediction\dataset\Advertising.csv')

# Step 2: Explore the data
print("First 5 rows of the dataset:")
print(data.head())

print("\nChecking for missing values:")
print(data.isnull().sum())

# Optional: Visualize relationships
sns.pairplot(data)
plt.show()

# Step 3: Prepare features and target
# Example columns: 'TV', 'Radio', 'Newspaper' are inputs (X); 'Sales' is output (y)
X = data[['TV', 'Radio', 'Newspaper']]   # input features
y = data['Sales']                        # target variable

# Step 4: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 8: Visualize actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()
