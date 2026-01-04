import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Load the Data ---
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

try:
    df = pd.read_csv('4) house Prediction Data Set.csv', sep='\s+', header=None, names=columns)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: '4) house Prediction Data Set.csv' not found in this folder.")
    exit()

# --- Select Features ---
# 'RM' (Number of rooms) as X (Input)
# 'MEDV' (Price) as y (Output to predict)
X = df[['RM']]
y = df['MEDV']

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Build the Model ---
model = LinearRegression()
model.fit(X_train, y_train) # Training the computer

# --- Make Predictions ---
y_pred = model.predict(X_test)

# --- Calculate Results ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Results ---")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# --- Create the Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual House Prices')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Our Prediction Line')
plt.title('Task 2: House Price vs Number of Rooms')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Price (MEDV)')
plt.legend()
plt.grid(True)

# saving the graph as an image
plt.savefig('task2_result_plot.png')
print("\n✅ Plot saved as 'task2_result_plot.png'")

# Show on your screen
plt.show()