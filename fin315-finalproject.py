#%%
#Import required libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf  

#%%
#Fetch and preprocess data
btc = yf.Ticker("BTC-USD")
data = btc.history(period="1y")  
#%%
# Preprocess data
data['Date'] = data.index
data.reset_index(drop=True, inplace=True)
data['Daily Return'] = data['Close'].pct_change()
data.dropna(inplace=True)  # Remove NaN values due to the first difference
#%%
#Visualize Data
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='BTC-USD Close Price')
plt.title('Bitcoin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=45)
plt.legend()
plt.show()
#%%
#Build  Simple Predicitve Model (Linear Regression)
data['Prev Close'] = data['Close'].shift(1)
data = data.dropna()  # Drop rows with NaN values due to the shift

X = data[['Prev Close']]  
y = data['Close']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
#%%
#Evaluate the model
# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
#%%
# Plot actual vs. predicted prices
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')  # Line of perfect prediction
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()


