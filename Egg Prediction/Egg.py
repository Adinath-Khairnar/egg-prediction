import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load historical egg price data (assuming CSV format)
data = pd.read_csv("egg_dataset_raw_values.tsv")

# Preprocess data (handle missing values, outliers, etc.)
data = data.fillna(method="ffill")  # Example: Fill missing values with previous values

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Model selection (assuming ARIMA based on time series nature)
model = ARIMA(train_data["egg_dataset_raw_values"], order=(1, 1, 1))  # Adjust order as needed
model_fit = model.fit()

# Make predictions on test set
predictions = model_fit.forecast(steps=len(test_data))[0]

# Evaluate model performance
mae = mean_absolute_error(test_data["egg_dataset_raw_values"], predictions)
print("Mean Absolute Error:", mae)

# Visualize predictions (optional)
# import matplotlib.pyplot as plt
# plt.plot(test_data["egg_price"], label="Actual")
# plt.plot(predictions, label="Predicted")
# plt.legend()
# plt.show()