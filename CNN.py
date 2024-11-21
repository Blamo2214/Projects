import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

###################################################################################################################################################################################################
# Model set up and architecture
###################################################################################################################################################################################################


# Load data from an Excel file
file_path = r"C:\Users\aubre\Downloads\VIX\UVXYEOD2011toDate.xlsx"
df = pd.read_excel(file_path)

# Set the date column as the index if necessary
# Assuming the date column is named 'Date' and the value column is 'Value'
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Display the first few rows
print(df.head())

# Scaling the data to the range [0, 1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['UVXYRetuns']])  # column name

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length
sequence_length = 60  # sequence length
X, y = create_sequences(data_scaled, sequence_length)

# Define CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)),
    Dropout(0.3),  # Dropout layer to prevent overfitting
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Reshape X to be 3D [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train the model
history = model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Get the last sequence from the training data for forecasting
last_sequence = data_scaled[-sequence_length:]
last_sequence = last_sequence.reshape((1, sequence_length, 1))

# Forecast the next value
forecast = model.predict(last_sequence)
forecast_value = scaler.inverse_transform(forecast)
print("Forecasted value:", forecast_value[0][0])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

n_forecasts = 3294  # Number of steps to forecast
forecasts = []

input_sequence = last_sequence

for _ in range(n_forecasts):
    forecast = model.predict(input_sequence)
    forecast_value = scaler.inverse_transform(forecast)
    forecasts.append(forecast_value[0][0])

    # Update the input sequence with the new forecasted value
    new_sequence = np.append(input_sequence[:, 1:, :], forecast.reshape((1, 1, 1)), axis=1)
    input_sequence = new_sequence

print("Forecasted values:", forecasts)

import matplotlib.pyplot as plt

###################################################################################################################################################################################################
# Export Forcasted Values
###################################################################################################################################################################################################

import pandas as pd

# Assuming `forecasts` is the list of your forecasted values
# And `forecast_dates` is the corresponding list of forecast dates

# Create a DataFrame with the forecasted values and corresponding dates
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted Value': forecasts
})

# Specify the file path for saving the Excel file
output_file_path = r"C:\Users\aubre\Downloads\VIX\UVXYEODForcastedValues.xlsx"

# Export to Excel
forecast_df.to_excel(output_file_path, index=False)

print(f"Forecasted values have been saved to {output_file_path}")


###################################################################################################################################################################################################
# All Data and Forecasted Values
###################################################################################################################################################################################################

# Create a sequence of dates corresponding to the forecasted values
forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_forecasts, freq='D')

# Plot the actual data and forecasted values
plt.figure(figsize=(10, 6))

# Plot actual data
plt.plot(df.index, df['UVXYRetuns'], label='Actual Data', color='blue')

# Plot forecasted values
plt.plot(forecast_dates, forecasts, label='Forecasted Data', color='red', marker='o')

# Add labels and title
plt.title('Actual vs Forecasted Values')
plt.xlabel('Date')
plt.ylabel('UVXYRetuns')

# Add a legend
plt.legend()

# Display the plot
plt.show()

###################################################################################################################################################################################################
# Some Actual Data and Forecasted Values
###################################################################################################################################################################################################

# Display the plot
plt.show()

# Define the number of actual data points to display (e.g., last 100)
n_actual_points = 100

# Get a subset of actual data
actual_data_subset = df['UVXYRetuns'].iloc[-n_actual_points:]

# Create a sequence of dates corresponding to the forecasted values
forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_forecasts, freq='D')

# Plot the actual data subset and all forecasted values
plt.figure(figsize=(10, 6))

# Plot the subset of actual data
plt.plot(actual_data_subset.index, actual_data_subset, label='Actual Data (Last 100)', color='blue')

# Plot forecasted values
plt.plot(forecast_dates, forecasts, label='Forecasted Data', color='red', marker='o')

# Add labels and title
plt.title('Actual (Subset) vs Forecasted Values')
plt.xlabel('Date')
plt.ylabel('UVXYRetuns')

# Add a legend
plt.legend()

# Display the plot
plt.show()


###################################################################################################################################################################################################
# Forecasted Values
###################################################################################################################################################################################################

# Create a sequence of dates corresponding to the forecasted values
forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_forecasts, freq='D')

# Plot only the forecasted values
plt.figure(figsize=(8, 5))

plt.plot(forecast_dates, forecasts, label='Forecasted Data', color='red', marker='o')

# Add title and labels
plt.title('Forecasted Values')
plt.xlabel('Date')
plt.ylabel('Forecasted Value')

# Add a legend
plt.legend()

# Display the plot
plt.show()

#######################################################################################################################################################################
# Forecasted Values Histogram
###################################################################################################################################################################################################


# Plot a histogram of forecasted values
plt.figure(figsize=(8, 5))
plt.hist(forecasts, bins=100, color='skyblue', edgecolor='black')

# Add title and labels
plt.title('Histogram of Forecasted Values')
plt.xlabel('Forecasted Value')
plt.ylabel('Frequency')

# Display the plot
plt.show()

#######################################################################################################################################################################
# Actual Values Histogram
###################################################################################################################################################################################################


# Plot a histogram of the actual data
plt.figure(figsize=(8, 5))
plt.hist(df['UVXYRetuns'], bins=100, color='skyblue', edgecolor='black')  # Adjust bins as needed

# Add title and labels
plt.title('Histogram of Actual Data (UVXY Returns)')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Display the plot
plt.show()

###################################################################################################################################################################################################
# Plot of Training and Validation Loss
###################################################################################################################################################################################################

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

###################################################################################################################################################################################################
# Box Plot of Forecasted Values
###################################################################################################################################################################################################

plt.boxplot(forecasts, vert=False)
plt.xlabel('Forecasted Value')
plt.title('Box Plot of Forecasted Values')
plt.show()

###################################################################################################################################################################################################
# Error Plot (Residual Plot)
###################################################################################################################################################################################################

residuals = df['UVXYRetuns'][-n_forecasts:] - forecasts  # Adjust based on your data

plt.plot(forecast_dates, residuals, marker='o', linestyle='-', color='purple')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.title('Residual Plot (Actual - Forecasted)')
plt.show()

###################################################################################################################################################################################################
# Scatter Plot of Actual vs Forecasted Values
###################################################################################################################################################################################################

plt.scatter(df['UVXYRetuns'][-n_forecasts:], forecasts, color='orange')
plt.plot([min(df['UVXYRetuns'][-n_forecasts:]), max(df['UVXYRetuns'][-n_forecasts:])],
         [min(df['UVXYRetuns'][-n_forecasts:]), max(df['UVXYRetuns'][-n_forecasts:])], 
         color='black', linestyle='--')
plt.xlabel('Actual Value')
plt.ylabel('Forecasted Value')
plt.title('Actual vs Forecasted Scatter Plot')
plt.show()

###################################################################################################################################################################################################
# Scatter Plot of Actual vs Forecasted Values
###################################################################################################################################################################################################

cumulative_forecast = np.cumsum(forecasts)
plt.plot(forecast_dates, cumulative_forecast, color='darkgreen')
plt.xlabel('Date')
plt.ylabel('Cumulative Forecasted Value')
plt.title('Cumulative Sum of Forecasted Values')
plt.show()

###################################################################################################################################################################################################
# Rolling Mean and Standard Deviation Plot
###################################################################################################################################################################################################

forecast_series = pd.Series(forecasts, index=forecast_dates)
rolling_mean = forecast_series.rolling(window=5).mean()
rolling_std = forecast_series.rolling(window=5).std()

plt.plot(forecast_dates, forecasts, label='Forecasted Values', color='red')
plt.plot(forecast_dates, rolling_mean, label='Rolling Mean', color='blue')
plt.plot(forecast_dates, rolling_std, label='Rolling Std Dev', color='green')
plt.title('Rolling Mean and Standard Deviation of Forecasted Values')
plt.xlabel('Date')
plt.legend()
plt.show()

###################################################################################################################################################################################################
# Confidence Interval Plot (Using Prediction Intervals)
###################################################################################################################################################################################################

from sklearn.metrics import mean_squared_error
import numpy as np

# Generate predictions on the validation set
val_predictions = model.predict(X)  # X_val should be your validation data
val_errors = y - val_predictions[:, 0]  # Residuals (assuming y_val is 1D)

# Calculate standard deviation of errors
error_std = np.std(val_errors)

confidence_level = 1.96  # for a 95% confidence interval

# Assuming `forecasts` is a NumPy array of forecasted values
forecasts = np.array(forecasts)
lower_bounds = forecasts - (confidence_level * error_std)
upper_bounds = forecasts + (confidence_level * error_std)

# Assuming you have computed `lower_bounds` and `upper_bounds` as confidence intervals
plt.plot(forecast_dates, forecasts, label='Forecasted Values', color='red')
plt.fill_between(forecast_dates, lower_bounds, upper_bounds, color='gray', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Forecast with Confidence Intervals')
plt.legend()
plt.show()

###################################################################################################################################################################################################
# Autocorrelation Plot of Forecasted Values
###################################################################################################################################################################################################

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(forecasts, lags=20)
plt.title('Autocorrelation of Forecasted Values')
plt.show()

###################################################################################################################################################################################################
# Heatmap of Prediction Errors Across Time and Lag
###################################################################################################################################################################################################

import seaborn as sns

# Assuming `errors` is a 2D array of errors over different lags
errors_matrix = np.array([forecasts - df['UVXYRetuns'][-n_forecasts:].values])  # Example placeholder
sns.heatmap(errors_matrix, cmap="coolwarm", annot=False)
plt.xlabel("Lag")
plt.ylabel("Time")
plt.title("Heatmap of Prediction Errors")
plt.show()