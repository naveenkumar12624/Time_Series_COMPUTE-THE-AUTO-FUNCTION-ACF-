### Developed By: Naveen Kumar S
### Register No: 212221240033
### Date: 

# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)

### AIM:
To Compute the AutoCorrelation Function (ACF) of the NVIDIA's StockPrice dataset and 
to determine the model type to fit the data.

### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.

### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the NVIDIA dataset
data = pd.read_csv('C:/Users/lenovo/Downloads/archive (2)/NVIDIA/NvidiaStockPrice.csv')

# Convert 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Resample data to yearly frequency and take mean of 'Close' prices
data_yearly = data['Close'].resample('Y').mean()

# Convert the data to a numpy array
data_array = data_yearly.values

# Compute mean and variance
mean = np.mean(data_array)
variance = np.var(data_array)

# Normalize the data
normalized_data = (data_array - mean) / np.sqrt(variance)

# Function to compute autocorrelation
def autocorrelation(data, lag):
    return np.corrcoef(data[:-lag], data[lag:])[0, 1]

# Compute autocorrelation for the first 35 lags
lags = range(1, 36)
acf_values = [autocorrelation(normalized_data, lag) for lag in lags]

# Plot the autocorrelation function
plt.figure(figsize=(12, 6))
plt.stem(lags, acf_values, use_line_collection=True)
plt.title('Autocorrelation Function (ACF) for the First 35 Lags')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True)
plt.show()
```

### OUTPUT:
#### AUTO CORRELATION:
![image](https://github.com/user-attachments/assets/44e58aa4-194f-45b5-a690-5211e9b5b7fb)

### RESULT: 
Thus, The python code for implementing auto correlation for NVIDIA's Stock Price is successfully executed.
