# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING

```
Devloped by: yenyganti prathyusha
Register Number: 212223240187
Date: 23-05-2025
```


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```
Read the Passengers dataset
```
data = pd.read_csv('/content/passengers_with_stronger_fluctuations.csv')
```
Focus on the '#Passengers' column
```
passengers = data[['Passengers']]
```
Display the shape and the first 10 rows of the dataset
```
print("Shape of the dataset:", passengers.shape)
print("First 10 rows of the dataset:")
print(data.head(10))
```
Plot Original Dataset (#Passengers Data)
```
plt.figure(figsize=(12, 6))
plt.plot(data['Passengers'], label='Original data')
plt.title('Passengers_count')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.grid()
plt.show()
```
Moving Average
Perform rolling average transformation with a window size of 5 and 10
```
rolling_mean_5 = passengers_data['Passengers'].rolling(window=5).mean()
rolling_mean_10 = passengers_data['Passengers'].rolling(window=10).mean()
```
Display the first 10 and 20 vales of rolling means with window sizes 5 and 10 respectively
```
rolling_mean_5.head(10)
rolling_mean_10.head(20)
```
Plot Moving Average
```
plt.figure(figsize=(12, 6))
plt.plot(data['Passengers'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Passnegers count Data')
plt.xlabel('Date')
plt.ylabel('Passnegers_count')
plt.legend()
plt.grid()
plt.show()

```

Perform data transformation to better fit the model
```
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

data_monthly = data.resample('MS').sum()
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten()
scaled_data = pd.Series(scaled_array, index=data_monthly.index)

```
Exponential Smoothing
```
# The data seems to have additive trend and multiplicative seasonality
scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, yes even zeros
x=int(len(scaled_data)*0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_data.var()),scaled_data.mean()
```
Make predictions for one fourth of the data
```
model = ExponentialSmoothing(data['Passengers'], trend='add', seasonal='mul', seasonal_periods=12).fit()

predictions = model.forecast(steps=int(len(data) / 4))

ax = data['Passengers'].plot(figsize=(10, 6))
predictions.plot(ax=ax)

ax.legend(["data_monthly", "predictions"])
ax.set_xlabel('Months')
ax.set_ylabel('Number of monthly passengers')
ax.set_title('Prediction')
plt.show()

```

### OUTPUT:

Original data:

![image](https://github.com/user-attachments/assets/4400e3b7-bb37-4e3c-af31-69c15d9fa3a0)


![image](https://github.com/user-attachments/assets/5f07a7da-85a2-42f6-b1a3-35fa5bc99e4d)


Moving Average:- (Rolling)

window(5):

![image](https://github.com/user-attachments/assets/71041f4b-a1bf-4989-a60e-c436b8291ee1)




window(10):

![image](https://github.com/user-attachments/assets/406ad41a-f4f9-4be9-a9e7-aa9c03b30241)


plot:

![image](https://github.com/user-attachments/assets/45c55d9d-b7a2-4069-b4c8-f55644c9dd18)



Exponential Smoothing:-

Test:

![image](https://github.com/user-attachments/assets/051f8188-0b5e-4639-9b17-80f6472fb6d9)


Performance:

![image](https://github.com/user-attachments/assets/3f8ff7d9-cbf8-4e17-be18-d113a702a73f)


Prediction:

![image](https://github.com/user-attachments/assets/be91b7c9-f978-44ab-a74a-9d6ec04b5902)



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
