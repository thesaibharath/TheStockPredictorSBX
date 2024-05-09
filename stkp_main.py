# Please Check The .README File For More Info 

# This Is A Project Which Predicts The Stock Price Using The Previous Data 
# For Example, I have Used The .csv File Of Infosys Stock
# You Can Choose Any Other .csv
# Now You Have To Install The Following Packages To Make It Run

import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Here Add Or Drop Any Of The .csv File Of A Stock Location
data = pd.read_csv("Infosys.csv")

# Here We Are Only Selecting The 'Close Price' for prediction
# Sometimes They May Vary Like 'Close' or 'CP', Etc.
# Not Only Close Prices, You Can Select Anything To Predict

dataset = data[['Close Price']].values

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# This Function Is To Create Dataset for LSTM 
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# This Is To Define the time step for LSTM
time_step = 100

# Create the dataset for training the LSTM model
X, y = create_dataset(dataset_scaled, time_step)

# Reshape data for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# To Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# To Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test_orig = scaler.inverse_transform(y_test.reshape(-1,1))

# Evaluating the model
mse = mean_squared_error(y_test_orig, y_pred)
print("Mean Squared Error:", mse)

# This Is To Visualize the results
# Feel Free To Customize
plt.plot(y_test_orig, color='blue', label='Actual Stock Price')
plt.plot(y_pred, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

print(data.columns)

# If You Loved This, Please Star This Repo