import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('traffic_data.csv')

# Preprocess the data
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# Assuming 'traffic_flow' is the column with traffic flow data
traffic_flow = df['traffic_flow'].values

# Normalize the traffic flow data
scaler = MinMaxScaler(feature_range=(0, 1))
traffic_flow_scaled = scaler.fit_transform(traffic_flow.reshape(-1, 1))

# Create a time series dataset for the RNN
def create_dataset(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 3  # for example, using 3 days to predict the 4th day
X, y = create_dataset(traffic_flow_scaled, n_steps)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Make predictions
y_pred = model.predict(X_test)

# Invert scaling to get actual values
y_test_actual = scaler.inverse_transform(y_test)
y_pred_actual = scaler.inverse_transform(y_pred)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual Traffic Flow')
plt.plot(y_pred_actual, label='RNN Predicted Traffic Flow')
plt.title('Traffic Flow Predictions using RNNs')
plt.xlabel('Day of the Month')
plt.ylabel('Traffic Flow (Vehicles)')
plt.legend()
plt.show()
