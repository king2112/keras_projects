import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

def create_timeSries(ds, series):
    X, Y =[], []
    for i in range(len(ds)-series - 1):
        item = ds[i:(i+series), 0]
        X.append(item)
        Y.append(ds[i+series, 0])
    return np.array(X), np.array(Y)

series = 7

df = pd.read_csv('absal.csv')

date = df.Date
df.drop('Date', 1)
df['close'] = pd.to_numeric(df['close'], downcast='float')
df = df.close
df = df.values.reshape(len(df), 1)
# ------------------------------- Scaling ------------------------- #
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)

# ----------------------------------------------------------------- #

#split data into train and test
train_size = int(len(df)* 0.7)
test_size = len(df) - train_size

df_train, df_test = df[0:train_size, :], df[train_size:len(df), :]

print('Split data into train and test: ', len(df_train), len(df_test))

x_train , y_train = create_timeSries(df_train , series)
x_test , y_test = create_timeSries(df_test , series)


# ---------------------- Reshaping for LSTM layer ------------------ #
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# ------------------------------------------------------------------ #
print(np.shape(x_train))

# ---------------------- Model Training ---------------------------- #
model = Sequential()
model.add(LSTM(2, input_shape=(series, 1)))
Dropout(0.2)
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
#fit the model
history = model.fit(x_train, y_train, epochs=1000, batch_size=32 , verbose=2)
# ------------------------------------------------------------------ #



y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

y = scaler.inverse_transform([y_test])
# print(y[0 , 100])
plt.plot(y_pred , c = 'r')
plt.plot(y[0 , :] , c = 'b')
plt.show()

print(y_pred)

model.save('absal.h5')


plt.plot(history.history['loss'])
