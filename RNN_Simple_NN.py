import numpy as np
from keras.models import Sequential
from keras.layers import Dense , LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


Data = [[[i+j] for i in range(5)]for j in range(100)]
target = [[i+5] for i in range(100)]

data = np.array(Data , dtype=(float))/100
target = np.array(target , dtype=(float))/100
print(np.shape(target))


x_train , x_test , y_train , y_test = train_test_split(
    data , target , test_size=0.2 , random_state=4
)


model  = Sequential([

    LSTM((1) , batch_input_shape=(None , 5, 1 ) , return_sequences=True),
    LSTM((1)  , return_sequences=False)
])

model.compile(loss = 'mean_absolute_error' , optimizer='adam', metrics=['accuracy'])
# monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
history = model.fit(
    x_train , y_train,
    epochs=500 , verbose=2,
    validation_data=(x_test , y_test),
  # callbacks=[monitor]
)


result = model.predict(x_test)
plt.scatter(range(20), result , c= 'r',)
plt.scatter(range(20), y_test , c = 'g')
plt.show()


plt.plot(history.history['loss'])
plt.show()


model.save("ped_6th_dig.h5")
