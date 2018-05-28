from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Activation , Dropout
import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D , Conv2D

from sklearn.model_selection import train_test_split






# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)






df = pd.read_csv('airdata.csv')
Date = df['Date']
df.drop(['Date', 'Time'],1,inplace=True)
x,y = to_xy(df,"NO2(GT)")


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=45)


model = Sequential([
    Dense(100 , input_dim=x.shape[1] , activation='relu'),
    Dropout(0.5),

    #Dropout(0.5),
   # Dense(20 , activation='relu'),

    Dropout(0.2),
    Dense(1,kernel_initializer='normal')

])

model.compile(
    loss = 'mean_squared_error',
    optimizer = 'adam'
)

monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-3,
    patience=5, verbose=1,
    mode='auto')

model.fit(
    x,y,
    validation_data=(x_test,y_test),
          callbacks=[monitor],
    verbose=2,epochs=1000)


pred=model.predict(x)
score = np.sqrt(metrics.mean_squared_error(pred,y))
print("Final score (RMSE): {}".format(score))
for i in range(10):
    print("{}. DAte : {}, NO2(GT): {}, predicted NO2(GT): {}".format(i+1,Date[i],y[i],pred[i]))









