import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras
from keras import regularizers
from keras.utils import to_categorical
from numpy import array
from scipy.stats import zscore



import pandas as pd
from sklearn import preprocessing
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











df = pd.read_csv('test_df.csv',na_values=['NA','?'])

df['Work load Average/day ']=zscore(df['Work load Average/day '])
df['Transportation ']=zscore(df['Transportation '])
df['Distance']=zscore(df['Distance'])
df['Service time']=zscore(df['Service time'])
df['Age']=zscore(df['Age'])
df['Hit target']=zscore(df['Hit target'])
df['Weight']=zscore(df['Weight'])
df['Height']=zscore(df['Height'])
df['Absenteeism time in hours']=zscore(df['Absenteeism time in hours'])
df['Body mass index']=zscore(df['Body mass index'])
df['Son']=zscore(df['Son'])
df['Pet']=zscore(df['Pet'])



# id=df['ID']
# df['Work load Average per day '] =df['Work load Average per day ']/df['Work load Average per day '].mean()
# df.drop('ID',axis=1,inplace=True)
# df = pd.get_dummies(df,columns=['Month of absence',
#                                'Seasons','Day of the week','Disciplinary failure',
#                                 'Education'] , drop_first= True )
df.to_csv('Absenteeism_at_work.csv',index=False)

x,y = to_xy(df,'Reason for absence')








x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=45)


print(y.shape[1])
model = Sequential([


    Dense(100, input_dim=x.shape[1], kernel_initializer='normal', activation='tanh'),
    Dense(50, input_dim=x.shape[1], kernel_initializer='normal', activation='tanh'),


    Dense(y_train.shape[1], kernel_initializer='normal',activation='softmax')


])
model.compile(loss='categorical_crossentropy', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor], verbose=2,epochs=1000)


############################################  validation    #################

np.set_printoptions(suppress=True)
pred = model.predict(x_test)
from sklearn.metrics import log_loss
print(log_loss(y_test,pred))


predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)

print("Predictions: {}".format(predict_classes[0:25]))
print("Expected:    {}".format(expected_classes[0:25]))


model.save('Absenteeism_at_work.h5')