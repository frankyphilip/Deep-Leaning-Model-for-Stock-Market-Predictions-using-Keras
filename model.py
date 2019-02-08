
# coding: utf-8

# In[1]:


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Importing the dataset
dataset_train=pd.read_csv('ongc_petrol_train.csv')
dataset_train=dataset_train.dropna()
x=dataset_train.iloc[:,2:].values
y=dataset_train.iloc[:,1:2].values


# In[3]:


# len(x)
x.shape


# In[4]:


#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1)) #x_scaled will be in the range of 0 and 1
x_scaled=sc.fit_transform(x)
sc_y=MinMaxScaler(feature_range=(0,1))
y=sc_y.fit_transform(y)
#Creating a datastructure with 60 timestamps 
#meaning that it will learn the trend from the previous 60 days(3 months) and based on that it will predict the current output
x_train=[]
y_train=[]
for i in range(20,2460):
    x_train.append(x_scaled[i-20:i,0:]) #Here 0 is the index of the column(y_train)
    y_train.append(y[i])
x_train,y_train=np.array(x_train),np.array(y_train)    


# In[5]:


x_train.shape


# In[6]:


#reshaping (adding more predictors)
x_train=np.reshape(x_train, (x_train.shape[0],x_train.shape[1],6))
x_train, y_train = np.array(x_train), np.array(y_train)


# In[7]:


#Building the RNN
import os
# os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

#initializing the RNN
regressor=Sequential()

#adding the first LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(None,6))) #7 is the no of predictors
regressor.add(Dropout(0.2))

#adding the second LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#adding the third LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#adding the fourth LSTM layer and some dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units=1))

#compiling the RNN
regressor.compile(optimizer='adam',loss='mse')
    
#Fitting the RNN to the training set
regressor.fit(x_train,y_train, epochs=100, batch_size=32)


# In[ ]:


# x.shape


# In[8]:


#making the predictions and visualizing the results
#getting the real stock price 
dataset_test=pd.read_csv('ongc_petrol_test.csv')
dataset_test=dataset_test.dropna()
x_test=dataset_test.iloc[:,2:].values
y_test=dataset_test.iloc[:,1:2].values
real=np.concatenate((x,x_test),axis=0)


# In[14]:


# real.shape
# dataset_train.shape


# In[9]:


#getting the predicted stock price 
scaled_real_stock_price = sc.fit_transform(real)
inputs=[]
for i in range(len(dataset_train), len(real)):
    inputs.append(scaled_real_stock_price[i-20:i, 0:])
inputs=np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 6))


# In[10]:


inputs.shape


# In[11]:


#rediction
pred=regressor.predict(inputs) #scaled values
pred=sc_y.inverse_transform(pred) 


# In[13]:


pred


# In[15]:


#Visualizing the result
plt.plot(y_test[:,0:],color='red',label='Real values')
plt.plot(pred,color='blue',label='Predicted values')
plt.title('ONGC Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('ONGC Stock Price')
plt.legend()
plt.show()

