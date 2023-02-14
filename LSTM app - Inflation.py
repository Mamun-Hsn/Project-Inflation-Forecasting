
from datetime import date
import matplotlib.pyplot as plt
import pandas_datareader as data
import plotly.express as px
import plotly.graph_objects as go
from keras.models import load_model
import streamlit as st
import base64
import pandas as pd
from datetime import datetime as dt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from pandas.plotting import lag_plot


import seaborn as sns


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pandas import Grouper

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,Conv1D,MaxPool1D,Conv2D,MaxPool2D


header   = st.container()
dataset  = st.container()
features = st.container()
Model    = st.container()


@st.cache(suppress_st_warning=True)
def get_data(filename):
    Inflation_data = pd.read_csv(filename) 
    return Inflation_data

with header:
    st.title('Inflation Prediction')
    st.markdown('Project by - **Group 5**')

with dataset:
    st.header('International Monetary Fund Dataset')
    
    india = get_data('C:/Users/manas/deployement_inflation_forecasting/New India data _1980(csv).csv')
    st.write(india.head(50))
    st.write(india.shape)
    
    # EDA
    st.header('EDA')
    st.subheader('Histogram Distributions')

    #Histogram Visualization
    fig= plt.figure(figsize=(14,7))
    sns.histplot(india['Inflation, average consumer prices(%)'])
    st.pyplot(fig)
    

    
    # Boxplot Visulization
    st.subheader('Boxplots to check for outliers')
    fig= plt.figure(figsize=(14,7))
    sns.boxplot(india['Inflation, average consumer prices(%)'], )
    st.pyplot(fig)

# Describing Data
st.markdown("***")
st.subheader('Descriptive Statistics')
st.write(india.describe())

india.dropna(inplace=True)
india.index

st.markdown("***")
from datetime import datetime as dt


india_month_interpolated = pd.read_csv(r'C:/Users/manas/deployement_inflation_forecasting/india_month.csv')
st.write(india_month_interpolated.head(40))
st.write(india_month_interpolated.shape)

# Visualizations
st.markdown("***")

st.subheader('Monthly Inflation Box Plot')
fig= plt.figure(figsize=(14,7))
sns.boxplot(x="month",y='inflation_rate',data=india_month_interpolated)
st.pyplot(fig)



st.markdown("***")

st.subheader('Yearly Inflation Line Plot')
fig= plt.figure(figsize=(14,7))
sns.lineplot(x='year',y='inflation_rate', data=india_month_interpolated)
st.pyplot(fig)

st.markdown("***")

india_month_interpolated1 = pd.read_csv(r'C:/Users/manas/deployement_inflation_forecasting/india_month_interpolated1.csv')
df1=india_month_interpolated1.reset_index()['inflation_rate']
scaler = MinMaxScaler(feature_range=(0,1)) #This transformation is often used as an alternative to zero mean, unit variance scaling.
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


training_size=int(len(df1)*0.65)


test_size=len(df1)-training_size


train_data=df1[:training_size]


test_data=df1[training_size:len(df1)]

# convert an array of values into a dataset matrix
def create_dataset(dataset , time_step =100):
    dataX ,dataY =[] , []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]   #i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i+time_step ,0])
    return np.array(dataX), np.array(dataY)

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

#st.write(print(X_train.shape), print(y_train.shape))
#st.write(print(X_test.shape), print(y_test.shape))

st.header('Model Building')
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

st.subheader('LSTM ')
model= load_model('Lstm.h5')

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

lstm_rmse=np.sqrt(mean_squared_error(y_train,train_predict))
st.write("RMSE Value Training :",lstm_rmse)

lstm_rmse_test = np.sqrt(mean_squared_error(y_test,test_predict))
st.write("RMSE Value Testing :",lstm_rmse_test)

#Final Visuals
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
st.subheader('Plotting The Predicted values By Model')
fig= plt.figure(figsize=(14,7))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(labels=["complete data","train data","test data"])
st.pyplot(fig)

x_input=test_data[77:].reshape(1,-1)# reshape because we have given the input size in that way

temp_input=list(x_input)
temp_input=temp_input[0].tolist()  # data type was converted from Series to List.

#demonstrate prediction for next 30  months
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} month input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} month output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

month_new = np.arange(1,101) #we use 101 because we use previous 100 data to forscast
month_pred = np.arange(101,131) # 131 because we are going to forecast next 30 months

st.subheader("Forecasting the Data for Next 30 Months")
fig= plt.figure(figsize=(14,7))
plt.plot(month_new,scaler.inverse_transform(df1[405:]))
plt.plot(month_pred,scaler.inverse_transform(lst_output))
plt.legend(labels=["Original data","Forecasted data"])
st.pyplot(fig)