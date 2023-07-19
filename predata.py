import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,GRU
from keras import optimizers 
from sklearn.metrics import mean_squared_error

seed = 1234
np.random.seed(seed)
plt.style.use('ggplot')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Data:
    def __init__(self, file_name, column_name) -> None:
        self.data = pd.read_csv(file_name, index_col="Datetime", parse_dates=["Datetime"])
        self.data = self.data.dropna()
        dataset = pd.DataFrame(self.data[column_name])

        # Normalize
        dataset_norm = dataset.copy()
        dataset[[column_name]]
        self.scaler = MinMaxScaler()
        dataset_norm[column_name] = self.scaler.fit_transform(dataset[[column_name]])

        # Partition data into data train, val & test
        totaldata = dataset.values
        totaldatatrain = int(len(totaldata)*0.75)
        totaldataval = int(len(totaldata)*0.1)
        totaldatatest = int(len(totaldata)*0.15)

        # Store data into each partition
        training_set = dataset_norm[0:totaldatatrain]
        val_set=dataset_norm[totaldatatrain:totaldatatrain+totaldataval]
        test_set = dataset_norm[totaldatatrain+totaldataval:]

        # Initiaton value of lag
        lag = 2
        # sliding windows function
        def create_sliding_windows(data,len_data,lag):
            x=[]
            y=[]
            for i in range(lag,len_data):
                x.append(data[i-lag:i,0])
                y.append(data[i,0]) 
            return np.array(x),np.array(y)

        # Formating data into array for create sliding windows
        array_training_set = np.array(training_set)
        array_val_set = np.array(val_set)
        array_test_set = np.array(test_set)

        # Create sliding windows into training data
        self.x_train, self.y_train = create_sliding_windows(array_training_set,len(array_training_set), lag)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0],self.x_train.shape[1],1))
        # Create sliding windows into validation data
        self.x_val,self.y_val = create_sliding_windows(array_val_set,len(array_val_set),lag)
        self.x_val = np.reshape(self.x_val, (self.x_val.shape[0],self.x_val.shape[1],1))
        # Create sliding windows into test data
        self.x_test, self.y_test = create_sliding_windows(array_test_set,len(array_test_set),lag)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0],self.x_test.shape[1],1))
            



            
    


