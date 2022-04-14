from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM,Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
import numpy as np

class CNN:
    def __init__(self, wandb):

        self.name = 'CNN model'
        self.author = None
        self.parameters = None
        self.model = None
        self.wandb = wandb

    def set_parameters(self, parameters):
        self.parameters = parameters


    def setup_model(self):

        input_dim = self.parameters['input_dim']
        max_lenght = self.parameters['max_lenght']
        output_dim = self.parameters['output_dim']
        dropout = self.parameters['dropout']
        learning_rate = self.parameters['learning_rate']
        self.model = Sequential()
        self.model.add(Embedding(input_dim = input_dim,output_dim = output_dim, input_length= max_lenght))

        self.model.add(Bidirectional(LSTM(100,return_sequences=False)))
        self.model.add(Dropout(dropout))

        self.model.add(Dense(1, activation ='sigmoid'))
        self.model.compile(loss="binary_crossentropy", optimizer= Adam(learning_rate=learning_rate), metrics=['acc'])
        return self.model

    def train(self, x_train, y_train, wandb_callback, epochs = 10, batch_size=1000):
       
        X_train, X_test, y_train, y_test =  train_test_split(x_train, y_train, test_size=0.33, random_state=42)
        self.parameters['epochs'] = epochs
        self.parameters['batch_size'] = batch_size
        
        #report wantdb
        self.wandb.log(self.parameters)
        self.model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epochs,batch_size=batch_size, callbacks =[wandb_callback])
        
        return self.model

    def evaluate(self, x_test, y_test):
       
    
        y_pred = self.model.predict(x_test)
        
        return y_pred