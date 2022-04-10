from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM,Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '/Users/phamvanmanh/sentiment-analysis-vietnamese/datasets')


import MakeDataset

def setup(vocabsize = 30000, max_lenght = 70):
    model = Sequential()
    # model.add(Embedding(input_dim = len(word_index), output_dim = 100, embeddings_initializer = tensorflow.keras.initializers.Constant(wv_matrix), input_length= MAX_SEQUENCE_LENGTH))
    model.add(Embedding(input_dim = vocabsize,output_dim = 50, input_length= max_lenght))

    model.add(Bidirectional(LSTM(100,return_sequences=False)))
    model.add(Dropout(0.3))

    # model.add(LSTM(20,return_sequences=False))
    # model.add(Dropout(0.3)) 

    model.add(Dense(1, activation ='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer= Adam(learning_rate=1e-2), metrics=['acc'])
    return model

def train(x_train, y_train, x_test, y_test, epochs = 10, batch_size=1000):
    model = setup()
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size)
    return model

def evalute(model, x_test,y_test ):
    y_pred=model.predict(x_test)
    y_pred = np.round(abs(y_pred))
    acc = accuracy_score(y_test, y_pred)
    cfm = accuracy_score(y_test, y_pred)
    print("Accuracy Score: ", acc)
    print("Confusion Matrix: \n", cfm)
    print(classification_report(y_test,y_pred))
    return {'acc': acc, 'cfm': acc}

if __name__ == '__main__':
    path = '/Users/phamvanmanh/sentiment-analysis-vietnamese/datasets/origin_data/train.csv'
    train_datasets = pd.read_csv(path)
    make_dataset = MakeDataset.MakeDataset()
    x_train, y_train = make_dataset.makeDataset(train_datasets)
    path = '/Users/phamvanmanh/sentiment-analysis-vietnamese/datasets/origin_data/test.csv'
    test_datasets = pd.read_csv(path)

    x_test, y_test = make_dataset.makeDataset(test_datasets)

 
    model = train(x_train, y_train, x_test, y_test)
    print(evalute(model, x_test, y_test))