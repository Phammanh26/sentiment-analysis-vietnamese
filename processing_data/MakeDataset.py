from processing_data.CleanData import CleanData
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
import pandas as pd
from collections import Counter


class MakeDataset:
    def __init__(self):
        self.vocabsize = 0
      
        self.negative = 0
        self.positive = 0
        self.lenght = 0
  
        self.x = None
        self.y = None
        self.datasets = None
        self.name_file = None
    def read_csv(self, folder, name_file, format = '.csv'):
        self.name_file = name_file
        path = folder +"/" + name_file + format
        self.datasets = pd.read_csv(path)

    def makeDataset(self, max_lenght, vocabsize):
      
        clean_data = CleanData()
        
        self.x = self.datasets['sentence'].values.tolist()
        self.y = self.datasets['label'].values.tolist()

        new_y = [1 if i == 'pos' else 0 for i in self.y]
        new_x = clean_data.processing_list_text(self.x)
        
        new_x_onehot=[one_hot(words,vocabsize)for words in new_x]
        new_x__embed=pad_sequences(new_x_onehot,padding='post',maxlen=max_lenght)

        return np.asarray(new_x__embed), np.asarray(new_y)
    
    def statistic(self, wandb):
        tokens = [sentence.split(" ") for sentence in self.x]
        vocab = Counter()
        for tokens in tokens:
            vocab.update(tokens)

        self.vocabsize = len(vocab)
        self.negative = sum(map(lambda x : x == 0, self.y))
        self.positive =  sum(map(lambda x : x == 1, self.y))
        self.lenght = len(self.x)
        

        result_statis =  {
            
            "{}:".format(self.name_file) + "vocabsize" :self.vocabsize ,
            "{}:".format(self.name_file) +"negative" : self.negative,
            "{}:".format(self.name_file) +"positive":  self.positive,
            "{}:".format(self.name_file) +"lenght" : self.lenght
        }
        wandb.log(result_statis)
        