from processing_data.CleanData import CleanData
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

import numpy as np
import pandas as pd
from collections import Counter


class MakeDataset:
    def __init__(self, config, wandb):
       
        self.folder = config['folder-datasets']
        self.f_train = config['train']
        self.f_test = config['test']
        self.vocabsize = config['vocabsize']
        self.dataset_train = None
        self.dataset_test = None
        self.wandb = wandb

        
    def read_csv(self, path):
        return pd.read_csv(path)

    def make(self, max_lenght):
      
        clean_data = CleanData()
        resutls = {}
        for type in ['test.csv', 'train.csv']:

            dataset = self.read_csv(self.folder + "/" + type)
            x = dataset['sentence'].values.tolist()
            y = dataset['label'].values.tolist()
            self.statistic (x, y, type, self.wandb)
            
            new_y = [1 if i == 'pos' else 0 for i in y]
            new_x = clean_data.processing_list_text(x)
            
            new_x_onehot=[one_hot(words,self.vocabsize)for words in new_x]
            new_x_embed=pad_sequences(new_x_onehot,padding='post',maxlen=max_lenght)
            resutls[type] =  [np.asarray(new_x_embed), np.asarray(new_y)]

        return resutls
    
    def statistic(self, x, y, type,  wandb):
        tokens = [sentence.split(" ") for sentence in x]
        vocab = Counter()
        for tokens in tokens:
            vocab.update(tokens)

        vocab_unique = len(vocab)
        negative = sum(map(lambda x : x == 'neg', y))
        positive =  sum(map(lambda x : x == 'pos', y))
        lenght = len(x)
        

        result_statis =  {
            
            "{}:".format(type) + "vocabsize" :vocab_unique ,
            "{}:".format(type) +"negative" : negative,
            "{}:".format(type) +"positive":  positive,
            "{}:".format(type) +"lenght" : lenght
        }
        wandb.log(result_statis)
        