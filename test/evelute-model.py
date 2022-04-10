import pandas as pd
import sys
import wandb
from wandb.keras import WandbCallback
from model.CNN import CNN
import sys

sys.path.insert(1, '/Users/phamvanmanh/sentiment-analysis-vietnamese/')
from processing_data import MakeDataset


def test_predict_model():
    #f1_score
    #report
    pass
def test_data():
    
    pass

    
def test_training_model():
    
    parameters.update({
       
     
        "max_lenght": 128,
        "output_dim": 100,
        "dropout": 0.4,
        "learning_rate": 1e-2
    }
        
    )
   
    model = CNN(wandb)

    model.set_parameters(parameters)
    model.setup_model()
    model.train(x_train, y_train,x_test, y_test, wandb_callback=WandbCallback())

    pass


def test_speed_model():
    #time
    #report
    pass

if __name__ == '__main__':
    #setup wandb
    run = wandb.init(project="my-awesome-project",tags=["debug", "push", "phammanh", "dataversion1"])
    #setup datasets
    f_datasets = '/Users/phamvanmanh/sentiment-analysis-vietnamese/datasets/test_data'
    files = ['train.csv', 'test.csv']
    #log datasets
    artifact = wandb.Artifact('datasets-testing', type='my_dataset')
    artifact.add_dir(f_datasets)
    run.log_artifact(artifact)

    #paramters datasets:
    parameters = {
        'max_lenght': 128,
         "vocabsize": 30000
    }
    
    make_dataset = MakeDataset.MakeDataset()

    make_dataset.read_csv(f_datasets , name_file = files[0].split(".")[0])

    x_train, y_train = make_dataset.makeDataset( max_lenght = parameters['max_lenght'],  vocabsize = parameters['vocabsize'])
    #statistic train datasets
    make_dataset.statistic(run)

    make_dataset.read_csv(f_datasets  , name_file =files[1].split(".")[0])
    x_test, y_test = make_dataset.makeDataset( max_lenght = parameters['max_lenght'], vocabsize = parameters['vocabsize'])
    #statistic test datasets
    make_dataset.statistic(run)

    test_training_model()