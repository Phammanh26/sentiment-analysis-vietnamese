import wandb
from wandb.keras import WandbCallback
import json
import sys
import os
import os
import wandb

sys.path.insert(1, os.getcwd())
from processing_data.MakeDataset import MakeDataset
from model_building.CNN import CNN

os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY')
author = os.getenv('AUTHOR')

def test_predict_model():
    #f1_score
    #report
    pass
def test_data():
    
    pass

    
def test_training_model(x_train, y_train, parameters):
    model = CNN(wandb)
    model.set_parameters(parameters)
    model.setup_model()
    model.train(x_train, y_train,  wandb_callback=WandbCallback())

    pass


def test_speed_model():
    #time
    #report
    pass

if __name__ == '__main__':
    wandb.login()
    project_name ="my-awesome-hello-1111"
    wandb.init(project = project_name) 
   
    wandb.run.name = author + wandb.run.id

    #setup wandb
    run = wandb.init(project=project_name,tags=["debug", "push", "phammanh", "dataversion1"])
    #paramters datasets:
    parameters = {
        'max_lenght': 128
    }
    path = os.getcwd()
    f = open(path + '/datasets/config.json')
    config = json.load(f)

     #log datasets
        
    artifact = wandb.Artifact('datasets-testing', type='my_dataset')
    
    artifact.add_dir(path + config['folder-datasets'])
    wandb.log_artifact(artifact)
    dataset = MakeDataset(config, wandb)
   
    dataset_dict = dataset.make(parameters['max_lenght'])
    dataset_train, dataset_test =  dataset_dict['train.csv'], dataset_dict['test.csv']
    x_train,y_train = dataset_train[0], dataset_train[1]
    x_test, y_test = dataset_test[0], dataset_test[1]

    parameters.update({
       
        "input_dim": config['vocabsize'],
        "max_lenght": 256,
        "output_dim": 100,
        "dropout": 0.2,
        "learning_rate": 1e-2
    })
    test_training_model(x_train,y_train, parameters)
