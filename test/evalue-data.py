
import pandas as pd
from MakeDataset import MakeDataset
import wandb
def test_statistic_data():
    #statistic data
    # report
    pass

if __name__ == '__main__':

    path = '/Users/phamvanmanh/sentiment-analysis-vietnamese/datasets/origin_data/test.csv'
    train_datasets = pd.read_csv(path)
    make_dataset = MakeDataset()
    dataset = make_dataset.makeDataset(train_datasets)
    statistic = make_dataset.statistic()


    path_dataset = '/Users/phamvanmanh/sentiment-analysis-vietnamese/datasets/origin_data'
    run = wandb.init(project="my-awesome-project")

    #log datasets
    artifact = wandb.Artifact('dataset_2022', type='my_dataset')
    artifact.add_dir(path_dataset)
    run.log_artifact(artifact)
    wandb.log(statistic)







