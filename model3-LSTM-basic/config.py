# config.py
from torch import nn
import torch.nn.functional as F


def get_config(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate):
    config = dict(
        seq_len=2048,
        batch_size=batch_size,
        criterion=nn.MSELoss(),
        #criterion=F.l1_loss,
        max_epochs=num_epochs,
        n_features=21,
        hidden_size=65,
        num_layers=2,
        dropout=0.2,
        learning_rate=learning_rate,
        results=results_directory,
        data=data_directory,
        testdata=test_data_directory,
        model_version = 'model_basic',
        augment = False
    )

    return config