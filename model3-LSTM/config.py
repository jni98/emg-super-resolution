# config.py
from torch import nn


def get_config(data_directory, test_data_directory, results_directory, num_epochs, batch_size):
    config = dict(
        seq_len=2048,
        batch_size=batch_size,
        criterion=nn.MSELoss(),
        max_epochs=num_epochs,
        n_features=21,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.005,
        results=results_directory,
        data=data_directory,
        testdata=test_data_directory,
        kernel_size=3
    )

    return config