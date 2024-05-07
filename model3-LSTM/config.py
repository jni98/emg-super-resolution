# config.py
from torch import nn


def get_config(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate):
    config = dict(
        seq_len=40960,
        batch_size=batch_size,
        criterion=nn.MSELoss(),
        max_epochs=num_epochs,
        n_features=2688,
        hidden_size=1344,
        num_layers=2,
        dropout=0.2,
        learning_rate=learning_rate,
        results=results_directory,
        data=data_directory,
        testdata=test_data_directory,
        model_version = 'model'
    )

    return config