# config.py
from torch import nn


def get_config(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate):
    config = dict(
        batch_size=batch_size,
        criterion=nn.MSELoss(),
        max_epochs=num_epochs,
        learning_rate=learning_rate,
        results=results_directory,
        data=data_directory,
        testdata=test_data_directory,
        augment = True,
        adv_weight = 0.001,
        model = 'model2-13x5-Layer-Lightning'
    )

    return config