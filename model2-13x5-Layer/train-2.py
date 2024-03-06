"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file is used to run the training of the GAN. It also calls the function to generate the super
resolution data. The grid is of shape 13x5 and the data is processed in 3D (height x width x samples).

You can run this file by running the main function.
"""
# Import necessary libraries
import argparse

import torch
import wandb  # Import wandb library

from trainer2 import Trainer


def setup_sweep():
    # Define the sweep config
    sweep_config = {
        'method': 'grid',
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters': {
            'batch_size': {
                'values': [12],  # Add the batch sizes you want to include
            },
            'learning_rate': {
                'values': [ 0.0001]
            },
            'num_res_blocks_last': {
                'values': [5]
            },
            'num_res_blocks': {
                'values': [5]
            },

        }
    }

    # Initialize and run the sweep
    sweep_id = wandb.sweep(sweep_config, project="emg-super-resolution", entity="masterthesis-sr")

    return sweep_config, sweep_id


def train(data_directory: str, test_data_directory: str, results_directory: str,
          num_epochs: int, sweepconfig, sweep_id) -> None:
    with wandb.init(config=sweepconfig):
        config = wandb.config
        # Create a object of the trainer
        trainer = Trainer(data_directory, test_data_directory, results_directory, num_epochs, config, sweep_id)

        # Prepare the data of the trainer (read csv, split train and validation set)
        trainer.prepare_data()

        # Initialize the generator and discriminator
        trainer.initialize_models()

        # Run the training
        #train_G_loss, train_D_loss, train_D_loss_real, train_D_loss_fake, val_loss, psnr = trainer.train()

        # Plot the training results
        #trainer._plot_result_losses(train_G_loss, train_D_loss_real, train_D_loss_fake, train_D_loss, val_loss, psnr)

        trainer.generate_sr_data()

        # Explicitly release CUDA memory
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument('--data', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--testdata', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--results', type=str, required=True, help='Path to results directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    return parser.parse_args()


if __name__ == "__main__":
    sweep_config, sweep_id = setup_sweep()  # Setting up the sweep configuration

    # Parse arguments
    args = parse_args()
    data_directory = args.data
    test_data_directory = args.testdata
    results_directory = args.results
    num_epochs = args.epochs

    # data_directory = '../dataset/train'
    # test_data_directory ='../dataset/test'
    # results_directory = 'runs'
    # num_epochs = 50

    # Run the sweep
    wandb.agent(sweep_id,
                function=lambda: train(data_directory, test_data_directory, results_directory, num_epochs, sweep_config,
                                       sweep_id))
