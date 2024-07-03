import argparse
import json

import torch

from model import LSTMRegressor  # Import your model class
from data import CustomDataModule  # Import your data module
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument('--data', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--testdata', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--results', type=str, required=True, help='Path to results directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    return parser.parse_args()



def main(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate, checkpoint_directory):
    config = get_config(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate)

    wandb_logger = WandbLogger(project="emg-super-resolution-LSTM", save_dir=checkpoint_directory, config=config, mode='online', offline=False)

    # Load the JSON file with values for standardization into a Python dictionary
    with open(data_directory + '/mean_std.json', 'r') as file:
        mean_std_dict = json.load(file)

    model_for_inference = LSTMRegressor.load_from_checkpoint(
        checkpoint_path=checkpoint_directory + "/best_model.ckpt",
        n_features=config['n_features'],
        hidden_size=config['hidden_size'],
        seq_len=config['seq_len'],
        batch_size=1,  # Set batch_size to 1 for inference
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate'],
        criterion=config['criterion'],
        mean=mean_std_dict['mean'],
        std=mean_std_dict['std']
    )

    checkpoint_path = checkpoint_directory + "/best_model.ckpt"

    # Load the checkpoint to extract the epoch information
    checkpoint = torch.load(checkpoint_path)
    best_epoch = checkpoint['epoch']
    print(f'The best checkpoint was saved at epoch {best_epoch}.')

    dm = CustomDataModule(
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        data_directory=config['data'],
        test_data_directory=config['testdata']
    )

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu"
    )

    trainer.test(model=model_for_inference, datamodule=dm)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    data_directory = args.data
    test_data_directory = args.testdata
    results_directory = args.results
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr

    # Add directory of best_model.ckpt file
    checkpoint_directory = '/home/hpc/iwso/iwso124h/model3-LSTM-812094'

    # data_directory = '../dataset_lstm_40960/train'
    # test_data_directory = '../dataset_lstm_40960/test'
    # results_directory = 'model3-LSTM-835361'
    # num_epochs = 2
    # batch_size = 1
    # learning_rate = 0.00001
    # checkpoint_directory = 'FINAL-BP500-model3-LSTM-838171'

    main(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate, checkpoint_directory)
