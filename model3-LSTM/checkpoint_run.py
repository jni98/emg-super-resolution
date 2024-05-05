import argparse
from model3 import LSTMRegressor  # Import your model class
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

    model_for_inference = LSTMRegressor.load_from_checkpoint(
        checkpoint_path=checkpoint_directory + "/best_model.ckpt",
        n_features=config['n_features'],
        hidden_size=config['hidden_size'],
        seq_len=config['seq_len'],
        batch_size=1,  # Set batch_size to 1 for inference
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate'],
        criterion=config['criterion']
    )

    dm = CustomDataModule(
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        data_directory=config['data'],
        test_data_directory=config['testdata']
    )

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
    )

    trainer.test(model_for_inference, datamodule=dm)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    data_directory = args.data
    test_data_directory = args.testdata
    results_directory = args.results
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    checkpoint_directory = '/home/hpc/iwso/iwso124h/model3-LSTM-812094'

    # data_directory = '../dataset_lstm/train'
    # test_data_directory = '../dataset_lstm/test'
    # results_directory = 'runs/'
    # num_epochs = 2
    # batch_size = 16
    # learning_rate = 0.00001
    # checkpoint_directory = 'runs'

    main(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate, checkpoint_directory)
