import argparse
import json

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from model import LSTMRegressor  # Make sure this import is correctly pointing to your model definition
from data import CustomDataModule
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument('--data', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--testdata', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--results', type=str, required=True, help='Path to results directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a model checkpoint to resume training')
    return parser.parse_args()


def main(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate,
         checkpoint_path=None):
    seed_everything(1)
    config = get_config(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate)

    wandb.init(project="emg-super-resolution-LSTM", config=config, mode='online')

    wandb_logger = WandbLogger(project="emg-super-resolution-LSTM", save_dir=config['results'], config=config,
                               mode='online', offline=False)

    # Load the JSON file with values for standardization into a Python dictionary
    with open(data_directory + '/mean_std.json', 'r') as file:
        mean_std_dict = json.load(file)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=config['results'],
        filename='best_model',
        save_top_k=1,
        save_weights_only=False,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Assuming model is being loaded from a checkpoint
    if checkpoint_path is not None:
        print("Received checkpoint path for training: ", checkpoint_path)
        model = LSTMRegressor.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            n_features=config['n_features'],
            hidden_size=config['hidden_size'],
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
            criterion=config['criterion'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            mean=mean_std_dict['mean'],
            std=mean_std_dict['std']
        )
    else:
        model = LSTMRegressor(
            n_features=config['n_features'],
            hidden_size=config['hidden_size'],
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
            criterion=config['criterion'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            mean=mean_std_dict['mean'],
            std=mean_std_dict['std']

        )

    dm = CustomDataModule(
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        data_directory=config['data'],
        test_data_directory=config['testdata'],
        augment=config['augment'],
        num_workers=4,  # Adjust based on your system's CPU and memory resources
    )

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        devices=1,  # Specify the number of GPUs
        accelerator="gpu",
        #strategy="ddp",  # Use DistributedDataParallel for multi-GPU training
        deterministic=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=config['results'],
        log_every_n_steps=1
    )

    if checkpoint_path is not None:
        print("Received checkpoint path for fitting: ", checkpoint_path)
        trainer.fit(model, dm, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, dm)

    model_for_inference = LSTMRegressor.load_from_checkpoint(
        checkpoint_path=config['results'] + "/best_model.ckpt",
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

    trainer.test(model=model_for_inference, datamodule=dm)


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.testdata, args.results, args.epochs, args.batch_size, args.lr, args.checkpoint)

    #main('../dataset/train', '../dataset/test', 'runs/', 6, 2, 0.0001, None)
