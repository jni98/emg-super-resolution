import argparse

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data import CustomDataModule
from model2 import LSTMRegressor
from pytorch_lightning import Trainer, seed_everything
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

def main(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate):

    seed_everything(1)

    config = get_config(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate)

    wandb_logger = WandbLogger(project="emg-super-resolution-LSTM", save_dir=config['results'], config=config, mode='online', offline=False)

    # Define the EarlyStopping callback
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',  # Monitor validation loss
    #     min_delta=0.00001,  # Minimum change to qualify as an improvement
    #     patience=3,  # Number of epochs to wait for improvement
    #     verbose=True,  # Print messages when early stopping is triggered
    #     mode='min',  # Stop when the monitored quantity stops decreasing
    #     strict=True  # Error if `monitor` is not found
    # )

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Monitor validation loss
        mode='min',  # Save the model with the lowest validation loss
        dirpath=config['results'],  # Directory to save checkpoints
        filename='best_model',  # File name prefix for saved checkpoints
        save_top_k=1,  # Save only the best model
        verbose=True
    )

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        devices=1,
        accelerator="gpu",
        deterministic=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=config['results'],
        log_every_n_steps=1,
    )

    model = LSTMRegressor(
        n_features=config['n_features'],
        hidden_size=config['hidden_size'],
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        criterion=config['criterion'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate']
    )

    dm = CustomDataModule(
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        data_directory=config['data'],
        test_data_directory=config['testdata']
    )

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Total trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    trainer.fit(model, dm)

    # # Load the trained model with potentially a new batch size for inference
    # model_for_inference = LSTMRegressor.load_from_checkpoint(
    #     checkpoint_path=config["results"] + "/best_model.ckpt",
    #     n_features=config['n_features'],
    #     hidden_size=config['hidden_size'],
    #     seq_len=config['seq_len'],
    #     batch_size=1,  # You can specify a different batch size here if your model supports it
    #     num_layers=config['num_layers'],
    #     dropout=config['dropout'],
    #     learning_rate=config['learning_rate'],
    #     criterion=config['criterion']
    # )
    #
    # trainer.test(model_for_inference, datamodule=dm)

if __name__ == "__main__":

    #Parse arguments
    args = parse_args()
    data_directory = args.data
    test_data_directory = args.testdata
    results_directory = args.results
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr

    # data_directory = '../dataset_lstm/train'
    # test_data_directory ='../dataset_lstm/test'
    # results_directory = 'runs/'
    # num_epochs = 2
    # batch_size = 16
    # learning_rate = 0.00001

    main(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate)