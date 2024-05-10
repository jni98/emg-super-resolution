import argparse
import json

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model_GAN import GAN  # Update this import to your GAN model
from data import CustomDataModule  # Ensure this data module is suitable for your GAN
from config import get_config
from model_gen import Generator  # Import your generator model
from model_disc import Discriminator  # Import your discriminator model

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

def main(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate, checkpoint_path=None):
    seed_everything(1)
    config = get_config(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate)

    wandb_logger = WandbLogger(project="GAN-Training", save_dir=config['results'], config=config, mode='online', offline=False)

    generator = Generator(num_res_blocks=5, num_res_blocks_last=5, in_channels=1)
    discriminator = Discriminator(in_channels=1)

    # Load the JSON file with values for standardization into a Python dictionary
    with open(data_directory + '/mean_std.json', 'r') as file:
        mean_std_dict = json.load(file)

    # Assuming you're loading from a checkpoint or creating a new model
    if checkpoint_path is not None:
        model = GAN.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            generator=generator,
            discriminator=discriminator,
            learning_rate_g=config['learning_rate'],
            learning_rate_d = config['learning_rate'],
            adv_weight=config['adv_weight'],
            mean=mean_std_dict['mean'],  # These should be defined as per your model's needs
            std=mean_std_dict['std']
        )
    else:
        model = GAN(
            generator=generator,
            discriminator=discriminator,
            learning_rate_g=config['learning_rate'],
            learning_rate_d=config['learning_rate'],
            adv_weight=config['adv_weight'],
            mean=mean_std_dict['mean'],  # These should be defined as per your model's needs
            std=mean_std_dict['std']
        )

    dm = CustomDataModule(
        data_directory=config['data'],
        test_data_directory=config['testdata'],
        batch_size=config['batch_size'],
        augment=config['augment']  # Ensure augmentations are suitable for GAN input
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=config['results'],
        filename='best_model',
        save_top_k=1,
        save_weights_only=False,
        save_last=True
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        devices=1,
        accelerator="gpu",
        deterministic=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=config['results'],
        log_every_n_steps=1
    )

    if checkpoint_path is not None:
        trainer.fit(model, dm, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, dm)

    model_for_inference = GAN.load_from_checkpoint(
        checkpoint_path=config['results'] + "/best_model.ckpt",
        generator=generator,
        discriminator=discriminator,
        learning_rate_g=config['learning_rate'],
        learning_rate_d=config['learning_rate'],
        adv_weight=config['adv_weight'],
        mean=mean_std_dict['mean'],  # These should be defined as per your model's needs
        std=mean_std_dict['std']
    )

    # Assuming a test phase is also required
    trainer.test(model=model_for_inference, datamodule=dm)

if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.testdata, args.results, args.epochs, args.batch_size, args.lr,  args.checkpoint)

    #main('../dataset/train', '../dataset/test', 'runs/', 5, 1, 0.00001, None)
