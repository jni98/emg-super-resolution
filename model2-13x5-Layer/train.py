"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file is used to run the training of the GAN. It also calls the function to generate the super
resolution data. The grid is of shape 13x5 and the data is processed in 3D (height x width x samples).

You can run this file by running the main function.
"""
import argparse
import sys
from datetime import datetime
import os
from trainer import Trainer

def train(data_directory: str , test_data_directory: str, results_directory: str,num_epochs: int, batch_size: int, learning_rate: float) -> None:

    # Create a object of the trainer
    trainer = Trainer(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate)

    # Prepare the data of the trainer (read csv, split train and validation set)
    trainer.prepare_data()

    # Initialize the generator and discriminator
    trainer.initialize_models()

    # Run the training
    train_G_loss, train_D_loss, train_D_loss_real, train_D_loss_fake, val_loss, psnr = trainer.train()

    # Plot the training results
    trainer._plot_result_losses(train_G_loss, train_D_loss_real, train_D_loss_fake, train_D_loss, val_loss, psnr)

    #trainer.generate_sr_data()

def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument('--data', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--testdata', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--results', type=str, required=True, help='Path to results directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data_directory = args.data
    test_data_directory = args.testdata
    results_directory = args.results
    num_epochs = args.epochs
    batch_size = args.bs
    learning_rate = args.lr

    # data_directory = '../dataset/train'
    # test_data_directory ='../dataset/test'
    # results_directory = 'runs/'
    # num_epochs = 10
    # batch_size = 16
    # learning_rate = 0.0001

    train(data_directory, test_data_directory, results_directory, num_epochs, batch_size, learning_rate)