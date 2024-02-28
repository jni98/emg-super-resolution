"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Trainer. It is used to train the GAN and to generate super resolution data.

The code is called from the train.py file which is started by the main function.
"""

import json
import os

import matplotlib
import wandb

matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from generator import Generator
from data import Custom_Dataset


class Trainer:
    """
    Class to train the model.
    """

    def __init__(self, data_directory: str, test_data_directory: str, results_directory: str, num_epochs: int,
                 config, sweep_id) -> None:
        """
        Constructs all the necessary attributes for the Trainer object.

        Parameters
        ----------
        data_directory: str
                    Path to the directory where the training data is stored
        test_data_directory: str
                    Path to the directory where the testing data is stored
        results_directory: str
                    Path to the directory where the results should be saved
        num_epochs: int
                    Number of epochs to train the model
        config: object
                    Configuration of the model for sweep
        sweep_id: str
                    ID of the sweep
        """

        self.data_directory = data_directory
        self.test_data_directory = test_data_directory
        self.run_folder = results_directory
        self.num_epochs = num_epochs
        self.config = config
        self.sweep_id = sweep_id

        self.lr = config.learning_rate
        self.batch_size = config.batch_size

        self.betas = (0.9, 0.999)

        # Create a dictionary for the hyperparameters
        self.hyperparameter = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        wandb.init(project="emg-super-resolution-NoGAN", entity="masterthesis-sr")

        print("init done")

    def prepare_data(self) -> None:
        """
        Prepares the training, validation and testing set.

        Returns
        -------
        None
        """

        # Load the JSON file with values for standardization into a Python dictionary
        with open(self.data_directory + '/mean_std.json', 'r') as file:
            self.mean_std_dict = json.load(file)

        data_train_val = pd.read_csv(self.data_directory + '/data.csv', sep=";")
        data_test = pd.read_csv(self.test_data_directory + '/data.csv', sep=";")

        train, val = train_test_split(data_train_val, train_size=0.8, test_size=0.2, random_state=42)

        # Use Custom_Dataset with standardization
        self.training = torch.utils.data.DataLoader(
            Custom_Dataset(train, self.data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std']),
            batch_size=self.batch_size, pin_memory=True
        )

        self.validation = torch.utils.data.DataLoader(
            Custom_Dataset(val, self.data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std']),
            batch_size=self.batch_size, pin_memory=True
        )
        self.testing = torch.utils.data.DataLoader(
            Custom_Dataset(data_test, self.test_data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std']), pin_memory=True)

        self.hyperparameter["num_samples"] = data_train_val.shape[0]
        self.hyperparameter["standardized"] = 'True'
        self.hyperparameter["device"] = str(self.device)

        # Ensure the directory exists
        os.makedirs(self.run_folder , exist_ok=True)

        # Save the dictionary to a file
        with open(self.run_folder + '/hyperparameter.json', 'w') as file:
            json.dump(self.hyperparameter, file)

        print("Successfully prepared the data")

    def initialize_models(self) -> None:
        """
        Initializes the Generator, Optimizers and Loss functions.

        Returns
        -------
        None
        """
        self.generator = Generator(self.config.num_res_blocks, self.config.num_res_blocks_last, dropout_prob_input=self.config.dropout_prob_input, dropout_prob=self.config.dropout_prob, in_channels=1)

        # move to gpu in case it's possible
        self.generator.to(self.device)
        # print("Generator summary")
        # print(str(summary(self.generator, (1, 7,3,2048))))

        self.mse = nn.MSELoss()  # Mean Squared error Loss

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)

        print("Successfully initialized the models")

    def train(self) -> tuple:
        """
        Trains the complete model for a number of epochs.

        Returns
        -------
        train_g_loss: list
                    Generator loss over the epochs
        val_loss: list
                    Validation loss over the epochs
        psnr: list
                    PSNR over the epochs
        """
        train_g_loss = []
        val_loss = []
        psnr = []
        best_val_loss = float('inf')

        # Add wandb.watch() here to start tracking gradients and parameters
        wandb.watch(self.generator, criterion='mse', log='all')

        for epoch in range(1, self.num_epochs + 1):
            print("epoch", epoch)

            # Check GPU memory usage before training
            initial_allocated = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_cached()

            epoch_loss_g = self._train_epoch(epoch)
            avg_val_loss, avg_psnr = self._validate(epoch)

            train_g_loss.append(epoch_loss_g)
            val_loss.append(avg_val_loss)
            psnr.append(avg_psnr)

            # Save best model based on validation loss or other metrics
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                # Save the final models when a new best model is found
                self._save_final_models()

            # self._save_checkpoint(epoch)
            # Clear unnecessary variables to release memory
            del epoch_loss_g, avg_val_loss, avg_psnr

            # Check GPU memory usage after training
            allocated_diff = torch.cuda.memory_allocated() - initial_allocated
            cached_diff = torch.cuda.memory_cached() - initial_cached

            print(f"After epoch {epoch}, GPU memory allocated increased by: {allocated_diff} bytes")
            print(f"After epoch {epoch}, GPU memory cached increased by: {cached_diff} bytes")

        return train_g_loss, val_loss, psnr

    def _train_epoch(self, epoch: int) -> tuple:
        """
        Trains the model for one epoch.

        Parameters
        ----------
        epoch: int

        Returns
        -------
        avg_loss_g: float
                    Average loss of the generator over the training set
        """
        self.generator.train()
        total_loss_g = 0.0

        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.training):
            print("batch index", batch_idx)

            with torch.no_grad():  # Disable gradient calculation within this block
                # Move data to CUDA device during training or validation loops
                lr_emg = lr_emg.to(self.device)
                hr_emg = hr_emg.to(self.device)

            # Train Generator
            self.optimizer_G.zero_grad()

            # Generate SR data from LR data
            sr_emg = self.generator(lr_emg).to(self.device)

            loss_generator = self.mse(sr_emg, hr_emg)  # Content loss

            loss_generator.backward()

            self.optimizer_G.step()

            total_loss_g += loss_generator.item()

        avg_loss_g = total_loss_g / len(self.training)

        wandb.log({'epoch': epoch, 'avg_loss_g': avg_loss_g})

        # Print progress per epoch
        print(f"Epoch [{epoch}/{self.num_epochs}] Average Loss G: {avg_loss_g}")

        return avg_loss_g

    def _validate(self, epoch: int) -> tuple:
        """
        Validates the model by calculating the loss on the validation set.

        Parameters
        ----------
        epoch: int

        Returns
        -------
        avg_validation_loss: float
                             Average validation loss over the validation set
        avg_psnr: float
                    Average PSNR over the validation set
        """
        self.generator.eval()  # Set the generator to evaluation mode

        total_validation_loss = 0.0
        total_psnr = 0.0

        with torch.no_grad():
            for val_batch_idx, (val_lr, val_hr, val_lr_path, val_hr_path) in enumerate(self.validation):
                val_lr = val_lr.to(self.device)
                val_hr = val_hr.to(self.device)

                # Generate SR data from LR data using the generator
                val_sr_emg = self.generator(val_lr).to(self.device)
                #val_sr_emg = self.destandardize_data(val_sr_emg)

                # Calculate validation loss
                val_loss = self.mse(val_sr_emg, val_hr)

                # Accumulate validation loss
                total_validation_loss += val_loss.item()

                psnr = 20 * torch.log10(1.0 / torch.sqrt(val_loss))

                total_psnr += psnr.item()

            avg_validation_loss = total_validation_loss / len(self.validation)
            print(f"Epoch [{epoch}/{self.num_epochs}] Validation Loss: {avg_validation_loss}")
            wandb.log({'val_loss': avg_validation_loss})

            avg_psnr = total_psnr / len(self.validation)

        self.generator.train()  # Set the generator back to training mode

        return avg_validation_loss, avg_psnr

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Saves the Generator and Discriminator at a given epoch.

        Parameters
        ----------
        epoch: int

        Returns
        -------
        None
        """
        if epoch % 10 == 0:
            torch.save(self.generator.state_dict(),
                       self.run_folder +  f'/generator_epoch_{epoch}.pt')

    def _save_final_models(self) -> None:
        """
        Saves the final Generator and Discriminator.

        Returns
        -------
        None
        """
        torch.save(self.generator.state_dict(), self.run_folder  + '/final_generator.pt')

    def _plot_result_losses(self, train_g_loss: list, val_loss: list, psnr: list) -> None:
        """
        Plots the Generator and Discriminator loss over the epochs and the validation loss and PSNR over the epochs.

        Parameters
        ----------
        train_g_loss: list
        val_loss: list
        psnr: list

        Returns
        -------
        None
        """

        # Convert lists to NumPy arrays
        train_g_loss_np = np.array(train_g_loss)
        val_loss_np = np.array(val_loss)
        psnr_np = np.array(psnr)

        # Save NumPy arrays to files
        np.save(self.run_folder  + '/train_g_loss.npy', train_g_loss_np)
        np.save(self.run_folder + '/val_loss.npy', val_loss_np)
        np.save(self.run_folder + '/psnr.npy', psnr_np)

        # Plot Generator and Discriminator loss
        fig_generator_discriminator, (
            ax_generator_loss) = plt.subplots(nrows=1, ncols=1)

        # Plot Generator loss
        ax_generator_loss.plot(np.arange(len(train_g_loss_np)), train_g_loss_np)
        ax_generator_loss.set_yscale('log')
        ax_generator_loss.set_xlabel('Epoch')
        ax_generator_loss.set_ylabel('Loss')
        ax_generator_loss.set_title('Generator Loss over Epochs')

        fig_generator_discriminator.tight_layout()
        fig_generator_discriminator.savefig(self.run_folder +  '/training_losses.png')

        # Plot Validation loss, PSNR in a combined figure
        fig_val_psnr, (ax_val_loss, ax_psnr) = plt.subplots(nrows=2, ncols=1)

        # Plot Validation loss
        ax_val_loss.plot(np.arange(len(val_loss_np)), val_loss_np)
        ax_val_loss.set_xlabel('Epoch')
        ax_val_loss.set_ylabel('Loss')
        ax_val_loss.set_title('Validation Loss over Epochs')

        # Plot PSNR
        ax_psnr.plot(np.arange(len(psnr_np)), psnr_np)
        ax_psnr.set_xlabel('Epoch')
        ax_psnr.set_ylabel('PSNR')
        ax_psnr.set_title('PSNR over Epochs')

        fig_val_psnr.tight_layout()
        fig_val_psnr.savefig(self.run_folder  + '/validation_loss_psnr.png')

    def generate_sr_data(self) -> None:
        """
        Generates super-resolved data from the test dataset and saves it to a file.

        Returns
        -------

        """

        # load the final generator
        self.generator.load_state_dict(torch.load(self.run_folder + '/final_generator.pt'))
        self.generator.eval()  # Set the generator to evaluation mode

        with torch.no_grad():
            for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
                lr_emg = lr_emg.to(self.device)
                hr_emg = hr_emg.to(self.device)

                # Generate super-resolved data (SR) from LR data
                lr_sr_emg = self.generator(lr_emg).detach().to(self.device)
                lr_sr_emg = lr_sr_emg.to(self.device).squeeze().numpy()
                lr_sr_emg_destandardized = self.destandardize_data(lr_sr_emg)

                # Generate super-resolved data (SR) from HR data
                hr_sr_emg = self.generator(hr_emg).detach().to(self.device)
                hr_sr_emg = hr_sr_emg.to(self.device).squeeze().numpy()
                hr_sr_emg_destandardized = self.destandardize_data(hr_sr_emg)

                # Save signal from lr to sr and hr to sr
                model_path = self.run_folder + '/results/'
                os.makedirs(model_path, exist_ok=True)
                np.save(model_path + lr_emg_paths[0], lr_sr_emg_destandardized)
                np.save(model_path + lr_emg_paths[0], hr_sr_emg_destandardized)

    def destandardize_data(self, standardized_data: np.ndarray) -> np.ndarray:

        """
        De-standardizes the data with the mean and standard deviation from the training set.

        Parameters
        ----------
        standardized_data: numpy array

        Returns
        -------
        destandardized_data: numpy array
        """

        return standardized_data * self.mean_std_dict['std'] + self.mean_std_dict['mean']
