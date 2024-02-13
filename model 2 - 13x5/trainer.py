"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Trainer. It is used to train the GAN and to generate super resolution data.

The code is called from the train.py file which is started by the main function.
"""

import json
import math
import os

import matplotlib

matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
import pandas as pd
from generator import Generator
from discriminator import Discriminator
from data import Custom_Dataset


class Trainer:
    """
    Class to train the model.
    """

    def __init__(self, num_epochs: int, batch_size: int, learning_rate: float, run_folder: str) -> None:
        """
        Constructs all the necessary attributes for the Trainer object.

        Parameters
        ----------
        num_epochs: int
                    Number of epochs to train the model
        batch_size: int
                    Batch size to use during training
        learning_rate: float
                    Learning rate to use during training
        run_folder: str
                    Path to the folder where the results should be saved
        """

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = learning_rate

        self.betas = (0.9, 0.999)

        self.run_folder = run_folder

        # Create a dictionary for the hyperparameters
        self.hyperparameter = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        print("init done")

    def prepare_data(self) -> None:
        """
        Prepares the training, validation and testing set.

        Returns
        -------
        None
        """

        # Load the JSON file with values for standardization into a Python dictionary
        with open('../dataset/mean_std.json', 'r') as file:
            self.mean_std_dict = json.load(file)

        data_train_val = pd.read_csv('../dataset/data.csv', sep=";")
        data_test = pd.read_csv('../testset/data.csv', sep=";")

        train, val = train_test_split(data_train_val, train_size=0.8, test_size=0.2, random_state=42)

        # Use Custom_Dataset with standardization
        self.training = torch.utils.data.DataLoader(
            Custom_Dataset(train, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], grid_shape=True),
            batch_size=self.batch_size
        )

        self.validation = torch.utils.data.DataLoader(
            Custom_Dataset(val, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], grid_shape=True),
            batch_size=self.batch_size
        )
        self.testing = torch.utils.data.DataLoader(
            Custom_Dataset(data_test, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], grid_shape=True))

        self.hyperparameter["num_samples"] = data_train_val.shape[0]
        self.hyperparameter["standardized"] = 'True'

        # Save the dictionary to a file
        with open(self.run_folder + '/hyperparameter.json', 'w') as file:
            json.dump(self.hyperparameter, file)

    def initialize_models(self) -> None:
        """
        Initializes the Generator, Discriminator, Optimizers and Loss functions.

        Returns
        -------
        None
        """
        self.generator = Generator()
        self.discriminator = Discriminator()

        # move to gpu in case it's possible
        self.generator.to(self.device)
        # print("Generator summary")
        # print(str(summary(self.generator, (1, 13,5,2048))))

        self.discriminator.to(self.device)
        print("Discriminator summary")
        print(str(summary(self.discriminator, (1, 13, 5, 2048))))

        self.adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.mse = nn.MSELoss()  # Mean Squared error Loss

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

    def train(self) -> tuple:
        """
        Trains the complete model for a number of epochs.

        Returns
        -------
        train_g_loss: list
                    Generator loss over the epochs
        train_d_loss: list
                    Discriminator loss over the epochs
        val_loss: list
                    Validation loss over the epochs
        psnr: list
                    PSNR over the epochs
        """
        train_d_loss = []
        train_d_loss_real = []
        train_d_loss_fake = []
        train_g_loss = []
        val_loss = []
        psnr = []
        best_val_loss = float('inf')

        for epoch in range(1, self.num_epochs + 1):
            print("epoch", epoch)
            epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d = self._train_epoch(epoch)
            avg_val_loss, avg_psnr = self._validate(epoch)

            train_g_loss.append(epoch_loss_g)
            train_d_loss.append(epoch_loss_d)
            train_d_loss_real.append(epoch_loss_d_real)
            train_d_loss_fake.append(epoch_loss_d_fake)
            val_loss.append(avg_val_loss)
            psnr.append(avg_psnr)

            # Save best model based on validation loss or other metrics
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                # Save the final models when a new best model is found
                self._save_final_models()

            self._save_checkpoint(epoch)

        return train_g_loss, train_d_loss, train_d_loss_real, train_d_loss_fake, val_loss, psnr

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
        avg_loss_d_real: float
                    Average loss of the discriminator over the training set when receiving real data
        avg_loss_d_fake: float
                    Average loss of the discriminator over the training set when receiving fake data
        avg_loss_d: float
                    Average loss of the discriminator over the training set
        """
        self.generator.train()
        self.discriminator.train()
        total_loss_g = 0.0
        total_loss_d_real = 0.0
        total_loss_d_fake = 0.0
        total_loss_d = 0.0

        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.training):
            print("batch index", batch_idx)

            # Move data to CUDA device during training or validation loops
            lr_emg = lr_emg.to(self.device)
            hr_emg = hr_emg.to(self.device)

            # Train Discriminator
            self.optimizer_D.zero_grad()

            # Generate super-resolved data (SR) from LR data
            # detaching sr_emg prevents gradients from flowing back through the generator network when computing
            # the loss for the discriminator. --> ensure that during the optimization step for the discriminator,
            # only the gradients with respect to the discriminator's parameters are computed and used to update those parameters.
            sr_emg = self.generator(lr_emg).detach().to(self.device)

            # Create real and fake labels to train discriminator for both cases
            real_labels = torch.ones(hr_emg.size(0), 1).to(self.device)
            fake_labels = torch.zeros(hr_emg.size(0), 1).to(self.device)

            # calculate loss for discriminator receiving real data
            output_discriminator_real = self.discriminator(hr_emg).to(self.device)
            loss_discriminator_real = self.adversarial_loss(output_discriminator_real, real_labels)

            # calculate loss for discriminator receiving fake data
            output_discriminator_fake = self.discriminator(sr_emg)
            loss_discriminator_fake = self.adversarial_loss(output_discriminator_fake, fake_labels)

            # calculate total loss for discriminator
            loss_discriminator = (loss_discriminator_real + loss_discriminator_fake)

            loss_discriminator.backward()

            self.optimizer_D.step()

            # Train Generator
            self.optimizer_G.zero_grad()

            # Generate SR data from LR data
            sr_emg = self.generator(lr_emg).to(self.device)

            output = self.discriminator(sr_emg)
            loss_generator_adv = self.adversarial_loss(output, real_labels)

            loss_generator_content = self.mse(sr_emg, hr_emg)  # Content loss

            loss_generator = (0.01 * loss_generator_adv + loss_generator_content)

            loss_generator.backward()

            self.optimizer_G.step()

            total_loss_g += loss_generator.item()
            total_loss_d += loss_discriminator.item()
            total_loss_d_real += loss_discriminator_real.item()
            total_loss_d_fake += loss_discriminator_fake.item()

        avg_loss_g = total_loss_g / len(self.training)
        avg_loss_d_real = total_loss_d_real / len(self.training)
        avg_loss_d_fake = total_loss_d_fake / len(self.training)
        avg_loss_d = total_loss_d / len(self.training)

        # Print progress per epoch
        print(f"Epoch [{epoch}/{self.num_epochs}] Average Loss D: {avg_loss_d}, Average Loss G: {avg_loss_g}")

        return avg_loss_g, avg_loss_d_real, avg_loss_d_fake, avg_loss_d

    def _delete_row_column(self, sr_emg: torch.tensor) -> torch.tensor:
        """
        Deletes the last row and column of the super-resolved data.

        Parameters
        ----------
        sr_emg: torch tensor

        Returns
        -------
        sr_emg_final: torch tensor
                      Super-resolved data without the last row and column
        """

        sr_emg_final = sr_emg[:, :, :-1, :-1, :]

        return sr_emg_final

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
        self.discriminator.eval()  # Set the discriminator to evaluation mode if needed

        total_validation_loss = 0.0
        total_psnr = 0.0

        with torch.no_grad():
            for val_batch_idx, (val_lr, val_hr, val_lr_path, val_hr_path) in enumerate(self.validation):
                val_lr = val_lr.to(self.device)
                val_hr = val_hr.to(self.device)

                # Generate SR data from LR data using the generator
                val_sr_emg = self.generator(val_lr).to(self.device)

                # Calculate validation loss
                val_loss = self.mse(val_sr_emg, val_hr)

                # Accumulate validation loss
                total_validation_loss += val_loss.item()

                psnr = 20 * torch.log10(1.0 / torch.sqrt(val_loss))

                total_psnr += psnr.item()

            avg_validation_loss = total_validation_loss / len(self.validation)
            print(f"Epoch [{epoch}/{self.num_epochs}] Validation Loss: {avg_validation_loss}")

            avg_psnr = total_psnr / len(self.validation)

        self.generator.train()  # Set the generator back to training mode
        self.discriminator.train()  # Set the discriminator back to training mode if needed

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
            torch.save(self.generator.state_dict(), self.run_folder + f'/generator_epoch_{epoch}.pt')
            torch.save(self.discriminator.state_dict(), self.run_folder + f'/discriminator_epoch_{epoch}.pt')

    def _save_final_models(self) -> None:
        """
        Saves the final Generator and Discriminator.

        Returns
        -------
        None
        """
        torch.save(self.generator.state_dict(), self.run_folder + '/final_generator.pt')
        torch.save(self.discriminator.state_dict(), self.run_folder + '/final_discriminator.pt')

    def _plot_result_losses(self, train_g_loss: list, train_d_loss_real: list, train_d_loss_fake: list,
                            train_d_loss: list, val_loss: list, psnr: list) -> None:
        """
        Plots the Generator and Discriminator loss over the epochs and the validation loss and PSNR over the epochs.

        Parameters
        ----------
        train_g_loss: list
        train_d_loss_real: list
        train_d_loss_fake: list
        train_d_loss: list
        val_loss: list
        psnr: list

        Returns
        -------
        None
        """

        # Convert lists to NumPy arrays
        train_g_loss_np = np.array(train_g_loss)
        train_d_loss_np = np.array(train_d_loss)
        train_d_loss_real_np = np.array(train_d_loss_real)
        train_d_loss_fake_np = np.array(train_d_loss_fake)
        val_loss_np = np.array(val_loss)
        psnr_np = np.array(psnr)

        # Save NumPy arrays to files
        np.save('train_g_loss.npy', train_g_loss_np)
        np.save('train_d_loss.npy', train_d_loss_np)
        np.save('train_d_loss_real.npy', train_d_loss_real_np)
        np.save('train_d_loss_fake.npy', train_d_loss_fake_np)
        np.save('val_loss.npy', val_loss_np)
        np.save('psnr.npy', psnr_np)

        # Plot Generator and Discriminator loss
        fig_generator_discriminator, (
            ax_generator_loss, ax_discriminator_loss, ax_discriminator_loss_r_f) = plt.subplots(nrows=3, ncols=1)

        # Plot Generator loss
        ax_generator_loss.plot(np.arange(len(train_g_loss_np)), train_g_loss_np)
        ax_generator_loss.set_yscale('log')
        ax_generator_loss.set_xlabel('Epoch')
        ax_generator_loss.set_ylabel('Loss')
        ax_generator_loss.set_title('Generator Loss over Epochs')

        # Plot Discriminator loss
        ax_discriminator_loss.plot(np.arange(len(train_d_loss_np)), train_d_loss_np)
        ax_discriminator_loss.set_yscale('log')
        ax_discriminator_loss.set_xlabel('Epoch')
        ax_discriminator_loss.set_ylabel('Loss')
        ax_discriminator_loss.set_title('Discriminator Loss over Epochs')

        # Plot Discriminator loss
        ax_discriminator_loss_r_f.plot(np.arange(len(train_d_loss_fake_np)), train_d_loss_fake_np, label='Fake labels')
        ax_discriminator_loss_r_f.plot(np.arange(len(train_d_loss_real_np)), train_d_loss_real_np, label='Real labels')
        ax_discriminator_loss_r_f.set_yscale('log')
        ax_discriminator_loss_r_f.set_xlabel('Epoch')
        ax_discriminator_loss_r_f.set_ylabel('Loss')
        ax_discriminator_loss_r_f.legend()
        ax_discriminator_loss_r_f.set_title('Discriminator Loss over Epochs')

        fig_generator_discriminator.tight_layout()
        fig_generator_discriminator.savefig(self.run_folder + '/training_losses.png')

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
        fig_val_psnr.savefig(self.run_folder + '/validation_loss_psnr.png')

    def generate_sr_data(self) -> None:
        """
        Generates super-resolved data from the test dataset and saves it to a file.

        Returns
        -------

        """
        # in case you only want to generate data:
        # self.run_folder = "runs/20240117172449"

        # load the final generator
        self.generator.load_state_dict(torch.load(self.run_folder + '/final_generator.pt'))
        self.generator.eval()  # Set the generator to evaluation mode

        with torch.no_grad():
            for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
                lr_emg = lr_emg.to(self.device)
                hr_emg = hr_emg.to(self.device)

                # Generate super-resolved data (SR) from LR data
                lr_sr_emg = self.generator(lr_emg).detach().to(self.device)
                lr_sr_emg = lr_sr_emg.to('cpu').squeeze().numpy()
                lr_sr_emg_destandardized = self.destandardize_data(lr_sr_emg)

                # Generate super-resolved data (SR) from HR data
                hr_sr_emg = self.generator(hr_emg).detach().to(self.device)
                hr_sr_emg = hr_sr_emg.to('cpu').squeeze().numpy()
                hr_sr_emg_destandardized = self.destandardize_data(hr_sr_emg)

                # Save signal from lr to hr
                last_slash_index_lr = lr_emg_paths[0].rfind("/")
                model_path = lr_emg_paths[0][:last_slash_index_lr] + '/model2_13x5'
                os.makedirs(model_path, exist_ok=True)
                np.save(model_path + (lr_emg_paths[0][last_slash_index_lr:]).replace('-lr.npy', '-lr-sr.npy'),
                        lr_sr_emg_destandardized
                        )

                # Save B-Spline interpolated signal from hr to sr
                last_slash_index_hr = hr_emg_paths[0].rfind("/")
                np.save(model_path + (hr_emg_paths[0][last_slash_index_hr:]).replace('-hr.npy', '-hr-sr.npy'),
                        hr_sr_emg_destandardized)

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
