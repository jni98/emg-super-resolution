"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Trainer. It is used to train the GAN and to generate super resolution
data.

The code is called from the train.py file which is started by the main function.
"""

import json
import os
from datetime import datetime

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
from discriminator import Discriminator
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
        batch_size: int
                    Batch size to use during training
        learning_rate: float
                    Learning rate to use during training
        """

        self.data_directory = data_directory
        self.test_data_directory = test_data_directory
        self.run_folder = results_directory
        self.num_epochs = num_epochs
        self.config = config
        self.sweep_id = sweep_id

        self.lr = config.learning_rate
        self.batch_size = config.batch_size

        self.unique_folder = '/' + datetime.now().strftime('%Y%m%d%H%M%S')

        self.betas = (0.9, 0.999)

        # Create a dictionary for the hyperparameters
        self.hyperparameter = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        wandb.init(project="emg-super-resolution", entity="masterthesis-sr")

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
        print("Creating folder")
        print(self.run_folder + self.unique_folder)
        os.makedirs(self.run_folder + self.unique_folder, exist_ok=True)

        # Save the dictionary to a file
        with open(self.run_folder + self.unique_folder + '/hyperparameter.json', 'w') as file:
            json.dump(self.hyperparameter, file)

        print("Successfully prepared the data")

    def initialize_models(self) -> None:
        """
        Initializes the Generator, Discriminator, Optimizers and Loss functions.

        Returns
        -------
        None
        """
        self.generator = Generator(self.config.num_res_blocks, self.config.num_res_blocks_last, in_channels=1)
        self.discriminator = Discriminator()

        # Inside your Trainer class __init__ method, wrap your model with nn.DataParallel
        self.generator = nn.DataParallel(self.generator)
        self.discriminator = nn.DataParallel(self.discriminator)

        # Ensure that your model and data are moved to the appropriate device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # print("Generator summary")
        # print(str(summary(self.generator, (1, 7,3,2048))))

        # print("Discriminator summary")
        # print(str(summary(self.discriminator, (1, 13, 5, 2048))))

        self.adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.mse = nn.MSELoss()  # Mean Squared error Loss

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

        print("Successfully initialized the models")

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

        # Add wandb.watch() here to start tracking gradients and parameters
        wandb.watch(self.generator, criterion='mse', log='all')
        wandb.watch(self.discriminator, criterion='mse', log='all')

        for epoch in range(1, self.num_epochs + 1):
            print("epoch", epoch)

            # Check GPU memory usage before training
            initial_allocated = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_cached()

            epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d = self._train_epoch(epoch)
            avg_val_loss, avg_psnr = self._validate(epoch)

            train_g_loss.append(epoch_loss_g)
            train_d_loss.append(epoch_loss_d)
            train_d_loss_real.append(epoch_loss_d_real)
            train_d_loss_fake.append(epoch_loss_d_fake)
            val_loss.append(avg_val_loss)
            psnr.append(avg_psnr)

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_final_models()  # Save the final models when a new best model is found

            # Clear unnecessary variables to release memory
            del epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d, avg_val_loss, avg_psnr

            # Check GPU memory usage after training
            allocated_diff = torch.cuda.memory_allocated() - initial_allocated
            cached_diff = torch.cuda.memory_cached() - initial_cached

            print(f"After epoch {epoch}, GPU memory allocated increased by: {allocated_diff} bytes")
            print(f"After epoch {epoch}, GPU memory cached increased by: {cached_diff} bytes")

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

            with torch.no_grad():  # Disable gradient calculation within this block
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

        wandb.log({'epoch': epoch, 'avg_loss_g': avg_loss_g, 'avg_loss_d_real': avg_loss_d_real,
                   'avg_loss_d_fake': avg_loss_d_fake, 'avg_loss_d': avg_loss_d})

        # Print progress per epoch
        print(f"Epoch [{epoch}/{self.num_epochs}] Average Loss D: {avg_loss_d}, Average Loss G: {avg_loss_g}")

        return avg_loss_g, avg_loss_d_real, avg_loss_d_fake, avg_loss_d

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
        total_abs_diff = 0.0

        with torch.no_grad():
            for val_batch_idx, (val_lr, val_hr, val_lr_path, val_hr_path) in enumerate(self.validation):
                print("validation batch index", val_batch_idx)

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

                # destacndardize the data for the logged plots
                val_hr_destandardized = self.destandardize_data(val_hr)
                val_sr_emg_destandardized = self.destandardize_data(val_sr_emg)

                # log absolute difference for one channel absolute difference
                abs_diff = torch.abs(val_hr_destandardized[0][0][6][2] - val_sr_emg_destandardized[0][0][6][2])
                abs_diff = abs_diff.squeeze().to('cpu')
                abs_diff = np.asarray(abs_diff)
                total_abs_diff += abs_diff

                if val_batch_idx == 0:
                    ############################## Log Plots given channel########################################
                    print("Logging plots for batch index 0 given channel")

                    # x-axis for the following logs
                    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
                    x_values = range(val_hr.shape[4])

                    # log absolute difference for one channel
                    axs[0].plot(x_values, abs_diff)
                    axs[0].set_title('Absolute difference given channel')

                    # Log one channel in the middle of the grid
                    val_channel = val_hr_destandardized[0][0][6][2]
                    val_channel = val_channel.to('cpu')
                    val_channel = np.asarray(val_channel)
                    axs[1].plot(x_values, val_channel)
                    axs[1].set_title('Validation channel [6][2]')

                    # Log one channel in the middle of the grid
                    sr_channel = val_sr_emg_destandardized[0][0][6][2]
                    sr_channel = sr_channel.to('cpu')
                    sr_channel = np.asarray(sr_channel)
                    axs[2].plot(x_values, sr_channel)
                    axs[2].set_title('Super-resolution channel [6][2]')

                    # log cross correlation
                    auto_correlation = np.correlate(val_channel, val_channel, mode='full')
                    cross_correlation = (np.correlate(val_channel, sr_channel, mode='full')) / np.max(auto_correlation)
                    cross_correlation = cross_correlation.squeeze()
                    axs[3].plot(range(len(cross_correlation)), cross_correlation)
                    axs[3].set_title('Cross Correlation channel [6][2]')

                    plt.tight_layout()
                    wandb.log({"Logged plots": [wandb.Image(fig)]})

                    # Close the figure
                    plt.close(fig)

                    ############################## Log Plots predicted channel########################################
                    print("Logging plots for batch index 0 predicted channel")

                    # x-axis for the following logs
                    fig1, axs1 = plt.subplots(4, 1, figsize=(8, 10))
                    x_values = range(val_hr.shape[4])

                    # log absolute difference for one channel
                    axs1[0].plot(x_values, torch.abs(
                        val_hr_destandardized[0][0][6][1] - val_sr_emg_destandardized[0][0][6][1]).squeeze().to('cpu'))
                    axs1[0].set_title('Absolute difference predicted channel')

                    # Log one channel in the middle of the grid
                    val_channel = val_hr_destandardized[0][0][6][1]
                    val_channel = val_channel.to('cpu')
                    val_channel = np.asarray(val_channel)
                    axs1[1].plot(x_values, val_channel)
                    axs1[1].set_title('Validation channel [6][1]')

                    # Log one channel in the middle of the grid
                    sr_channel = val_sr_emg_destandardized[0][0][6][1]
                    sr_channel = sr_channel.to('cpu')
                    sr_channel = np.asarray(sr_channel)
                    axs1[2].plot(x_values, sr_channel)
                    axs1[2].set_title('Super-resolution channel [6][1]')

                    # log cross correlation
                    auto_correlation = np.correlate(val_channel, val_channel, mode='full')
                    cross_correlation = (np.correlate(val_channel, sr_channel, mode='full')) / np.max(auto_correlation)
                    cross_correlation = cross_correlation.squeeze()
                    axs1[3].plot(range(len(cross_correlation)), cross_correlation)
                    axs1[3].set_title('Cross Correlation channel [6][1]')

                    plt.tight_layout()
                    wandb.log({"Logged plots": [wandb.Image(fig1)]})

                    # Close the figure
                    plt.close(fig1)

            # log validation loss and psnr
            avg_validation_loss = total_validation_loss / len(self.validation)
            avg_psnr = total_psnr / len(self.validation)
            wandb.log({'val_loss': avg_validation_loss})
            wandb.log({'psnr': avg_psnr})

            # log absolute difference for one channel absolute difference
            avg_abs_diff = total_abs_diff / len(self.validation)
            fig2, axs2 = plt.subplots(1, 1, figsize=(8, 10))
            axs2.plot(range(len(avg_abs_diff)), avg_abs_diff)
            axs2.set_title('Average absolute difference')
            plt.tight_layout()
            wandb.log({"Average absolute difference": [wandb.Image(fig2)]})
            wandb.log({'Mean absolute difference': np.mean(avg_abs_diff)})

            # Close the figure
            plt.close(fig2)

            print(f"Epoch [{epoch}/{self.num_epochs}] Validation Loss: {avg_validation_loss}")

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
            torch.save(self.generator.state_dict(),
                       self.run_folder + f'/generator_epoch_{epoch}.pt')
            torch.save(self.discriminator.state_dict(),
                       self.run_folder + f'/discriminator_epoch_{epoch}.pt')

    def _save_final_models(self) -> None:
        """
        Saves the final Generator and Discriminator.

        Returns
        -------
        None
        """
        torch.save(self.generator.state_dict(), self.run_folder + self.unique_folder + '/final_generator.pt')
        torch.save(self.discriminator.state_dict(), self.run_folder + self.unique_folder + '/final_discriminator.pt')

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
        np.save(self.run_folder + self.unique_folder + '/train_g_loss.npy', train_g_loss_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss.npy', train_d_loss_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss_real.npy', train_d_loss_real_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss_fake.npy', train_d_loss_fake_np)
        np.save(self.run_folder + self.unique_folder + '/val_loss.npy', val_loss_np)
        np.save(self.run_folder + self.unique_folder + '/psnr.npy', psnr_np)

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
        fig_generator_discriminator.savefig(self.run_folder + self.unique_folder + '/training_losses.png')

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
        fig_val_psnr.savefig(self.run_folder + self.unique_folder + '/validation_loss_psnr.png')

    def generate_sr_data(self) -> None:
        """
        Generates super-resolved data from the test dataset and saves it to a file.

        Returns
        -------

        """
        #free memory
        del self.discriminator
        del self.optimizer_G
        del self.optimizer_D
        del self.adversarial_loss
        del self.mse
        del self.training
        del self.validation
        torch.cuda.empty_cache()

        print("Starting to generate SR data")
        # load the final generator
        self.generator.load_state_dict(torch.load(self.run_folder + self.unique_folder + '/final_generator.pt'))
        self.generator.eval()  # Set the generator to evaluation mode

        with torch.no_grad():
            for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
                lr_emg = lr_emg.to(self.device)
                hr_emg = hr_emg.to(self.device)

                # Generate super-resolved data (SR) from LR data
                lr_sr_emg = self.generator(lr_emg).to('cpu').numpy().squeeze()
                lr_sr_emg_destandardized = self.destandardize_data(lr_sr_emg)

                # Generate super-resolved data (SR) from HR data
                hr_sr_emg = self.generator(hr_emg).to('cpu').numpy().squeeze()
                hr_sr_emg_destandardized = self.destandardize_data(hr_sr_emg)

                # Save signal from lr to sr and hr to sr
                model_path = self.run_folder + self.unique_folder + '/results/'
                os.makedirs(model_path, exist_ok=True)
                np.save(model_path + lr_emg_paths[0], lr_sr_emg_destandardized)
                np.save(model_path + hr_emg_paths[0], hr_sr_emg_destandardized)

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
