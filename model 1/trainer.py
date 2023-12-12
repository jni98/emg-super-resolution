import json
import os

import matplotlib
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from generator import Generator
from discriminator import Discriminator
from data import Dataset

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Trainer:
    """
    A class to train a GAN for super-resolution.

    Attributes
    ----------
    epoch : int
        Current epoch of the training process

    Methods
    -------
    prepare_data:
        Prepares training and validation set.
    initialize_models:
        Initializes model, optimizer and loss functions.
    train:
        Trains the complete model for a number of epochs.
    _train_epoch(epoch):
        Trains the model for one epoch.
    _validate(epoch):
        Validates the model.
    _save_checkpoint(epoch):
        Saves the generator and discriminator state at a given epoch.
    _save_final_models:
        Saves the final generator and discriminator.

    """

    def __init__(self, num_epochs, batch_size, learning_rate):
        """
        Constructs all the necessary attributes for the Trainer object.

        Parameters
        ----------
        num_epochs: int
                    Number of epochs that the model will be trained
        batch_size: int
                    Number of samples that are in one batch.
        learning_rate: float
                       Learning rates for the optimizers.
        """

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = learning_rate

        self.betas = (0.9, 0.999)

        # Create a directory for the run if it doesn't exist
        self.run_folder = 'runs/' + datetime.now().strftime('%Y%m%d%H%M%S')
        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)

        # Create a dictionary for the hyperparameters
        self.hyperparameter = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr
        }

        print("init done")

    def prepare_data(self):
        """
        Prepares the training and validation dataset from a given csv file.
        Returns
        -------
        None
        """
        data = pd.read_csv('../dataset/data.csv', sep=";")
        train, test = train_test_split(data, train_size=0.75, test_size=0.25, random_state=42)
        self.training = torch.utils.data.DataLoader(Dataset(train, "train"), batch_size=self.batch_size)
        self.validation = torch.utils.data.DataLoader(Dataset(test, "val"), batch_size=self.batch_size)

        self.hyperparameter["num_samples"] = data.shape[0]
        # Save the dictionary to a file
        with open(self.run_folder + '/hyperparameter.json', 'w') as file:
            json.dump(self.hyperparameter, file)

    def initialize_models(self):
        """
        Initializes the generator and discriminator, as well as the loss functions and optimizers.
        Returns
        -------
        None
        """
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.content_loss = nn.MSELoss()  # Mean Squared error Loss

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

    def train(self):
        """
        Trains the Generator and Discriminator for a defined number of epochs and validates the current model.
        It saves the best and final model.

        Returns
        -------
        None
        """
        train_d_loss = []
        train_g_loss = []
        val_loss = []
        psnr = []
        best_val_loss = float('inf')

        for epoch in range(1,self.num_epochs+1):
            print("epoch",epoch)
            epoch_loss_g, epoch_loss_d = self._train_epoch(epoch)
            avg_val_loss, avg_psnr = self._validate(epoch)

            train_g_loss.append(epoch_loss_g)
            train_d_loss.append(epoch_loss_d)
            val_loss.append(avg_val_loss)
            psnr.append(avg_psnr)

            # Save best model based on validation loss or other metrics
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                # Save the final models when a new best model is found
                self._save_final_models()

            self._save_checkpoint(epoch)

        return train_g_loss, train_d_loss, val_loss, psnr

    def _train_epoch(self, epoch):
        """
        Runs the training step for one epoch for the generator and discriminator.
        Parameters
        ----------
        epoch: int
               current epoch

        Returns
        -------
        avg_loss_g: float
                    Generator loss averaged over the batch
        avg_loss_d: float
                    Discriminator loss averaged over the batch
        """
        self.generator.train()
        self.discriminator.train()
        total_loss_g = 0.0
        total_loss_d = 0.0

        for batch_idx, (lr_emg, hr_emg,lr_emg_paths, hr_emg_paths) in enumerate(self.training):
            print("batch index", batch_idx)
            # Train Discriminator
            self.optimizer_D.zero_grad()

            # Generate super-resolved data (SR) from LR data
            sr_emg = self.generator(lr_emg).detach()

            # Train discriminator with HR data
            real_labels = torch.ones(hr_emg.size(0), 1)
            fake_labels = torch.zeros(hr_emg.size(0), 1)

            output_discriminator_real = self.discriminator(hr_emg)
            loss_discriminator_real = self.adversarial_loss(output_discriminator_real, real_labels)

            output_discriminator_fake = self.discriminator(sr_emg)
            loss_discriminator_fake = self.adversarial_loss(output_discriminator_fake, fake_labels)

            loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) * 0.5
            loss_discriminator.backward()
            self.optimizer_D.step()

            # Train Generator
            self.optimizer_G.zero_grad()

            # Generate SR data from LR data
            sr_emg = self.generator(lr_emg)

            output = self.discriminator(sr_emg)
            loss_generator_adv = self.adversarial_loss(output, real_labels)

            loss_generator_content = self.content_loss(sr_emg, hr_emg)  # Content loss

            loss_generator = loss_generator_adv + 0.01 * loss_generator_content
            loss_generator.backward()
            self.optimizer_G.step()

            total_loss_g += loss_generator.item()
            total_loss_d += loss_discriminator.item()

        avg_loss_g = total_loss_g / len(self.training)
        avg_loss_d = total_loss_d / len(self.training)

        # Print progress per epoch
        print(f"Epoch [{epoch}/{self.num_epochs}] Average Loss D: {avg_loss_d}, Average Loss G: {avg_loss_g}")

        return avg_loss_g, avg_loss_d

    def _validate(self, epoch):
        """
        Validates the current model with the validation dataset.

        Parameters
        ----------
        epoch

        Returns
        -------
        avg_validation_loss: float
                             Validation loss averaged over the batch.
        """
        self.generator.eval()  # Set the generator to evaluation mode
        self.discriminator.eval()  # Set the discriminator to evaluation mode if needed

        total_validation_loss = 0.0
        total_psnr = 0.0

        with torch.no_grad():
            for val_batch_idx, (val_lr, val_hr, val_lr_path, val_hr_path) in enumerate(self.validation):
                # Generate SR data from LR data using the generator
                val_sr_emg = self.generator(val_lr)

                # Calculate validation loss
                val_loss = self.content_loss(val_sr_emg, val_hr)

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

    def _save_checkpoint(self, epoch):
        """
        Saves the state of the Generator and Discriminator for a given epoch.

        Parameters
        ----------
        epoch: int
               Current epoch

        Returns
        -------
        None
        """
        if epoch % 5 == 0:
            torch.save(self.generator.state_dict(), self.run_folder + f'/generator_epoch_{epoch}.pt')
            torch.save(self.discriminator.state_dict(), self.run_folder + f'/discriminator_epoch_{epoch}.pt')

    def _save_final_models(self):
        """
        Saves the final model.

        Returns
        -------
        None
        """
        torch.save(self.generator.state_dict(), self.run_folder + '/final_generator.pt')
        torch.save(self.discriminator.state_dict(), self.run_folder + '/final_discriminator.pt')

    def _plot_result_losses(self, train_g_loss, train_d_loss, val_loss, psnr):
        """
        Shows and saves a plot of the losses of the Generator during training, the Discriminator during training
        and the validation loss.

        Parameters
        ----------
        train_g_loss: list
                      Averaged losses per epoch for the generator.
        train_d_loss: list
                      Averaged losses per epoch for the discriminator.
        val_loss: list
                  Averaged validation losses.
        psnr: list
              PSNR values per epoch.

        Returns
        -------
        None
        """

        # Convert lists to NumPy arrays
        train_g_loss_np = np.array(train_g_loss)
        train_d_loss_np = np.array(train_d_loss)
        val_loss_np = np.array(val_loss)
        psnr_np = np.array(psnr)

        # Plot Training losses in a separate figure
        fig_train_losses, ax_train_losses = plt.subplots()
        ax_train_losses.plot(np.arange(len(train_g_loss_np)), train_g_loss_np, label='Train Generator loss')
        ax_train_losses.plot(np.arange(len(train_d_loss_np)), train_d_loss_np, label='Train Discriminator loss')
        ax_train_losses.set_yscale('log')
        ax_train_losses.set_xlabel('Epoch')
        ax_train_losses.set_ylabel('Loss')
        ax_train_losses.set_title('Losses over Epochs')
        ax_train_losses.legend()
        fig_train_losses.savefig(self.run_folder + '/training_losses.png')
        plt.show(fig_train_losses)

        # Plot Validation loss, PSNR in a combined figure
        fig_val_psnr, (ax_val_loss, ax_psnr) = plt.subplots(nrows=2, ncols=1)

        # Plot Validation loss
        ax_val_loss.plot(np.arange(len(val_loss_np)), val_loss_np, label='Validation loss')
        ax_val_loss.set_xlabel('Epoch')
        ax_val_loss.set_ylabel('Loss')
        ax_val_loss.set_title('Validation Loss over Epochs')
        ax_val_loss.legend()

        # Plot PSNR
        ax_psnr.plot(np.arange(len(psnr_np)), psnr_np, label='PSNR')
        ax_psnr.set_xlabel('Epoch')
        ax_psnr.set_ylabel('PSNR')
        ax_psnr.set_title('PSNR over Epochs')
        ax_psnr.legend()

        fig_val_psnr.tight_layout()
        fig_val_psnr.savefig(self.run_folder + '/validation_loss_psnr.png')
        plt.show(fig_val_psnr)

    def generate_sr_data(self):
        """
        Generates super-resolved (SR) data for the test set.

        Parameters
        ----------
        test_set_csv: str
            Path to the CSV file containing information about the test set.

        Returns
        -------
        None
        """
        # load the final generator
        self.generator.load_state_dict(torch.load('runs/20231206115019/final_generator.pt'))
        self.generator.eval()  # Set the generator to evaluation mode

        data = pd.read_csv('../testset/data.csv', sep=";")
        self.testing = torch.utils.data.DataLoader(Dataset(data, "test"))

        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
            # Generate super-resolved data (SR) from LR data
            lr_sr_emg = self.generator(lr_emg).detach()
            lr_sr_emg = lr_sr_emg.squeeze().numpy()

            # Generate super-resolved data (SR) from LR data
            hr_sr_emg = self.generator(hr_emg).detach()
            hr_sr_emg = hr_sr_emg.squeeze().numpy()

            # Save signal from lr to hr
            last_slash_index_lr = lr_emg_paths[0].rfind("/")
            model1_path = lr_emg_paths[0][:last_slash_index_lr] + '/model1'
            os.makedirs(model1_path, exist_ok=True)
            np.save(model1_path + (lr_emg_paths[0][last_slash_index_lr:]).replace('-lr.npy', '-lr-sr.npy'), lr_sr_emg
                    )

            # Save B-Spline interpolated signal from hr to sr
            last_slash_index_hr = hr_emg_paths[0].rfind("/")
            np.save(model1_path + (hr_emg_paths[0][last_slash_index_hr:]).replace('-hr.npy', '-hr-sr.npy'),
                    hr_sr_emg)