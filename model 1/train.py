import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from generator import Generator
from discriminator import Discriminator
from data import Dataset


class Trainer:
    def __init__(self):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 20
        self.batch_size = 8
        self.lr = 0.001
        self.betas = (0.9, 0.999)

    def prepare_data(self):
        data = pd.read_csv('dataset/data/Dominik/offline/20230720-085154/data.csv', sep=";")
        train, test = train_test_split(data, train_size=0.75, test_size=0.25, random_state=42)
        self.training = torch.utils.data.DataLoader(Dataset(train, "train"), batch_size=self.batch_size)
        self.validation = torch.utils.data.DataLoader(Dataset(test, "val"), batch_size=self.batch_size)

    def initialize_models(self):
        self.generator = Generator(in_channels=1)
        self.discriminator = Discriminator(in_channels=1)

        self.adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.content_loss = nn.MSELoss()  # Mean Squared error Loss

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

    def train(self):
        train_D_loss = []
        train_G_loss = []
        val_loss = []
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            epoch_loss_G, epoch_loss_D = self._train_epoch(epoch)
            avg_val_loss = self._validate(epoch)

            train_G_loss.append(epoch_loss_G.item())
            train_D_loss.append(epoch_loss_D.item())
            val_loss.append(avg_val_loss)

            # Save best model based on validation loss or other metrics
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.generator.state_dict(), 'pytorch files/best_generator.pt')
                torch.save(self.discriminator.state_dict(), 'pytorch files/best_discriminator.pt')

                # Save the final models when a new best model is found
                self._save_final_models()

            self._save_checkpoint(epoch)

        return train_G_loss, train_D_loss, val_loss

    def _train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        for batch_idx, (lr_images, hr_images) in enumerate(self.training):

            ### Train Discriminator ###
            self.optimizer_D.zero_grad()

            # Generate super-resolved images (SR) from LR images
            sr_images = self.generator(lr_images).detach()

            # Train discriminator with HR images
            real_labels = torch.ones(hr_images.size(0), 1)
            fake_labels = torch.zeros(hr_images.size(0), 1)

            output_discriminator_real = self.discriminator(hr_images)
            loss_discriminator_real = self.adversarial_loss(output_discriminator_real, real_labels)

            output_discriminator_fake = self.discriminator(sr_images)
            loss_discriminator_fake = self.adversarial_loss(output_discriminator_fake, fake_labels)

            loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) * 0.5
            loss_discriminator.backward()
            self.optimizer_D.step()

            ### Train Generator ###
            self.optimizer_G.zero_grad()

            # Generate SR images from LR images
            sr_images = self.generator(lr_images)

            output = self.discriminator(sr_images)
            loss_generator_adv = self.adversarial_loss(output, real_labels)

            loss_generator_content = self.content_loss(sr_images, hr_images)  # Content loss

            loss_generator = loss_generator_adv + 0.01 * loss_generator_content  # Weighted sum
            loss_generator.backward()
            self.optimizer_G.step()

            # Print progress
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{self.num_epochs}] Loss D: {loss_discriminator.item()}, Loss G: {loss_generator.item()}")

            return loss_generator / len(self.training), loss_discriminator / len(self.training)

    def _validate(self, epoch):
        self.generator.eval()  # Set the generator to evaluation mode
        self.discriminator.eval()  # Set the discriminator to evaluation mode if needed

        total_validation_loss = 0.0

        with torch.no_grad():
            for val_batch_idx, (val_lr_images, val_hr_images) in enumerate(self.validation):
                # Generate SR images from LR images using the generator
                val_sr_images = self.generator(val_lr_images)

                # Calculate validation loss (e.g., pixel-wise loss or other metrics)
                val_loss = self.content_loss(val_sr_images, val_hr_images)

                # Accumulate validation loss
                total_validation_loss += val_loss.item()

        avg_validation_loss = total_validation_loss / len(self.validation)
        print(f"Epoch [{epoch}/{self.num_epochs}] Validation Loss: {avg_validation_loss}")

        self.generator.train()  # Set the generator back to training mode
        self.discriminator.train()  # Set the discriminator back to training mode if needed

        return avg_validation_loss

    def _save_checkpoint(self, epoch):
        if epoch % 5 == 0:
            torch.save(self.generator.state_dict(), f'pytorch files/generator_epoch_{epoch}.pt')
            torch.save(self.discriminator.state_dict(), f'pytorch files/discriminator_epoch_{epoch}.pt')

    def _save_final_models(self):
        torch.save(self.generator.state_dict(), 'pytorch files/final_generator.pt')
        torch.save(self.discriminator.state_dict(), 'pytorch files/final_discriminator.pt')


if __name__ == "__main__":
    trainer = Trainer()
    trainer.prepare_data()
    trainer.initialize_models()
    train_G_loss, train_D_loss, val_loss = trainer.train()

    # Convert lists to NumPy arrays
    train_G_loss_np = np.array(train_G_loss)
    train_D_loss_np = np.array(train_D_loss)
    val_loss_np = np.array(val_loss)

    # plot the results
    plt.plot(np.arange(len(train_G_loss_np)), train_G_loss_np, label='train Generator loss')
    plt.plot(np.arange(len(train_D_loss_np)), train_D_loss_np, label='train Discriminator loss')
    plt.plot(np.arange(len(val_loss_np)), val_loss_np, label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('losses.png')
    plt.show()