<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">GAN-Based Neural Networks Project</h3>

  <p align="center">
    This project contains four different GAN-based neural network models and two CNN-LSTM to generate hig-resolution data from low-resolution input. The project is organized into separate folders for each model and includes scripts for training, data preparation, and visualization. The training can be performed both locally and on a cluster.
    <br />
    <a href="https://github.com/github_username/repo_name/-/wikis/home"><strong>Explore the docs »</strong></a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

This project contains four different GAN-based neural network models and two CNN-LSTM. The project is organized into
separate folders for each model and includes scripts for training, data preparation, and visualization. The training can
be performed both locally and on a cluster. Each model can be trained using only the files in that folder.

### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [WandB](https://www.wandb.com/)

<!-- CONTACT -->

### Contact

Janina Kloos - janina.kloos@fau.de


<!-- GETTING STARTED -->

## Getting Started

### Installation

1. Create environment using the enviroment.yaml file

   ```conda env create --name my_custom_env --file environment.yaml```

2. Activate the new environment

   ```conda activate my_custom_env```

3. Clone the repository

   ```git clone https://gitos.rrze.fau.de/n-squared-lab/students/master-theses/janina-kloos/code.git```

### Prerequisites

- To use the wandb logging, create an account on [WandB](https://docs.wandb.ai/quickstart) and login using the command
  `wandb login`.

<!-- USAGE EXAMPLES -->

## Usage

To run each model, submit a job to the cluster using the `train-script.sh` file located in each model's directory. To
run the models locally, modify `train.py` by commenting and uncommenting the relevant blocks of code in the main method.
Directories for the train and test dataset have to be specified accordingly in the main or train-script.sh file.

### Project Structure

The project is structured as follows:

```
project_root/
│
├── model1-SRGAN/
│   ├── data.py
│   ├── discriminator.py
│   ├── emg_augmentations.py
│   ├── generator.py
│   ├── train.py
│   ├── train-script.sh
│   ├── trainer.py
│   └── utils.py
│
├── model2-UNet/
│   ├── ...
│
├── model2-UNet-Fourier/
│   ├── ...
│
├── model2-UNet-Wavelet/
│   ├── ...
│
├── model3-LSTM/
│   ├── checkpoint_run.py
│   ├── config.py
│   ├── data.py
│   ├── emg_augmentations.py
│   ├── model.py
│   ├── submit-jobs.sh
│   ├── train-lstm.sh
│   ├── train-lstm-continue.sh
│   └── trainer.py
│    
├── model3-LSTM-basic/
│   ├── checkpoint_run.py
│   ├── config.py
│   ├── data.py
│   ├── emg_augmentations.py
│   ├── model_basic.py
│   ├── submit-jobs.sh
│   ├── train-lstm.sh
│   ├── train-lstm-continue.sh
│   └── trainer.py
│
├── data_extractor.py
├── numpy_to_mat.py
└── Visualization/
    └── visualization.py
```

### Data Preparation

Run `data_extractor.py` to create the dataset files. Specify the file locations in the main method, which will create a
directory containing the training and test data, a CSV file, and a tar file. Window length and file name can be defined
in the function call. The tar file can be used when using the dataset on the HPC.

### Results and Visualization

- Results generated on the cluster should be copied to a directory called `results`, with subdirectories for each model.
- Run `numpy_mat.py` to generate MATLAB files with a reference signal for decomposition. Edit files in main method
  accordingly.
- Run `visualization.py` in the `Visualization` directory to create plots from the `results` folder.

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

Resources and libraries used in this project:

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [WandB](https://www.wandb.com/)
* [Scikit-Learn](https://scikit-learn.org/)
* [Matplotlib](https://matplotlib.org/)
* [Plotly](https://plotly.com/)
* [Raul Simpetru for augmentation code]

<!-- CONTACT -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>
