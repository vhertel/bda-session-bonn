# Post-event building damage assessment

This repository is part of a lecture series at the University of Bonn and is designed as an introduction to deep
learning methods in the remote sensing domain. It contains code and resources for the development of an AI model that
performs rapid building damage assessment. Deep learning techniques are utilized to analyze satellite or aerial
imagery and predict the level of damage to buildings after natural disasters or other events. After training, the model
can be used to analyze large-scale disaster areas and help prioritize rescue and recovery efforts. Additional materials
on the workflow and further background information can be found in the
dedicated presentation ([res/background.pdf](https://github.com/vhertel/bda-session-bonn/tree/main/res/background.pdf)).

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Workflow](#workflow)

## Installation

### Software

The following software packages or equivalents have to be installed.

- **Integrated development environment (IDE)**  
  An IDE is a software application that provides comprehensive tools and features to aid programmers in developing,
  testing, and debugging software. While all IDEs for Python can be
  used, [PyCharm](https://www.jetbrains.com/de-de/pycharm/) offers a user-friendly, powerful, and free option.


- **conda package manager**  
  The conda package manager is an open-source package management system primarily used for managing software
  environments. Conda simplifies package management and environment creation, avoiding compatibility issues and ensuring
  that all the necessarey dependencies are properly installed. It is widely used in the data science and machine
  learning communities due to its robustness and extensive package ecosystem. To install the conda package manager,
  either [Miniforge](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
  can be downloaded.


- **Git**  
  The distributed version control system (VCS) [Git](https://git-scm.com/downloads) is widely used for tracking changes
  in source code during software development. It allows multiple developers to collaborate on a project, managing and
  merging their code efficiently.

### Repository

After successful installation of the software above, follow the instructions to download and set up the repository.

1. Clone the repository using the `git CMD` application:

```
cd path-to-target-directory
git clone https://github.com/vhertel/bda-session-bonn.git
```

2. Install the repository environment using `miniforge prompt` or `Anaconda` application:

```
cd path-to-repository
conda env create -f environment.yml
```

3. Open the project and select the conda environment `bda` as interpreter. In PyCharm, this can be done via `File ->
   Settings... -> Project: bda-session-bonn -> Project Interpreter -> Add Interpreter -> Add Local Interpreter... -> Conda Environment`
   . Click on the `...` button next to the `Interpreter` field to open the `Select Python Interpreter` dialog. Select
   the `python.exe` of the desired environment. PyCharm will now use the selected Conda environment as the interpreter
   for your project.

## Dataset

The model is trained on the xBD dataset, which is a valuable resource consisting of pairs of pre-disaster and
post-disaster satellite imagery with corresponding damage labels on building level. By utilizing the xBD dataset,
researchers and developers can advance the field of building damage assessment by leveraging machine learning techniques
to automatically analyze satellite imagery and provide accurate and timely information about the extent of damage after
natural disasters.

## Workflow

This repository hosts a U-Net model that has been pre-trained on the xBD dataset. The model underwent 25 epochs of
training, excluding volcanic events. The main purpose of this session is to utilize the pre-trained model for assessing
building damage caused by volcanic eruptions. Subsequently, the model will be re-trained using a specific subset of the
xBD dataset that comprises volcanic eruption events. The ultimate goal is to observe improved accuracy when performing
inference with the re-trained model.

For doing so, follow the instructions:

1. Download the [xBD data subset](https://dlrmax.dlr.de/get/be2d2cf5-7e7e-5e11-85fb-f632f5db20d3/) for training and
   inference (link expires on 26/07/2023).
2. Open `inference.py`, adjust the *CONFIG* parameters and run the script. Note the macro and micro F1 scores after the
   process. Note: if memory errors occur, reduce the batch size.
3. Open `train.py`, adjust the *CONFIG* parameters and run the script. During the training process, model weights are
   stored under `res/models/` in the format *epoch-f1_macro-loss.pth.tar*.
4. Open `inference.py`, adjust the *CONFIG* parameters and run the script again. Make sure to select the model with
   highest macro F1 score of the training in step 3. Note and compare the macro and micro F1
   scores after the process.

During the training process, the evaluation of the loss as well as macro and micro F1 scores are stored
under `res/logs/`. Example tiles with the size of 256x256 pixels are plotted and stored under `res/output/` during
inference.
