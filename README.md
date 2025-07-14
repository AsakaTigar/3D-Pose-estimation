# 3D Human Pose Estimation in PyTorch

This repository contains a PyTorch implementation of a 3D human pose estimation model. It is based on the architectural foundations laid by the works of Martinez et al. and Pavllo et al., offering a streamlined and easy-to-use framework for training and evaluating 3D pose models.

![demo-gif](https://user-images.githubusercontent.com/your-username/your-repo/your-gif.gif) 
*An example GIF demonstrating the model's performance. Please replace this with your own GIF.*

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Training](#training)
- [Testing](#testing)
- [Generating Visualizations](#generating-visualizations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Requirements

First, ensure you have Python and Conda installed. Then, you can create and activate a Conda environment and install the necessary dependencies using the following commands:

```bash
conda create --name Pose3D python=3.8
conda activate Pose3D
pip install -r requirements.txt
```
The requirements.txt file lists all Python libraries required to run this project.
#Dataset
Our code is designed to be compatible with the dataset setup introduced by Martinez et al. and Pavllo et al..
Please follow the instructions provided by VideoPose3D to download and prepare the Human3.6M dataset. All data should be placed in the ./data directory at the root of this project.
The expected directory structure is as follows:


.
├── data/
│   └── ... (Human3.6M dataset files)
├── checkpoint/
├── common/
├── run.py
└── requirements.txt

#Training
You can train the model using the provided scripts. We recommend a two-stage training process, where the second stage involves fine-tuning with a lower learning rate.

```bash
python run.py -k cpn_ft_h36m_dbb -f 27 -lr 0.0001 -lrd 0.99
```

```bash
python run.py -k cpn_ft_h36m_dbb -f 27 -lr 0.00004 -lrd 0.99
```

Model checkpoints will be saved automatically in the checkpoint/ directory during training.

#Testing
To evaluate the performance of a trained model on the Human3.6M test set, use the following command. Make sure you have a trained model checkpoint (e.g., best_epoch.bin) in the checkpoint/ directory.


