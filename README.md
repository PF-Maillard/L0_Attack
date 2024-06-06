# Minimal $L_0$ attack

## About


This repository contains the implementation code for the "Minimal Adversarial Attack on the $L_0$ Norm,". The technique focuses on generating adversarial image examples while altering the fewest possible pixels with a low computation time. It also features comparisons with JSMA and CW-$L_0$ attacks across the MNIST, CIFAR10, and GTSRB datasets

Note the utilization of the following repository: 
- torchattacks: https://github.com/Harry24k/adversarial-attacks-pytorch
- robustbench: https://robustbench.github.io/
- Carlini Git: https://github.com/carlini/nn_robust_attacks
- GTSRB model: https://www.kaggle.com/code/chibani410/gtsrb-99-test-accuracy-pytorch

## Procedure


You must manually download the GTSRB dataset and place it in the "data" folder, renaming it to "Signal". The dataset is available here: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign.

Included are three primary notebooks:
- BenchmarkCifar10.ipynb
- BenchmarkMNIST.ipynb
- BenchmarkSignal.ipynb

These notebooks are designed to automatically fetch the necessary libraries, allowing for use without any additional prerequisites.


## Licenses

This project is licensed under the MIT License for the majority of its content, with specific portions covered under separate licenses as detailed below.

The code located in [Sources/GTSRBModel.py] is licensed under the Apache License 2.0. A copy of this license can be found in the file [Sources/GTSRBModel-License.md] or at the official Apache 2.0 license website.

Code from [Sources/CW_L0.py] is licensed under the BSD 2-Clause License. A copy of this license is included in the file [Sources/CW_L0-License.md] or can be viewed at the Open Source Initiative website.

For all other parts of the project not explicitly mentioned above, the MIT License applies. The full text of the MIT License can be found in the LICENSE.md file at the root of this project.

