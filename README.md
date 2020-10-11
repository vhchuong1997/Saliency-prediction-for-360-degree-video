# Saliency prediction for 360 degree video
### Introduction
This repo contains the codes that used in the Capstone project of student: **Vo Hoang Chuong** from **the University of Science and Technology - University of Danang**.

### Configuration
  - Python 3.6 is required.
  - Module Used in this project:
    + pytorch = 1.5.0
    + CUDA toolkit: 10.2
    + CuDnn: 7.6.5
    + Numpy: 1.17.4
    + torchvision: 0.6.0
    + visdom: 0.1.8.9
    
### How to use
Pending...

### Dataset
The Dataset we used is taken from paper *Saliency Detection in 360° Videos* by **Ziheng Zhang, Yanyu Xu**, Jingyi Yu and Shenghua Gao.
The original download link is in [[Baidu Pan]](https://pan.baidu.com/share/init?surl=akj0-8obIwC9oykTYSUm9Q) the passcode is *ry6u*.
However, since you need to install Baidu NetDisk to download the folder and Baidu requires premium account for full speed downloading, it takes a lot of time to download the Dataset.
I have reuploaded it in [[Onedrive]](https://thcsxuanduong-my.sharepoint.com/:f:/g/personal/hoangchuong_thcsxuanduong_onmicrosoft_com/Eu87CfMhvHlLrGXV9B8mTmwBpgxkz-qNTNMFZFTl7t5JmQ?e=lKZfuX), password: Saliency2018 for more convenient access.
There are 12 zip files, and train/test index. After downloading these zips, unzip them together. Then, put the unzipped folders and the file 'vinfo.pkl' into the same folder named '360_Saliency_dataset_2018ECCV'

### TO DO
  - [x] Reupload the Dataset to a more accessible platform
  - [x] Release training code
  - [ ] Release weight file and testing code
  
### License

This project is released under the MIT License (refer to the LICENSE file for details).
