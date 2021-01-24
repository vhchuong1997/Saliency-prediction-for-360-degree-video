# Saliency prediction for 360-degree video

![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?) ![code\_score](https://www.code-inspector.com/project/16760/score/svg) ![code\_status](https://www.code-inspector.com/project/16760/status/svg) ![Read the Docs](https://readthedocs.org/projects/yt2mp3/badge/?version=latest) [![dependencies Status](https://david-dm.org/request/request/status.svg)](https://david-dm.org/request/request)

## Introduction

This repo contains the codes that are used in the paper **"Saliency prediction for 360-degree video"** published in [\[GTSD2020\]](https://ieeexplore.ieee.org/document/9303135) conference. This work is also a part of the Capstone project of student: **Vo Hoang Chuong** from **the University of Science and Technology - The University of Danang**.

## Saliency prediction for 360-degree video

In this work, we proposed a novel spherical convolutional network concentrating on 360° video saliency prediction in which the kernel is defined as a spherical cap. In the process of convolution, instead of using neighboring pixels with regular relationship in the equirectangular projection coordinate, the convolutional patches will be changed to preserve the spherical perspective of the spherical signal. The proposed spherical convolutional network is evaluated by Pearson correlation coefficient \(CC\) and Kullback–Leibler divergence \(KLD\). Our experiments show the efficiency of our pro-posed spherical convolution method’s application in 360° video saliency detection utilizing spherical U-net model.

In this case, we choose the kernel used in convolutional layer is 3x3 kernel. In the convolution process, the kernel will be elementwised multiplied with each 3x3 patch of the image. Our intention is to change the values in each 3x3 patch so that the perspective of the spherical image can be preserved during the convolution.

![spherical\_conv](.gitbook/assets/workflow.png)
|:---:|
| _Spherical convolution framework_ |


**Preprocessing function:** For a conventional picture with resolution \(240x480\), we firstly take out 240x480 = 1,152,000 pixels and put them as the center pixel of 1,152,000 \(3x3\) patches. After that, we replace all the surrounding pixels of these patches with the corresponding surrounding pixels according to their coordinate in spherical domain. Next, we put these patches together in order to form a new image with resolution: 720x1440.


![Preprocessing\_function](.gitbook/assets/Preprocessing%20function.png)
|:---:|
| _Framework of the preprocessing function_ |



For an example 3x3 patch from the image The central coordinate is: \(0,0\). By searching the \(0,0\) coordinate in the dictionary file, we acknowledge that the corresponding surrounding coordinates are: 
![coor](.gitbook/assets/coor.png) 
The patch’s surrounding pixels will be replaced by the pixels with the other pixels on the image having the surrounding coordinates in the matrix obtained above.

![Replacing](.gitbook/assets/process%203x3%20patch.png)
|:---:|
| _The process of replacing the surrounding pixels in the 3x3 patch_ |



After changing each of the patches in the image, we can put them together to create a new image with each size is 3 times larger than the original one. The new image will become the input of the next convolutional layer.

The model utilized for saliency map inference in this work is Spherical U-net model.

![U-net](.gitbook/assets/Unet%20-%20new.png)
|:---:|
| _Spherical U-net model_ |



## Configuration

* Python 3.6 is required.
* Module Used in this project:
  * pytorch: 1.5.0
  * CUDA toolkit: 10.2
  * CuDnn: 7.6.5
  * Numpy: 1.17.4
  * torchvision: 0.6.0
  * Pillow: 7.0.0
  * visdom: 0.1.8.9

## How to use

**Testing:** In order to reproduce the results of the project, you just need to simply run `Testing_model.py`. The Groundtruths, frames, saliency maps will be saved respectively in an auto-generated result folder. We used the visdom lib to visualize the output of the model. Before running the code, please initiate visdom by typing `visdom` in `cmd`\(Windows\) or `Terminal prompt`\(Ubuntu\). Then, go to `http://localhost:8097/`, choose env `test1` to watch the visualization. For in depth usage, please refer to the official git of [visdom](https://github.com/facebookresearch/visdom) to know how to use it. \(The code will still run normally if you disable visdom related lines\).

**Training:** In order to retrain the model from scratch, you just need to simply run `train.py`. Feel free to change the training parameters if you would like to improve the results. We also use visdom to visualize the process of training. Please follow te steps as mentioned above and choose env `final` to watch the visualization.

## Dataset

The Dataset we used is downloaded from paper _Saliency Detection in 360° Videos_ by **Ziheng Zhang, Yanyu Xu**, Jingyi Yu and Shenghua Gao. The original download link is in [\[Baidu Pan\]](https://pan.baidu.com/share/init?surl=akj0-8obIwC9oykTYSUm9Q) the passcode is _ry6u_. However, since you need to install Baidu NetDisk to download the folder and Baidu requires premium account for full speed downloading, it takes a lot of time to download the Dataset. I have reuploaded it in [\[Onedrive\]](http://bit.ly/3c618Pf), password: **Saliency2018** for more convenient access. There are 12 zip files, and train/test index. After downloading these zips, unzip them together. Then, put the unzipped folders and the file 'vinfo.pkl' into the same folder named '360\_Saliency\_dataset\_2018ECCV'

## Result

![U-net](.gitbook/assets/Result.png)
|:---:|
| _Results_ |



## Inspiration

This code is inspired by the work of [Ziheng Zhang, Yanyu Xu](https://github.com/xuyanyu-shh/Saliency-detection-in-360-video).

## Acknowledgement

I would like to express my gratitude towards [**Assoc. Prof. Jui-Chiu Chiang**](https://ieeexplore.ieee.org/author/37416120400) for her guildance and supersivison. I also want to give special thanks to [**Mr. Ziheng Zhang**](https://scholar.google.com/citations?user=QQ2-OOUAAAAJ&hl=en) for helping me in understanding the workflow of his spherical CNN. Extremely thanks to the [**Pytorch**](https://discuss.pytorch.org/) community for the precious information!

## TO DO

* [x] Reupload the Dataset to a more accessible platform
* [x] Release supporting files and functions
* [x] Release training code
* [x] Release weight file and testing code
* [ ] Release the code for generating the coordinates of the surrounding pixels corresponding to each center pixel in panorama format
* [ ] Implement generic custom spherical convolution for any input size 

## License

This work is released under the MIT License.

## Citation

If you find this work helpful for your research, please consider citing:
```
@INPROCEEDINGS{9303135,
  author={C. H. {Vo} and J. -C. {Chiang} and D. H. {Le} and T. T. A. {Nguyen} and T. V. {Pham}},
  booktitle={2020 5th International Conference on Green Technology and Sustainable Development (GTSD)}, 
  title={Saliency Prediction for 360-degree Video}, 
  year={2020},
  volume={},
  number={},
  pages={442-448},
  doi={10.1109/GTSD50082.2020.9303135}}
```
