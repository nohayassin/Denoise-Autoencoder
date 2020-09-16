# Denoise-Autoencoder
Denoise Autoencoder For 3D Cameras

Generating accurate depth frames from infra red images becomes a challenge due to environmental noises created from waves (e.g electromagnetic, sound, heat, etc).
Knowing the exact depth of object in 3D cameras is critical to avoid drones and robots crashes.
In this project I trained Conventional, Unet and GAN autoencoder networks on depth and infra red frames to remove unwanted noises and predict the exact placement of objects in each frame.


## Unet Network Architecture

![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/images/u-net-architecture.png)




## 3D Demo of Unet Network

### Ground Truth Depth Frames

![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/GAN/3D%20pure-%20100%20epochs%20-%20strides%20200%20-%20erosion%202%20-%20Binary%20-%20NO%20IR.gif)

### Denoised Depth Frames
The goal is to train the neural network so that denoised frames look as ground truth frames as possible.
![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/GAN/3D%20denoised-%20100%20epochs%20-%20strides%20200%20-%20erosion%202%20-%20Binary%20-%20NO%20IR.gif)


## Unet Network Training Evolution 
Notice the improvement of Unet network learning process by:
1. Increasing data set size 
2. Integrating infra red image as a second channel to depth image

### Ground Truth Depth Frame
The target frame Unet network should reach
![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/images/GT-3D.PNG)

### Noisy Depth Frame
Depth frame as captured by DS5 camera (input to Unet network)
![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/images/noisy-3D.PNG)

### Denoised Depth Frame
Output of Unet network after training on ground truth and noisy depth frame above
![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/images/Unet%20Evolution.gif)


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Keras and Tensorflow.

```bash
pip install tensorflow-gpu
pip install keras
```
#### Versions
```bash
Tensorflow - 2.3.0 
Tensorflow-gpu - 2.2.0
Keras - 2.4.3
protobuf - 3.12.2
```

## Usage

Run python autoencoder.py with relevant arguments as clarified below.
Prepare 2 directories of images : one for training and the other contains images for testing. Pass the directories to autoencoder.py using relevant arguments.
Set flag -c to crop only if you have new training images.

```bash
usage: Denoise Autoencoder Parser [-h] [-n] [-t] [-s] [-c]
                                  [--train_path [TRAIN_PATH]]
                                  [--test_path [TEST_PATH]]
                                  [--keras_model_path [KERAS_MODEL_PATH]]

optional arguments:
  -h, --help            show this help message and exit
  -n, --train           train flag
  -t, --test            test flag
  -s, --statistics      statistics flag
  -c, --crop            crop training images
  --train_path [TRAIN_PATH] directory for training images
  --test_path [TEST_PATH]  directory for images to test
  --keras_model_path [KERAS_MODEL_PATH]  Keras model path
```                    

## Output File Tree of Autoencoder Application 
This file tree will be created automatically by the application
```bash
.
├───images
│   ├───denoised
│   ├───test
│   ├───test_cropped
│   ├───train
│   ├───train_cropped
│   └───train_masked
├───logs
└───models
    └───DEPTH_20200910-203131.model
        ├───assets
        └───variables
```



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
