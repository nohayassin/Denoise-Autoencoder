# Denoise-Autoencoder
Denoise Autoencoder For 3D Cameras

Generating accurate depth frames from infra red images becomes a challenge due to environmental noises created from waves (e.g electromagnetic, sound, heat, etc).
Knowing the exact depth of object in 3D cameras is critical to avoid drones and robots crashes.
In this project I trained Conventional, Unet and GAN autoencoder networks on depth and infra red frames to remove unwanted noises and predict the exact placement of objects in each frame.


## Unet Network Architecture

![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/images/u-net-architecture.png)




## 3D Demo of Unet Network

### Ground Truth Frames

![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/GAN/3D%20pure-%20100%20epochs%20-%20strides%20200%20-%20erosion%202%20-%20Binary%20-%20NO%20IR.gif)

### Denoised Frames
#### The goal is to train the neural network so that denoised frames look as ground truth frames as possible.
![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/GAN/3D%20denoised-%20100%20epochs%20-%20strides%20200%20-%20erosion%202%20-%20Binary%20-%20NO%20IR.gif)



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Keras and Tensorflow.

```bash
pip install tensorflow-gpu
pip install keras
```



## Files Tree of Outputs
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
