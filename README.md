# Denoise-Autoencoder
Denoise Autoencoder For 3D Cameras

Generating accurate depth frames from infra red images becomes a challenge due to the environmental noises created from waves (e.g electromagnetic, sound, heat, etc).
Knowing the exact depth of object in 3D cameras is critical to avoid drones and robots crashes.
In this project I trained Conventional, Unet and GAN autoencoder networks on depth and infra red frames to remove unwanted noises and predict the exact placement of objects in each frame.

## 3D Demo of Unet Network

### Ground Truth Frames

![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/GAN/3D%20pure-%20100%20epochs%20-%20strides%20200%20-%20erosion%202%20-%20Binary%20-%20NO%20IR.gif)

### Denoised Frames

![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/GAN/3D%20denoised-%20100%20epochs%20-%20strides%20200%20-%20erosion%202%20-%20Binary%20-%20NO%20IR.gif)


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Keras and Tensorflow.

```bash
pip install tensorflow-gpu
pip install keras
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Files Tree
```bash
.
├── images
│   ├── cropped_images
│   │   ├── ir
│   │   │   └── images [500 entries, not opening dir]
│   │   ├── noisy
│   │   │   └── images [500 entries, not opening dir]
│   │   └── pure
│   │    └── images [500 entries, not opening dir]
│   ├── cropped_tests
│   │   ├── ir
│   │   │   └── images [500 entries, not opening dir]
│   │   └── depth
│   │    └── images [500 entries, not opening dir]
│   ├── denoised
│   │   └── images [500 entries, not opening dir]
│   ├── diff_compare
│   │   ├── colored_diff_denoised
│   │   │   └── images [30 entries, not opening dir]
│   │   ├── colored_diff_tested
│   │   │   └── images [30 entries, not opening dir]
│   │   ├── diff_denoised
│   │   │   └── images [30 entries, not opening dir]
│   │   ├── diff_tested
│   │   │   └── images [30 entries, not opening dir]
│   │   └── logs
│   │    └── txt files 
│   ├── normalized
│   │   └── images [500 entries, not opening dir]
│   ├── real_scenes_raw
│   │   └── images [*.raw files, not opening dir]
│   ├── real_scenes_png
│   │   └── images [*.png files, not opening dir]
│   ├── tests
│   │   ├── ir
│   │   │   └── images [500 entries, not opening dir]
│   │   ├── depth
│   │   │   └── images [500 entries, not opening dir]
│   │   ├── masked_depth
│   │   │   └── images [500 entries, not opening dir]
│   │   └── pure
│   │    └── images [500 entries, not opening dir]
│   └── train
│   │   ├── ir
│   │   │   └── images [500 entries, not opening dir]
│   │   ├── pure
│   │   │   └── images [500 entries, not opening dir]
│   │   ├── noisy
│   │   │   └── images [500 entries, not opening dir]
│   │   └── masked_pure
│   │    └── images [500 entries, not opening dir]
```
