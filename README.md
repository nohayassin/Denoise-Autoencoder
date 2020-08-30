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
├── data
│   ├── class_10_train
│   │   ├── n01882714
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n01882714_boxes.txt
│   │   ├── n02165456
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n02165456_boxes.txt
│   │   ├── n02509815
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n02509815_boxes.txt
│   │   ├── n03662601
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n03662601_boxes.txt
│   │   ├── n04146614
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n04146614_boxes.txt
│   │   ├── n04285008
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n04285008_boxes.txt
│   │   ├── n07720875
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n07720875_boxes.txt
│   │   ├── n07747607
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n07747607_boxes.txt
│   │   ├── n07873807
│   │   │   ├── images [500 entries exceeds filelimit, not opening dir]
│   │   │   └── n07873807_boxes.txt
│   │   └── n07920052
│   │       ├── images [500 entries exceeds filelimit, not opening dir]
│   │       └── n07920052_boxes.txt
│   ├── class_10_val
│   │   ├── test_images [250 entries exceeds filelimit, not opening dir]
│   │   └── val_images [250 entries exceeds filelimit, not opening dir]
│   ├── class_dict_10.json
│   └── val_class_dict_10.json
├── data.zip
├── environment.yaml
└── tiny-vgg.py
```
