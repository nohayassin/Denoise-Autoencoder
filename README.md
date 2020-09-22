# Denoise-Autoencoder
Denoise Autoencoder For 3D Cameras

Generating accurate depth frames from infra red images becomes a challenge due to environmental noises created by waves (e.g electromagnetic, sound, heat, etc).
Knowing the exact depth of object in 3D cameras is critical to avoid drones and robots crashes.
In this project I'm training Conventional, Unet and GAN autoencoder networks on depth and infra red frames to remove unwanted noises and predict the exact placement of objects in each frame.


## Unet Network 

Unet is a deep learning Architecture used for image segmentation problems
It was particularly released as a solution for biomedical segmentation tasks
but it became used for any segmentation problem. Unlike classification problems that use hundred of layers, Unet represent a very compact architecture
with limited number of parameters.
self-driving cars for example use image segmentation for awareness about the environment.
In this project, image segmentation is needed to know the exact depth of each object. 
Unet network is the ideal architecture that serves our goal.

Unlike image classification that classify the entire image, image segmentation classify every single pixel
and to what class it belongs to.
It is not enough to see if some feature is present or not, we need to know the exact shape and all the entricate 
properities: how each object behaves, the shapes and forms it takes to exactly know which pixels belong to that object.

This kind of task requires the network to have considerably more in-depth understanding about the object.
Before Unet, segmentation tasks were approched using a modified convolution networks by replacing
fully connected with fully convolutional layers then they were just upsample it to the same
resolution of the image and try to use it as a segmentation mask, but because the image was compressed
so much and it gone through so much processing (e.g max pooling and convolutions) a lot of information have thrown away.

Just doing upsampling will not really give us the fine resolution that we want. It is very
coarse and not very accurate. so regular upsampling doesn't work very well.
What Unet does, the first (left) pathway of it looks very similar to classic convolutional network
instead of just directly upsampling, it has another pathway which it gradually builds upon the up sampling
procedure, so it serves as directly up sampling via learning the best way that this compressed image should
be up sampled and using convolution filters for that.
    
#### Architecture
Input image goes through some convolutions, then it is downsampled by using
max pooling then it goes through more convolutions, downsampled again, and so on until it reaches the deepest layer
after that it is upsampled by half so we get back the sizes (see image below), then we concatenate the features of each connection 
and these concatenated features go through some more convolutions, then upsampled then it is joined (concatenated) back 
with the parallel layer, but we lose information as we go down through max pooling (mostly because it reduces the dimention by half), and
also through convolution because convolution throw away information from raw input to repupose them into
meaningful features. That what happens also in classification networks, where a lot of information is 
thrown away by the last layer. But in segmentation we want those low-level features because those
are essential to deconstructing the image. 

In the left pathway of Unet, the number of filters (features) increase as we go down, it means that it becomes
very good at detecting more and more features, the first few layers of a convolution network capture a very small semantic information and lower level
features, as you go down these features become larger and larger, but when we throw away information the CNN
knows only approximate location about where those features are.
When we upsample we get the lost information back (by the concatination process)
so we can see last-layer features in the perspective of the layer above them.
    
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

To install all relevant python packages, run the following command by using the package manager [pip](https://pip.pypa.io/en/stable/) :
```bash
pip install -r "...\requirements.txt"
```
Find requirements.txt in installation folder.
If you have GPU, install Tensorflow with GPU:

```bash
pip install tensorflow-gpu
pip install keras
```
#### Versions
```bash
Tensorflow-gpu - 2.2.0
Keras - 2.4.3
```

## Usage

Run autoencoder.py with relevant arguments as clarified below.

Prepare 2 directories of images : one for training and the other contains images for testing.

Pass the directories to autoencoder.py using relevant arguments. If no directories are selected, the application will use previous images.

Set flag -c for images cropping only if you have new training images.

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
  --epochs [EPOCHS]     epochs number
  --train_path [TRAIN_PATH] training images directory
  --test_path [TEST_PATH]  testing images directory
  --keras_model_path [KERAS_MODEL_PATH]  Keras model path
```                    

## Usage Examples
#### Crop new images for training
Nueral Networks cannot train large-size images (e.g 848 x 480), we need to crop each depth and infra red image to optimal squared size so the network could learn all features in each image in optimal way. In Unet Network experiments, image of size 848x480 is cropped to images of size 128x128. 
Run with flag -c to crop training images. Use this flag only when you have new images to train. 
```bash
python autoencoder.py -c --train_path <training images folder>
```
#### Train a new model
To train a new model, prepare images to train then crop them as explained above. It is possible to run one command for cropping images and start a training process.
Use the flag -n for training :
```bash
python autoencoder.py -n -c --train_path <training images folder>
```
To train the network again, no need to provide a path for training images again, and of course no need to crop the images again, simply run:
```bash
python autoencoder.py -n 
```
#### Epochs
Epochs number is set to 100 by default, it could be controlled by the argument "epochs" :
```bash
python autoencoder.py -n -c --epochs <epochs number> --train_path <training images folder>
```

#### Test a model
To test an existing model, run this command:
```bash
python autoencoder.py -t --test_path <testing images folder> --keras_model_path <Keras model path>
```
After selecteing images to test and a valid Keras model, the application will put predicted images in the folder images\denoised (see file tree below). If the folder of testing images is empty, denoised folder will stay empty. If no Keras model is provided, the application will check models folder and select most recent model. If models folder is empty, an error will be thrown stating that no Keras model was found.
Testing images path should be given only when new testing images are added.

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
