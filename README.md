# Denoise-Autoencoder
Denoise Autoencoder For 3D Cameras

Generating accurate depth frames from infra red images becomes a challenge due to the environmental noises created from waves (e.g electromagnetic, sound, heat, etc).
Knowing the exact depth of object in 3D cameras is critical to avoid drones and robots crashes.
In this project I trained Conventional, Unet and GAN autoencoder networks on depth and infra red frames to remove unwanted noises and predict the exact placement of objects in each frame.

## 3D Ground Truth Frames

![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/3D%20pure%20-%20100%20epochs%20-%20strides%20200%20-%20erosion%202%20-%20Binary%20-%20NO%20IR.gif)

## 3D Denoised Frames

![foxdemo](https://github.com/nohayassin/RealSense-ML/blob/master/3D%20denoised-%20100%20epochs%20-%20strides%20200%20-%20erosion%202%20-%20Binary%20-%20NO%20IR.gif)
