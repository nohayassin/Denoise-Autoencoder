#### Training Dataset
The dataset is located here: https://drive.google.com/file/d/1cXJRD4GjsGnfXtjFzmtLdMTgFXUujrRw/view?usp=drivesdk
It containes 3 types of 484x480 png images : 
###### 1. Ground Truth Images
- clean depth images that Neural Network should learn to predict. 
- 1-channel image of 16 bits depth
- name : gt-*.png
###### 2. Depth Images : 
- noisy depth images as captured by ds5.
- 1-channel image of 16 bits depth
- name : res-*.png
###### 3. Infra Red (IR) Images 
- used to help Unet learning the exact depth of each object
- 3-channel image of 8 bits depth for each channel. 
- name : left-*.png
		
#### Data Augmentation
To help the neural network learning images features, the images should be cropped to optimal size.
Large images contain many features, it requires adding more layers to detect all the features, this will impact the learning process negatively.
On the other hand, very small images may not contain any explicit feature, because most likely each feature would be splitted to several number of images.
It is very essential to choose the cropped images size optimally, considering the original size the average size of features inside the image.
In the set of experiments we did, image size of 128x128 found to be optimal.


Each ground truth image has a corressponding depth and infra red (IR) image. Given that, the dataset is augmented as following:

###### 1. Cropping 

Each image in the dataset is padded to get a size of 896x512 then each of them is cropped to 128x128. In total, each image is cropped to 28 images of size 128x128.  
Each cropped image is saved with the original image name, adding to it information about the column and row the image was cropped from. It helps corresponding to each ground-truth cropped-image, the IR and depth image from the cropped set.

![foxdemo](images/cropping.PNG)

###### 2. Channeling
Before channeling, IR images are converted to 1-channel image of 16-bits depth.
IR (infra red) image is added as a second channel to both ground truth and depth image, to add more information about the depth of each object in the image.

Eventually, the data that is fed to Unet network contains:
- Noisy images: 
consistes of 2 channels: first channel is a depth image and second channel is the corressponding IR image
- Pure images: 
consistes of 2 channels: first channel is a ground truth image and second channel is the corressponding IR image. 
Each channel in both pure and noisy is a 16-bits depth.

![foxdemo](images/channeling.PNG)

#### Training Process
In order to start a training process, the following is required:
- Unet Network Implementation : choosing the sizes of convolutions, max pools, filters and strides, along downsampling and upsampling.
- Data Augmentation: preparing dataset that contains noisy and pure images as explained above.
- Old model (optional): there is an option of training the network starting from a previously-trained model. 
- Epochs : epoch is one cycle through the full training dataset (forth and back). The default value of epochs number is 100, it could be contolled by an argument passed to the application.

#### Files Tree 
the application will create automatically a file tree:
- images folder: contains original and cropped images for training and testing, also the predicted images
- logs folder: all tensorflow outputs throughout the training are stored in txt file that has same name as the created model. It contains also a separate log for testing statistics.
- models folder: each time a training process starts, it creates a new folder for a model inside models folder. If the traing starts with old model, 
				 it will create a folder with same name as old model adding to it the string "_new"
		
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
		
####  Testing Process
The tested images should be augmented as trained images, except the cropping size should be 480x480 (each image is cropped to 2 images), considering performance improvement.
For testing, there is no need to ground truth data, only depth and IR images are required.
The relevant folders in file tree are: 
- test: original images to test of sizes 848x480
- test_cropped: cropped testing images, size: 480x480
- denoised: the application stores predicted images in this folder.
