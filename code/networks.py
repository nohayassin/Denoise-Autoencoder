from unet_network import Unet
from basic_conv_network import Basic
from ccgan_network import CCGAN
from convautoencoder import ConvAutoencoder


class Network:
    def __init__(self, train_config):
        self.train_config = train_config

    def get(self):
        if self.train_config.MODEL == self.train_config.BASIC:
            return Basic(self.train_config)
        if self.train_config.MODEL == self.train_config.UNET:
            return Unet(self.train_config.train_cropped_images_path, self.train_config.channels, input_size=(self.train_config.img_width, self.train_config.img_height, self.train_config.channels), unet_epochs=self.train_config.epochs, pretrained_weights=None)
        if self.train_config.MODEL == self.train_config.CCGAN:
            return CCGAN()
        if self.train_config.MODEL == self.train_config.CONV:
            return ConvAutoencoder.build(self.train_config.img_width, self.train_config.img_height, self.train_config.channels, filters=(16, 32, 64))


    def ccgan_model(self):
        pass