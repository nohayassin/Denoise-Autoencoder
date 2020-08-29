from unet_network import Unet
from basic_conv_network import Basic
from ccgan_network import CCGAN
from convautoencoder import ConvAutoencoder
import configurations


class Network:
    def __init__(self, model):
        self.__model = model

    def get(self):
        if self.__model == configurations.BASIC:
            return Basic()
        if self.__model == configurations.UNET:
            return Unet(pretrained_weights=None, input_size=(configurations.img_width, configurations.img_height, configurations.channels))
        if self.__model == configurations.CCGAN:
            return CCGAN()
        if self.__model == configurations.CONV:
            return ConvAutoencoder.build(configurations.img_width, configurations.img_height, configurations.channels, filters=(16, 32, 64))
        return self.basic_model()


    def ccgan_model(self):
        pass