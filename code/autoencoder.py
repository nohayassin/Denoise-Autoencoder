import os, sys, argparse
from image_process import SplitImage
from image_statistics import *
from network_testing import *
from network_training import NetworkTraining
from configurations import *

def autoencoder(network_type, train, test, statistics, crop, train_img_dir, test_img_dir, keras_model_path, epochs, load_trained_model, load_model_name):
    network_config = NetworkConfig(train=train, test=test, statistics=statistics, network_type=network_type, crop=crop, epochs=epochs)
    train_config = TrainConfig(network_config, train_img_dir=train_img_dir, load_trained_model=load_trained_model, load_model_name=load_model_name)
    image_processing = SplitImage(network_config, train_config, None)

    if network_config.MASK_PURE_DATA:
        image_processing.mask_pure_images(train_config.get_mask_pure_inputs())
        train_config.imgdir_pure = train_config.masked_pure

    if network_config.CROP_DATA:
        print('Cropping training data ..')
        image_processing.get_split_img(train_config.img_width, train_config.img_height)

    if network_config.CONVERT_RAW_TO_PNG:
        image_processing.raw_to_png(848, 480)

    if network_config.TRAIN_DATA:
        network_train = NetworkTraining(train_config, image_processing)
        network_train.train()

    if network_config.TEST_DATA:
        test_config = TestConfig(network_config, keras_model_path=keras_model_path, test_img_dir=test_img_dir)
        image_processing = SplitImage(network_config, None, test_config)
        network_test = NetworkTesting(test_config, image_processing)
        network_test.test()

    if network_config.DIFF_DATA:
        statistics_config = StatisticsConfig(network_config, test_config)
        statistics = Statistics(statistics_config, image_processing)
        statistics.calc_diff()


if __name__ == '__main__':
    BASIC = 0
    UNET = 1
    CCGAN = 2

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser('Denoise Autoencoder Parser')
    parser.add_argument("-n", "--train", action="store_true", help="train flag")
    parser.add_argument("-t", "--test", action="store_true", help="test flag")
    parser.add_argument("-s", "--statistics", action="store_true", help="statistics flag")
    parser.add_argument("-c", "--crop", action="store_true", help="crop training images")
    parser.add_argument("-l", "--load_model", action="store_true", help="load a model flag")

    # nargs='?' : means 0-or-1 arguments
    # const=1 : sets the default when there are 0 arguments
    # type=int : converts the argument to int
    parser.add_argument('--epochs', nargs='?', const=1, default=100, type=int, help="epochs number")
    parser.add_argument('--train_path', nargs='?', const=1, default="", type=str, help="directory for training images")
    parser.add_argument('--test_path', nargs='?', const=1, default="", type=str, help="directory for images to test")
    parser.add_argument('--keras_model_path', nargs='?', const=1, default="", type=str, help="Keras model path")
    parser.add_argument('--load_model_name', nargs='?', const=1, default="", type=str, help="directory of a trained model")
    args = parser.parse_args()

    print(args.train_path)
    paths = [args.train_path, args.test_path, args.keras_model_path]
    for path in paths:
        if path: print("Test ", path, " ..")
        if path and not os.path.isdir(path):
            sys.exit("Invalid directory!")
    autoencoder(UNET, args.train, args.test, args.statistics, args.crop, args.train_path, args.test_path, args.keras_model_path, args.epochs, args.load_model, args.load_model_name)
