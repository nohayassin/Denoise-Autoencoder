import os

class NetworkConfig:
    def __init__(self, train=0, test=0, statistics=0, network_type=0):

        self.root = r"C:\Users\user\Documents\test_unet_flow"
        self.images_path = self.root + r"/images"
        self.models_path = self.root  + r"/models"
        self.logs_path = self.root  + r"/logs"
        self.paths = [self.images_path, self.models_path, self.logs_path]
        self.create_folders()

        # models
        self.BASIC = 0
        self.UNET = 1
        self.CCGAN = 2
        self.MODEL = network_type

        # Flags and Parameters
        self.TRAIN_DATA = train
        self.DIFF_DATA = statistics
        self.TEST_DATA = test and (1 - self.TRAIN_DATA) and (1 - self.DIFF_DATA)

        self.MASK_PURE_DATA = 0 and self.TRAIN_DATA
        self.REMOVE_BACKGROUND = 0 and self.MASK_PURE_DATA
        self.NORMALIZE = 0 and self.MASK_PURE_DATA
        self.CROP_DATA = (1 or self.MASK_PURE_DATA) and self.TRAIN_DATA
        self.TEST_REAL_DATA = 0 and self.TEST_DATA

        self.OUTPUT_EQUALS_INPUT = 0 and self.TRAIN_DATA
        self.REMOVE_IR = 0 and self.TRAIN_DATA  # not relevant for now
        self.IMAGE_EXTENSION = '.png'  # '.tif'#
        self.CONVERT_RAW_TO_PNG = 0

        # other configuration
        self.channels = 2
        self.img_width, self.img_height = 128, 128  # 64, 64 ##256, 256 #512, 512
        self.EROSION_ITERATIONS = 1

    def create_folders(self):
        for path in self.paths:
            if not os.path.exists(path):
                print("Creating  ", path)
                os.makedirs(path)


class TrainConfig(NetworkConfig):
    def __init__(self, network_config):
        NetworkConfig.__init__(self, network_config.TRAIN_DATA, network_config.TEST_DATA, network_config.DIFF_DATA, network_config.MODEL)

        self.load_model_name = self.models_path + r"\DEPTH_20200903-132536.model"
        self.LOAD_TRAINED_MODEL = 0 and self.TRAIN_DATA

        self.images_path = self.root + r"/images"
        self.models_path = self.root + r"/models"
        self.logs_path = self.root + r"/logs"
        self.train_images = self.images_path + r"/train"
        self.train_cropped_images_path = self.images_path + r"/train_cropped"
        self.masked_pure = self.images_path + r"/train_masked"
        self.paths = [self.root, self.images_path, self.models_path, self.logs_path, self.train_images, self.masked_pure, self.train_cropped_images_path]

        print("Creating folders for training process ..")
        self.create_folders()

    def get_mask_pure_inputs(self):
        return self.imgdir_pure, self.imgdir_noisy, self.masked_pure

class TestConfig(NetworkConfig):
    def __init__(self, network_config):
        NetworkConfig.__init__(self, network_config.TRAIN_DATA, network_config.TEST_DATA, network_config.DIFF_DATA,
                               network_config.MODEL)
        self.origin_files_index_size_path_test = {}
        self.test_img_width, self.test_img_height = 480, 480

        self.test_model_name = r"C:\Users\user\Documents\ML\models\DEPTH_20200903-132536.model_new"

        self.test_images = self.images_path + r"/test"
        self.test_cropped_images_path = self.images_path + r"/test_cropped"
        self.denoised_dir = self.images_path + r"/denoised"
        self.paths = [self.test_images, self.test_cropped_images_path, self.denoised_dir]
        self.pngdir = self.images_path + r"\real_data"

        print("Creating folders for testing process ..")
        self.create_folders()

    def get_image_to_array_test_input(self):
        return self.test_img_width, self.test_img_height, self.channels

class StatisticsConfig(TestConfig):
    def __init__(self, network_config):
        TestConfig.__init__(self, network_config)
        self.diff_denoised_path = self.images_path + r"\diff_compare\diff_denoised"
        self.diff_tested_path = self.images_path + r"\diff_compare\diff_tested"
        self.colored_diff_denoised_path = self.images_path + r"\diff_compare\colored_diff_denoised"
        self.colored_diff_tested_path = self.images_path + r"\diff_compare\colored_diff_tested"
        self.diff_log_path = self.images_path + r"\diff_compare\logs"
        self.denoised_path = self.images_path + r"\denoised"
        self.tested_path = self.images_path + r"\tests\depth"
        self.pure_path = self.images_path + r"\train\masked_pure"

