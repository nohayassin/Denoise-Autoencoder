
class NetworkConfig:
    def __init__(self, train=0, test=0, statistics=0, network_type=0):

        self.images_path = r"C:\Users\user\Documents\ML\images"
        self.models_path = r"C:\Users\user\Documents\ML\models"
        self.logs_path = r"C:\Users\user\Documents\ML\logs"

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
        self.CROP_DATA = (0 or self.MASK_PURE_DATA) and self.TRAIN_DATA
        self.TEST_REAL_DATA = 0 and self.TEST_DATA
        self.LOAD_TRAINED_MODEL = 0 and self.TRAIN_DATA
        self.OUTPUT_EQUALS_INPUT = 0 and self.TRAIN_DATA
        self.REMOVE_IR = 0 and self.TRAIN_DATA  # not relevant for now
        self.IMAGE_EXTENSION = '.png'  # '.tif'#

        # other configuration
        self.channels = 1
        self.img_width, self.img_height = 128, 128  # 64, 64 ##256, 256 #512, 512
        self.EROSION_ITERATIONS = 1

class TrainConfig(NetworkConfig):
    def __init__(self, train=0, test=0, statistics=0, network_type=0):
        NetworkConfig.__init__(self, train, test, statistics, network_type)
        self.origin_files_index_size_path_pure = {}
        self.origin_files_index_size_path_noisy = {}
        self.origin_files_index_size_path_ir = {}

        self.load_model_name = self.models_path + r"\DEPTH_20200810-153235.model_new_new"

        self.imgdir_pure = self.images_path + r"/train/pure"  # "./images/train/masked_pure"
        self.imgdir_noisy = self.images_path + "/train/noisy"
        self.imgdir_ir = self.images_path + "/train/ir"
        self.savedir_pure = self.images_path + "./cropped_images/pure"
        self.savedir_noisy = self.images_path + "./cropped_images/noisy"
        self.cropped_train_images_ir = self.images_path + "./cropped_images/ir"
        self.masked_pure = self.images_path + r"./train/masked_pure"
        self.masked_noisy = self.images_path + r"./train/masked_noisy"
        self.cropped_train_images_pure = self.images_path + "./cropped_images/pure"
        self.cropped_train_images_noisy = self.images_path + "./cropped_images/noisy"



    def get_mask_pure_inputs(self):
        return self.imgdir_pure, self.imgdir_noisy, self.masked_pure

    def get_train_data_inputs(self, image_set="pure"):
        if image_set == "ir":
            return self.imgdir_ir, self.cropped_train_images_ir, self.img_width, self.img_height, self.origin_files_index_size_path_ir
        if image_set == "noisy":
            return self.imgdir_noisy, self.savedir_noisy, self.img_width, self.img_height, self.origin_files_index_size_path_noisy
        if image_set == "pure":
            return self.imgdir_pure, self.savedir_pure, self.img_width, self.img_height, self.origin_files_index_size_path_pure

    def get_image_to_array_train_input(self, type="pure"):
        if type == "pure":
            return self.img_width, self.img_height, self.cropped_train_images_pure, self.cropped_train_images_ir, self.channels
        if type == "noisy":
            return self.img_width, self.img_height, self.cropped_train_images_noisy, self.cropped_train_images_ir, self.channels

class TestConfig:
    def __init__(self):
        self.origin_files_index_size_path_test = {}
        self.test_img_width, self.test_img_height = 480, 480

        self.test_model_name = r"C:\Users\user\Documents\ML\models\DEPTH_20200828-124309.model"

        self.imgdir = r"./images/tests/depth"
        self.realDataDir = r".\images\real_scenes_png"
        self.ir_imgdir = r"./images/tests/ir"
        self.denoised_dir = r"./images/denoised"
        self.cropped_images = r"./images/cropped_tests/depth"
        self.ir_cropped_images = r"./images/cropped_tests/ir"
        self.denoised_dir = r"./images/denoised"
        self.normalized_dir = r"./images/normalized"

        self.pngdir = r".\images\real_scenes_raw"
        self.pngoutdir = r".\images\real_scenes_png"

    def get_test_data_inputs(self, image_set="test"):
        if image_set == "ir":
            return self.ir_cropped_images, self.test_img_width, self.test_img_height, self.origin_files_index_size_path_test
        if image_set == "test":
            return self.cropped_images, self.test_img_width, self.test_img_height, self.origin_files_index_size_path_test

    def get_image_to_array_test_input(self):
        return self.test_img_width, self.test_img_height, self.channels

class StatisticsConfig:
    def __init__(self):
        self.diff_denoised_path = r".\images\diff_compare\diff_denoised"
        self.diff_tested_path = r".\images\diff_compare\diff_tested"
        self.colored_diff_denoised_path = r".\images\diff_compare\colored_diff_denoised"
        self.colored_diff_tested_path = r".\images\diff_compare\colored_diff_tested"
        self.diff_log_path = r".\images\diff_compare\logs"
        self.denoised_path = r".\images\denoised"
        self.tested_path = r".\images\tests\depth"
        #pure_path = r".\images\tests\pure"
        self.pure_path = r"c:\users\user\documents\ml\images\train\masked_pure"

