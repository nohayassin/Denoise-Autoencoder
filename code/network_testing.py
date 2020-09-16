import os, sys
import glob
import time
import numpy as np
from skimage import img_as_uint
import cv2
import keras

class NetworkTesting:
    def __init__(self, test_config, image_config):
        self.test_config = test_config
        self.image_config = image_config

    def test(self):
        old_stdout = sys.stdout
        model = keras.models.load_model(self.test_config.test_model_name)
        print('Testing model', str(os.path.basename(self.test_config.test_model_name).split('.')[0]), '..')
        name = self.test_config.logs_path + '/output_' + str(os.path.basename(os.path.normpath(self.test_config.test_model_name)).split('.')[0]) + '.log'
        log_file = open(name, "w")
        sys.stdout = log_file
        print('prediction time : ')

        if self.test_config.TEST_REAL_DATA:
            self.test_config.imgdir = self.test_config.realDataDir
        total_cropped_images = self.image_config.get_split_img(self.test_config.test_img_width, self.test_config.test_img_height, True)
        cropped_noisy_images = [f for f in glob.glob(self.test_config.test_cropped_images_path + "**/res*", recursive=True)]
        cropped_noisy_images.sort()
        for i, directory in enumerate(cropped_noisy_images):
            cropped_image_offsets = []
            samples = self.image_config.image_to_array_test(directory, cropped_image_offsets)
            width, height, origin_file_name = self.test_config.origin_files_index_size_path_test[i]
            cropped_w, cropped_h = self.test_config.test_img_width, self.test_config.test_img_height
            whole_image = np.zeros((height, width, self.test_config.channels), dtype="float32")

            t1 = time.perf_counter()
            for i in range(total_cropped_images[i]):
                # testing
                sample = samples[i:i+1]
                row, col = cropped_image_offsets[i]
                row, col = int(row), int(col)
                denoised_image = model.predict(sample)
                row_end = row + cropped_h
                col_end = col + cropped_w
                denoised_row = cropped_h
                denoised_col = cropped_w
                if row + cropped_h >= height:
                    row_end = height-1
                    denoised_row = abs(row-row_end)
                if col + cropped_w >= width:
                    col_end = width-1
                    denoised_col = abs(col - col_end)
                # combine tested images
                whole_image[row:row_end, col:col_end]=  denoised_image[:, 0:denoised_row,0:denoised_col, :]
            t2 = time.perf_counter()
            print('test: ', directory.split('/')[-1], ': ', t2 - t1, 'seconds')
            denoised_name = os.path.basename(directory.split('/')[-1])
            outfile = self.test_config.denoised_dir + '/' + denoised_name.split('-')[0] + '' + '_denoised-' + denoised_name.split('-')[1] + self.test_config.IMAGE_EXTENSION
            whole_image = img_as_uint(whole_image)
            cv2.imwrite(outfile, whole_image[:,:,0])
        sys.stdout = old_stdout
        log_file.close()