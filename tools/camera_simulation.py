import pyrealsense2 as rs
import numpy as np
import cv2

import os, sys, shutil
import keras
import time
import tensorflow as tf
from keras import backend as kb
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import glob
from PIL import Image
import numpy as np
import cv2
from skimage import img_as_uint

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30) # 1 = left
# Start streaming
pipeline.start(config)


def test(noisy_image, ir_image):
    channels = 2
    cropped_w, cropped_h = 480, 480
    test_model_name = r"C:\work\OLD ML\expirements_models_denoised\NEW DATA - NO MASK\4443 images\unet model - 100 epochs - strides 1500 - NO mask - Binary - WITH IR - NOT FINISHED\DEPTH_20200903-132536.model_new"
    model = keras.models.load_model(test_model_name)

    ir_image = np.array(ir_image).astype("uint16")
    cropped_ir , cropped_noisy = [], []
    width, height = 848, 480
    w, h = cropped_w, cropped_h
    for col_i in range(0, width, w):
        for row_i in range(0, height, h):
            #cropped_ir.append(ir_image.crop((col_i, row_i, col_i + w, row_i + h)))
            #cropped_noisy.append(noisy_image.crop((col_i, row_i, col_i + w, row_i + h)))
            cropped_ir.append(ir_image[row_i:row_i+h, col_i:col_i+w])
            cropped_noisy.append(noisy_image[row_i:row_i+h, col_i:col_i+w])
    fill = np.zeros((h, w - cropped_ir[-1].shape[1]), dtype="uint16")
    cropped_ir[-1] = np.hstack((cropped_ir[-1], fill))
    cropped_noisy[-1] = np.hstack((cropped_noisy[-1], fill))
    ########### IMAGE TO ARRAY  ##################
    cropped_image_offsets = [(0,0), (0,480)]
    for i in range(len(cropped_ir)):
        noisy_images_plt = cropped_noisy[i].reshape(1, cropped_w, cropped_h, 1)
        ir_images_plt = cropped_ir[i].reshape(1, cropped_w, cropped_h, 1)

        im_and_ir = np.stack((noisy_images_plt, ir_images_plt), axis=3)
        im_and_ir = im_and_ir.reshape(1, cropped_w, cropped_h, channels)

        img = np.array(im_and_ir)
        # Parse numbers as floats
        img = img.astype('float32')

        # Normalize data : remove average then devide by standard deviation
        img = (img - np.average(img)) / np.var(img)
        sample = img

        whole_image = np.zeros((height, width, channels), dtype="float32")

        t1 = time.perf_counter()
        for i in range(total_cropped_images[i]):
            # testing
            #sample = samples[i:i + 1]
            row, col = cropped_image_offsets[i]
            denoised_image = model.predict(sample)
            row_end = row + cropped_h
            col_end = col + cropped_w
            denoised_row = cropped_h
            denoised_col = cropped_w
            if row + cropped_h >= height:
                row_end = height - 1
                denoised_row = abs(row - row_end)
            if col + cropped_w >= width:
                col_end = width - 1
                denoised_col = abs(col - col_end)
            # combine tested images
            whole_image[row:row_end, col:col_end] = denoised_image[:, 0:denoised_row, 0:denoised_col, :]
        t2 = time.perf_counter()
        print('test: ', os.path.basename(directory.split('/')[-1]), ': ', t2 - t1, 'seconds')
        denoised_name = os.path.basename(directory.split('/')[-1])
        outfile = denoised_dir + '/' + denoised_name.split('-')[0] + '' + '_denoised-' + denoised_name.split('-')[
            1] + IMAGE_EXTENSION
        whole_image = img_as_uint(whole_image)

        cv2.imwrite(outfile, whole_image[:, :, 0])
    sys.stdout = old_stdout
    log_file.close()
    print("Testing process is done successfully !")

#=============================================================================================================
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        #color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()

        if not depth_frame or not ir_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        #color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())

        predicted_image = test(depth_image, ir_image)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((depth_colormap, predicted_image))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()