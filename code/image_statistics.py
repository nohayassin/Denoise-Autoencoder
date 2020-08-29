import os, sys
import math
import time
import numpy as np
from itertools import chain
import cv2



class Statistics:
    def __init__(self, statistics_config, image_config):
        self.statistics_config = statistics_config
        self.image_config = image_config

    def find_files(self, dir_path, lst):
        for fname in os.listdir(dir_path):
            path = os.path.join(dir_path, fname)
            if os.path.isdir(path):
                continue
            lst.append(path)

    def rmsdiff(self, im1_path, im2_path):

        im1 = cv2.imread(im1_path, cv2.IMREAD_UNCHANGED)
        im2 = cv2.imread(im2_path, cv2.IMREAD_UNCHANGED)

        im1, im2 = im1.astype('float32'), im2.astype('float32')

        ignored_pixels = len(im1[np.where(im1 == 0)])
        ignored_pixels += len(im2[np.where(im2 == 0)])

        # pixels with value 0 that appear in both images should not be ignored
        lst1 = np.where(im1 == 0)
        lst2 = np.where(im2 == 0)
        set1, set2 = set(), set()
        for i in range(len(lst1[0])):
            set1.add((lst1[0][i], lst1[1][i]))
        for i in range(len(lst2[0])):
            set2.add((lst2[0][i], lst2[1][i]))
        set3 =  set1.intersection(set2)
        ignored_pixels -= len(set3)

        diff = im1 - im2

        # turn off dead pixels
        im1[np.where(im2 == 0 )] = float('inf')
        im2[np.where(im1 == 0 )] = float('inf')

        diff2 = im1 - im2

        # remove the item for all its occurrences
        flatten_list = list(chain.from_iterable(diff2))
        len1 = len(flatten_list)
        flatten_list = np.array(flatten_list)
        flatten_list = np.delete(flatten_list, np.argwhere(flatten_list == np.inf))
        flatten_list = np.delete(flatten_list, np.argwhere(flatten_list == -np.inf))
        dropped_items = len1 - len(flatten_list) # should be equal to ignored_pixels

        median = np.median(flatten_list)

        diff2[np.where(diff2 == float('inf'))] = 0
        diff2[np.where(diff2 == float('-inf'))] = 0
        avg = np.sum(diff2) / (float(im1.size) - ignored_pixels)

        #normalize
        m = np.min(diff2)
        M = np.max(diff2)
        div = max(abs(max(abs(m),abs(M))), 1)
        diff2 = diff2 / div

        sq = np.square(diff2)
        sum_of_squares = sum(sum(sq))
        rms = 100* math.sqrt(sum_of_squares / (float(im1.size)-ignored_pixels))

        return avg, median, rms, diff

    def colorize_helper(self, i, positive):
        m = np.min(i)
        M = np.max(i)

        i = (i - m).astype(np.float)
        i = np.divide(i, np.array([M - m], dtype=np.float)).astype(np.float)


        i8 = (i * 128.0).astype(np.uint8)
        if positive:
            i8 += 128
            i8[np.where(i8 == 0)] = 255

        return i8

        i8 = cv2.equalizeHist(i8)
        colorized = cv2.applyColorMap(i8, cv2.COLORMAP_JET)

        if not positive:
            colorized[i8 == int(m)] = 0
            return colorized

        colorized[i8 == int(m)] = 128.0
        return colorized

    def colorize_diff(self, diff_image, path, name):

        outfileN = path + '\\' + name.split('-')[0] + '' + '_diff_colorN-' + name.split('-')[1]
        outfileP = path + '\\' + name.split('-')[0] + '' + '_diff_colorP-' + name.split('-')[1]
        outfile = path + '\\' + name.split('-')[0] + '' + '_diff_color-' + name.split('-')[1]

        diff_negative = diff_image.copy()
        diff_positive = diff_image.copy()

        # replace all positive value with 0
        diff_negative = np.where(diff_negative >= 0, 0, diff_negative)
        # replace all negative value with 0
        diff_positive = np.where(diff_positive <= 0, 0, diff_positive)

        i8_negative = self.colorize_helper(diff_negative, False)
        i8_positive = self.colorize_helper(diff_positive, True)

        mN = np.min(i8_negative)
        mP = np.min(i8_positive)

        # ================================================================
        i8_negative_save , i8_positive_save = i8_negative, i8_positive

        i8_negative = i8_negative.astype(np.uint8)
        i8_negative = cv2.equalizeHist(i8_negative)
        i8_negativeC = cv2.applyColorMap(i8_negative, cv2.COLORMAP_JET)
        i8_negativeC[i8_negative == int(mN)] = 0

        i8_positive = i8_positive.astype(np.uint8)
        i8_positive = cv2.equalizeHist(i8_positive)
        i8_positiveC = cv2.applyColorMap(i8_positive, cv2.COLORMAP_JET)
        i8_positiveC[i8_negative == int(mP)] = 128

        i8_negativeC = np.array(i8_negativeC, dtype="float32")
        i8_positiveC = np.array(i8_positiveC, dtype="float32")
        cv2.imwrite(outfileN, i8_negativeC)
        cv2.imwrite(outfileP, i8_positiveC)

        i8_negative, i8_positive = i8_negative_save, i8_positive_save
        # ================================================================

        i8_positive[np.where(i8_positive == 128)] = 0
        i8_negative[np.where(i8_negative == 128)] = 0
        i8_negative[(i8_positive -i8_negative )== 0] = 128

        i8 = i8_negative + i8_positive
        i8 = i8.astype(np.uint8)
        i8 = cv2.equalizeHist(i8)
        colorized = cv2.applyColorMap(i8, cv2.COLORMAP_JET)
        m = np.min(i8)
        colorized[i8 == int(m)] = 0

        colorized = np.array(colorized, dtype="float32")
        cv2.imwrite(outfile, colorized)


    def calc_diff(self):
        # clean directories before processing
        self.image_config.clean_directory(self.statistics_config.diff_denoised_path)
        self.image_config.clean_directory(self.statistics_config.diff_tested_path)
        self.image_config.clean_directory(self.statistics_config.colored_diff_denoised_path)
        self.image_config.clean_directory(self.statistics_config.colored_diff_tested_path)

        denoised_filelist = []
        tested_filelist = []
        pure_filelist = []

        self.find_files(self.statistics_config.denoised_path, denoised_filelist)
        self.find_files(self.statistics_config.tested_path, tested_filelist)
        self.find_files(self.statistics_config.pure_path, pure_filelist)

        denoised_filelist.sort()
        tested_filelist.sort()
        pure_filelist.sort()

        old_stdout = sys.stdout

        print('Running Statistics ..')
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_name = 'DIFF_' + timestr + '.model'
        name = self.statistics_config.diff_log_path + '\\' + log_name + '.log'
        log_file = open(name, "w")
        sys.stdout = log_file

        for i in range(len(tested_filelist)):
            denoised_diff_name = denoised_filelist[i].split('\\')[-1]
            tested_diff_name = tested_filelist[i].split('\\')[-1]
            denoised_diff_outfile = self.statistics_config.diff_denoised_path + '\\' + denoised_diff_name.split('-')[0] + '' + '_diff-' + denoised_diff_name.split('-')[1]
            tested_diff_outfile = self.statistics_config.diff_tested_path + '\\' + tested_diff_name.split('-')[0] + '' + '_diff-' + tested_diff_name.split('-')[1]

            avg1, median1, denoised_rms, diff1 = self.rmsdiff(denoised_filelist[i], pure_filelist[i])
            avg2, median2, tested_rms, diff2 = self.rmsdiff(tested_filelist[i], pure_filelist[i])

            self.colorize_diff(diff1, self.statistics_config.colored_diff_denoised_path, denoised_diff_name)
            #colorize_diff(diff2, paths.colored_diff_tested_path, tested_diff_name)

            diff1 = np.array(diff1, dtype="float32")
            diff2 = np.array(diff2, dtype="float32")

            cv2.imwrite(denoised_diff_outfile, diff1)
            cv2.imwrite(tested_diff_outfile, diff2)
            print('---- image: ', i, '----')
            print('denoised_rms = ', denoised_rms, 'average = ', avg1, 'median = ', median1)
            print('tested_rms = ', tested_rms, 'average = ', avg2, 'median = ', median2)

        sys.stdout = old_stdout
        log_file.close()
