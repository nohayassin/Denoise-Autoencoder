import os, glob
import numpy as np
import cv2
import image_process

def raw_to_png(imgdir, outDir, width, height):
    image_process.clean_directory(outDir)
    filelist = [f for f in glob.glob(imgdir + "**/*.raw", recursive=True)]

    for f in filelist:
        name = os.path.basename(f)
        name = os.path.splitext(name)[0]
        outfile = outDir + '/' + name + paths.IMAGE_EXTENSION

        img = np.fromfile(f, dtype='int16', sep="")
        # Parse numbers as floats
        img = img / max(img)
        img = img * 56535
        #img = img.astype('float32')
        # Normalize data
        img = img.reshape([height, width])
        img = img.astype('uint16')
        cv2.imwrite(outfile, img)



