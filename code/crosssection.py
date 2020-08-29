import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt

filename = r"C:\Users\user\Documents\ML\expirements_models_denoised\unet model - 40 epochs - NO IR\rosbag\res_denoised-0.png" #r"C:\Users\user\Documents\ML\images\denoised\res_denoised-0.png"
filename = r"C:\Users\user\Documents\ML\images\train\noisy\res-0.png"
#filename = r"C:\Users\user\Documents\ML\images\diff_compare\diff_denoised\res_denoised_diff-1.png"
if (len(sys.argv) > 1):
    filename = str(sys.argv[1])

x0 = 0
y0 = 0
x1 = 0
y1 = 0
xx = 0
yy = 0

i = cv.imread(filename, -1).astype(np.float)

m = np.min(i)
M = np.max(i)

orig = i.copy()


def click_and_crop(event, x, y, flags, param):
    global x0
    global y0
    global x1
    global y1
    global xx
    global yy
    global orig

    xx = x
    yy = y

    if event == cv.EVENT_LBUTTONDOWN:
        return
    elif event == cv.EVENT_LBUTTONUP:
        if (x0 == 0):
            x0 = x
            y0 = y
        else:
            if (x1 == 0):
                x1 = x
                y1 = y

                X = np.arange(0, 1, 0.01)
                Y = np.array([])
                for t in X:
                    a = x1 * t + x0 * (1 - t)
                    b = y1 * t + y0 * (1 - t)
                    Y = np.append(Y, [orig[int(b), int(a)]])
                plt.plot(X, Y)
                plt.show()
            else:
                x0 = x
                y0 = y
                x1 = 0
                y1 = 0
        return


cv.namedWindow("image")
cv.setMouseCallback("image", click_and_crop)

#plt.hist(i.ravel(), int(M - m), [int(m), int(M)]);
#plt.show()

i = np.divide(i, np.array([M - m], dtype=np.float)).astype(np.float)
i = (i - m).astype(np.float)

i8 = (i * 255.0).astype(np.uint8)

if i8.ndim == 3:
    i8 = cv.cvtColor(i8, cv.COLOR_BGRA2GRAY)

i8 = cv.equalizeHist(i8)

colorized = cv.applyColorMap(i8, cv.COLORMAP_JET)

colorized[i8 == int(m)] = 0

font = cv.FONT_HERSHEY_SIMPLEX
colorized = cv.putText(colorized, str(m) + " .. " + str(M), (20, 50), font, 1, (255, 255, 255), 2, cv.LINE_AA)

while True:
    im = colorized.copy()

    if (x0 > 0 and x1 == 0):
        im = cv.line(im, (x0, y0), (xx, yy), (255, 255, 255), 2)
    if (x0 > 0 and x1 > 0):
        im = cv.line(im, (x0, y0), (x1, y1), (255, 255, 255), 2)

    # display the image and wait for a keypress
    cv.imshow("image", im)
    key = cv.waitKey(1) & 0xFF
    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break

cv.destroyAllWindows()