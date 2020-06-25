"""
ssim, psnr metric to evaluate generated images
"""

import numpy as np
import math
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr 
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sns

def mse(a, b):
    err = np.sum((a.astype('float')-b.astype('float'))**2)
    err /= float(a.shape[0]*a.shape[1])
    return err

def compare_image(A, B, title):
    m = mse(A, B)
    s = ssim(A, B, multichannel=True)
    p = psnr(A, B)

    # fig = plt.figure(title)
    # plt.suptitle("MSE : %.4f, SSIM : %.4f, PSNR : %.4f"%(m,s,p))

    # ax = fig.add_subplot(1,2,1)
    # plt.imshow(A,cmap = plt.cm.gray)
    # plt.axis("off")
    # ax = fig.add_subplot(1,2,2)
    # plt.imshow(B,cmap = plt.cm.gray)
    # plt.axis("off")

    plt.show()

def result(num):
    label = img.imread("result/{}_label.png".format(num))
    out = img.imread("result/{}_output.png".format(num))

    compare_image(out, label,'{}'.format(num))

# result(100)
# result(1000)
# result(3000)
# result(6000)
# result(8000)
# result(10000)

mm = []
ss = []
pp = []
for i in range(100,10001,100):
    B = img.imread("result/{}_label.png".format(i))
    A = img.imread("result/{}_output.png".format(i))

    m = mse(A, B)
    s = ssim(A, B, multichannel=True)
    p = psnr(A, B)

    mm.append(m)
    ss.append(s)
    pp.append(p)

# print(len(mm),mm)

plt.plot(np.linspace(100,10000,100),mm)
plt.title('MSE')
plt.xlabel('Iter')
plt.ylabel('MSE')
plt.show()

plt.plot(np.linspace(100,10000,100),ss)
plt.title('SSIM')
plt.xlabel('Iter')
plt.ylabel('SSIM')
plt.show()

plt.plot(np.linspace(100,10000,100),pp)
plt.title('PSNR')
plt.xlabel('Iter')
plt.ylabel('PSNR')
plt.show()