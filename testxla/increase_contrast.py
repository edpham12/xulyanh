import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

img = cv2.imread('./anhtest.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(1)
plt.title('Ảnh gốc')
plt.imshow(img, cmap='gray')


plt.figure(2)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.title('Ảnh xám')
plt.imshow(gray, cmap='gray')




def filter_f2(img, a, b, beta, L):
    w, h, c = img.shape
    ret = np.zeros(img.shape, dtype=np.uint8)
    for z in range(c):
        for i in range(w):
            for j in range(h):
                if 0 <= img[i][j][z] and img[i][j][z] < a:
                    ret[i][j] = 0
                elif a <= img[i][j][z] and img[i][j][z] < b:
                    ret[i][j][z] = beta*(img[i][j][z] - a)
                elif b <= img[i][j][z] and img[i][j][z] < L:
                    ret[i][j][z] = beta*(b - a)
    return ret


def tangTuongPhan():
    img_1 = filter_f2(img, a=16, b=170, beta=1.5, L=255)

    plt.figure(3)
    plt.title('Tang do tuong phan')
    plt.imshow(img_1)
    plt.show()


tangTuongPhan()

img2 = cv2.imread('./anhtest.jpg',0) 
histg = cv2.calcHist([img2],[0],None,[256],[0,256])
plt.figure(4)
plt.title('Histogram')
plt.plot(histg)
plt.show()

img_1 = filter_f2(img, a=30, b=170, beta=1.5, L=255)
histg = cv2.calcHist([img_1],[0],None,[256],[0,256])
plt.figure(5)
plt.title('Histogram')
plt.plot(histg)
plt.show()


