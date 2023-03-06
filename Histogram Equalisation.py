import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca citra
img = cv2.imread('1.jpg', 0)

# melakukan histogram equalization
img_eq = cv2.equalizeHist(img)

# menampilkan citra dan histogram sebelum dan sesudah equalisasi
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('1 Awal')
plt.subplot(2,2,2)
plt.hist(img.ravel(),256,[0,256])
plt.title('Histogram Awal')
plt.subplot(2,2,3)
plt.imshow(img_eq, cmap='gray')
plt.title('Citra Setelah Equalisasi')
plt.subplot(2,2,4)
plt.hist(img_eq.ravel(),256,[0,256])
plt.title('Histogram Setelah Equalisasi')

plt.show()
