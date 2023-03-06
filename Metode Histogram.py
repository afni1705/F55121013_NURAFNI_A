import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca citra
img = cv2.imread('1.jpg', 0)

# menghitung histogram
hist = cv2.calcHist([img],[0],None,[256],[0,256])

# menghitung total piksel
total_pixels = img.shape[0] * img.shape[1]

# menghitung jumlah piksel pada setiap intensitas gelap dan terang
dark_pixels = np.sum(hist[0:128])
bright_pixels = np.sum(hist[128:256])

# menghitung persentase piksel gelap dan terang
dark_percentage = (dark_pixels / total_pixels) * 100
bright_percentage = (bright_pixels / total_pixels) * 100

# menampilkan histogram citra awal
plt.subplot(2, 3, 1)
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Histogram Awal')

# citra terang
img_bright = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
hist_bright = cv2.calcHist([img_bright],[0],None,[256],[0,256])

# menampilkan histogram citra terang
plt.subplot(2, 3, 2)
plt.hist(img_bright.ravel(), 256, [0, 256])
plt.title('Histogram Terang')

# citra gelap
img_dark = cv2.convertScaleAbs(img, alpha=0.5, beta=50)
hist_dark = cv2.calcHist([img_dark],[0],None,[256],[0,256])

# menampilkan histogram citra gelap
plt.subplot(2, 3, 3)
plt.hist(img_dark.ravel(), 256, [0, 256])
plt.title('Histogram Gelap')

# citra kontras rendah
img_low_contrast = cv2.equalizeHist(img)
hist_low_contrast = cv2.calcHist([img_low_contrast],[0],None,[256],[0,256])

# menampilkan histogram citra kontras rendah
plt.subplot(2, 3, 4)
plt.hist(img_low_contrast.ravel(), 256, [0, 256])
plt.title('Histogram Kontras Rendah')

# citra kontras tinggi
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_high_contrast = clahe.apply(img)
hist_high_contrast = cv2.calcHist([img_high_contrast],[0],None,[256],[0,256])

# menampilkan histogram citra kontras tinggi
plt.subplot(2, 3, 5)
plt.hist(img_high_contrast.ravel(), 256, [0, 256])
plt.title('Histogram Kontras Tinggi')

# menampilkan citra
plt.subplot(2, 3, 6)
plt.imshow(img_high_contrast, cmap='gray')
plt.title('Citra Kontras Tinggi')

plt.show()

# menampilkan persentase piksel gelap dan terang
print('Persentase Piksel Gelap: ', dark_percentage, '%')
print('Persentase Piksel Terang: ', bright_percentage, '%')
