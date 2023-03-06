import cv2
import numpy as np

# membaca tiga citra
img1 = cv2.imread('uco.jpg')
img2 = cv2.imread('uco.jpg')
img3 = cv2.imread('uco.jpg')

# mengubah citra menjadi grayscale
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# menghitung rata-rata nilai piksel
averaged_img = cv2.addWeighted(gray_img1, 1/3, gray_img2, 1/3, 0)
averaged_img = cv2.addWeighted(averaged_img, 1, gray_img3, 1/3, 0)

# menampilkan citra hasil image averaging
cv2.imshow('Hasil Image Averaging', averaged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
