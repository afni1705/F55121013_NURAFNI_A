import cv2

# membaca citra pertama
img1 = cv2.imread('uco.jpg')

# membaca citra kedua
img2 = cv2.imread('uco.jpg')

# mengubah citra menjadi grayscale
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# pengurangan citra
hasil = cv2.absdiff(gray_img1, gray_img2)

# menampilkan citra hasil pengurangan
cv2.imshow('Hasil Pengurangan Citra', hasil)
cv2.waitKey(0)
cv2.destroyAllWindows()

