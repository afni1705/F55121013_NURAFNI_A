import cv2

# membaca dua citra
img1 = cv2.imread('uco.jpg')
img2 = cv2.imread('uco.jpg')

# mengubah citra menjadi grayscale
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# mengurangi kedua citra
diff_img = cv2.absdiff(gray_img1, gray_img2)

# menampilkan citra hasil difference image
cv2.imshow('Hasil Difference Image', diff_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
