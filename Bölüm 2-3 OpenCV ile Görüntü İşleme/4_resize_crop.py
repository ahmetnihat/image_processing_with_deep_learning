import cv2

# resmi içeri aktar
img = cv2.imread("Lenna_512.png")
print("Resim Boyutu: ", img.shape)

cv2.imshow("Orijinal",img)

# resized
imgResized = cv2.resize(img, (800,800))
print("Resized Img Shape: ", imgResized.shape)
cv2.imshow("Resized Img",imgResized)


# kırp
imgCropped = img[:200,:300] # y ekseni, x ekseni
print("Cropped Img Shape: ", imgCropped.shape)
cv2.imshow("Kirpilmiş Resim", imgCropped)