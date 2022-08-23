import cv2
import numpy as np

# resmi içeri aktar
img = cv2.imread("Lenna_512.png")
cv2.imshow("Orijinal", img)

# yatay
hor = np.hstack((img,img))
cv2.imshow("Horizontal", hor)

# dikey
ver = np.vstack((img,img))
cv2.imshow("Vertical", ver)
