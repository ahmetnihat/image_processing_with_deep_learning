import cv2
import numpy as np

# içe aktar resim
img = cv2.imread("kart2.png")
cv2.imshow("Orijinal", img)

width = 600
height = 820

pts1 = np.float32([[145,3],[5,820],[730,100],[590,915]])

pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])


matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

# çevirme işlemi; (resim, rotasyon matrisi, çıktının boyutu)
imgOutput = cv2.warpPerspective(img, matrix, (width,height))
cv2.imshow("Nihai Resim", imgOutput)