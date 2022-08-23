import cv2
import matplotlib.pyplot as plt

# resmi içeri aktar
img = cv2.imread("img1.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


plt.figure()
# cmap= color map
plt.imshow(img, cmap = "gray")
plt.axis("off")
plt.show()

# eşikleme
_, thresh_img = cv2.threshold(img, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY)
plt.figure()
plt.imshow(thresh_img, cmap = "gray")
plt.axis("off")
plt.show()

# uyarlamal eşik değeri
thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,8)
plt.figure()
plt.imshow(thresh_img2, cmap = "gray")
plt.axis("off")
plt.show()