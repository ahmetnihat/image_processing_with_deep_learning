import cv2
import matplotlib.pyplot as plt
import numpy as np

# resmi içe aktar
img = cv2.imread("london.jpg",0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

# kenar tespiti, threshold kullanmadığımız için her şeyin kenarını tespit etti sudaki dalgaların bile
edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

med_val = np.median(img)
print(med_val)

low = int(max(0, ( 1- 0.33) * med_val))
high = int(min(255, (1 + 0.33) * med_val))
print(low,high)

edges = cv2.Canny(image=img, threshold1=low, threshold2=high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

# blur
blurred_img = cv2.blur(img, ksize = (5,5))
plt.figure(), plt.imshow(blurred_img, cmap = "gray"), plt.axis("off")

med_val2 = np.median(blurred_img)
print(med_val2)

low2 = int(max(0, ( 1- 0.33) * med_val2))
high2 = int(min(255, (1 + 0.33) * med_val2))
print(low2, high2)



edges2 = cv2.Canny(image=blurred_img, threshold1=low2, threshold2=high2)
plt.figure(), plt.imshow(edges2, cmap = "gray"), plt.axis("off")