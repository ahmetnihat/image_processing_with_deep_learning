import cv2
import matplotlib.pyplot as plt
import numpy as np

# resmi içeri aktar
img = cv2.imread("sudoku.png",0)
img = np.float32(img) # ondalıklı sayılara çeviriyoruz.
print(img.shape)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

# harris corner detection
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
# blockSize = komşuluk boyutu ne kadar komşusuna bakacağımızı belirliyor.
# ksize = kutucuğun boyutu, k = harris dedektöründeki free parametrelerden bir tanesi
plt.figure(), plt.imshow(dst, cmap= "gray"), plt.axis("off")

# tespit ettiği noktaları genişletme
dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap= "gray"), plt.axis("off")


# shi tomsai detection
img = cv2.imread("sudoku.png",0)
img = np.float32(img) # ondalıklı sayılara çeviriyoruz.
corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
corners = np.int64(corners)


for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,(125,125,125),cv2.FILLED)
    
plt.imshow(img), plt.axis("off")