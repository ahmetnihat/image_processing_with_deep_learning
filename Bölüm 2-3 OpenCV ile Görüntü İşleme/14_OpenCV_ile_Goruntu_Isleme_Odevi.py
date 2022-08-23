from math import ceil
# opencv kütüphanesini içe aktaralım
import cv2
# matplotlip kütüphanesini içe aktaralm
import matplotlib.pyplot as plt

# resmi siyah-beyaz olarak içe aktaralım
img = cv2.imread("odev1.jpg",0)
# resmi çizdirelim
cv2.imshow("Kedi Kopek At", img)

# resmin boyutlarına bakalım
x, y = img.shape
print(x, y)

# resmi 4/5 oranında yeniden boyutlandırıp resmi çizdirelim.
x = int(ceil(((x/5)*4)))
y = int(ceil(((y/5)*4)))
newSize = cv2.resize(img,(y,x))
print("newSize: ", newSize.shape)
cv2.imshow("New Size", newSize)


# orijinal resme bir yazı ekleyelim mesela "kopek" ve resmi çizdirelim.
cv2.putText(img, "kopek", (200,35), cv2.FONT_HERSHEY_PLAIN, fontScale = 3, color = (255,0,0))
cv2.imshow("Kopek Yazili",img)

# orijinal resmin 50 threshold değeri üzerindekileri beyaz yap altındakileri siyah yapalım,
# binary threshold yöntemi kullanalım ve resmi çizdirelim
img2 = cv2.imread("odev1.jpg")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

plt.figure(), plt.imshow(img2, cmap = "gray"), plt.axis("off"), plt.show()

_, thresh_img = cv2.threshold(img, thresh = 50, maxval = 255, type = cv2.THRESH_BINARY)
plt.figure(), plt.imshow(thresh_img, cmap = "gray"), plt.axis("off"), plt.show()

# orijinal resme gaussian bulanıklaştırma uygulayalım ve resmi çizdirelim
img3 = cv2.imread("odev1.jpg")
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img3), plt.axis("off"), plt.title("Orijinal Image"), plt.show()

gb = cv2.GaussianBlur(img3,(3,3),7)
plt.figure(), plt.imshow(gb), plt.axis("off"), plt.title("Gaussian Blur"), plt.show()


# orijinal resme laplacian gradyan uygulayalım ve resmi çizdirelim.
lg = cv2.Laplacian(img3, cv2.CV_16S)
plt.figure(), plt.imshow(lg), plt.axis("off"), plt.title("Laplacian"),plt.show()

# orijinal resmin histogramını çizdirelim
img_hist = cv2.calcHist([img3], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(img_hist), plt.title("Image Histogram"), plt.show()
