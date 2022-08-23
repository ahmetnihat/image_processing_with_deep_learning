import cv2
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# blurring (detayı azaltır, gürültüyü engeller)
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(), plt.imshow(img), plt.axis("off"), plt.title("Orijinal"), plt.show()

"""
ortalama bulanıklaştırma yöntemi
---------------
Belirlediğimiz kutucuktaki piksellerin ortalamasını alır.

"""
# (resim, ortalama alacak kutucuk)
# opencv'de girdiler src = source, çıktılar = dst olarak adlandırılır.
dst2 = cv2.blur(img, ksize = (3,3))
plt.figure(), plt.imshow(dst2), plt.axis("off"), plt.title("Ortalama Blur"), plt.show()

"""
Gaussian Blur
---------------
Kernel olarak adlandırılan kutucuğumuz mevcut ortalama almak yerine x ve y yönlerinde
sigma değerleri yazarak bu kutucukların 2 boyutlu bir gauss olmasını sağlıyoruz.
Kutucukların içerisinden bulunan değerlere göre piksellerin üzerinde işlemler yapılıyor.

"""

gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
plt.figure(), plt.imshow(gb), plt.axis("off"), plt.title("Gauss Blur")

"""
medyan blur
---------------
kutucukarın içerisinde 3,3 lük bir matris düşünürsek kutucuktaki matris sıralı şekilde vektör gibi
yazılarak medyan değeri alınır.

"""

mb = cv2.medianBlur(img, ksize = 3)
plt.figure(), plt.imshow(mb), plt.axis("off"), plt.title("Median Blur"), plt.show()


def gaussianNoise(image):
    row, col, ch = image.shape
    # satır sütun ve renk değeri 0-3
    mean = 0
    var = 0.05
    sigma = var**0.5
    # Gaussian'da ortalama ve standart sapmaya ihtiyacımız vardır.
    
    gauss = np.random.normal(mean, sigma, (row,col,ch))
    # normal dağılımlı 0 ortalamalı, sigma standart sapmalı, satır sütun ve kanallı
    gauss = gauss.reshape(row,col,ch) # boyutundan bir kez daha emin oluyoruz.
    noisy = image + gauss # resim ve eld ettiğimiz gürültü toplanıyor ve gürültülü resim elde ediliyor.
    
    return noisy
    
    
# içe aktar normalize et
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
# /255 = 255 olan 1, 0 olan 0
plt.figure(), plt.imshow(img), plt.axis("off"), plt.title("Orijinal"), plt.show()


gaussianNoisyImage = gaussianNoise(img)
plt.figure(), plt.imshow(gaussianNoisyImage), plt.axis("off"), plt.title("Gaussian Noisy"), plt.show()


# gauss blur    
gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize = (3,3), sigmaX = 7)
plt.figure(), plt.imshow(gb2), plt.axis("off"), plt.title("Gaussian Noisy Gauss Blur")


def saltPepperNoise(image):
    
    row, col, ch = image.shape
    s_vs_p = 0.5
    
    amount = 0.004
    
    noisy = np.copy(image)
    
    # salt beyaz
    num_salt = int(np.ceil(amount * image.size * s_vs_p))
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords] = 1
    
    # pepper siyah
    num_pepper = int(np.ceil(amount * image.size * (1-s_vs_p)))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords] = 0
    
    return noisy





spImage = saltPepperNoise(img)
plt.figure(), plt.imshow(spImage), plt.axis("off"), plt.title("SP Image")


mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize = 3)
plt.figure(), plt.imshow(mb2), plt.axis("off"), plt.title("Median Blur 2"), plt.show()
