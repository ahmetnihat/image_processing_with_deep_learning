# opencv kütüphanesini içe aktaralım
import cv2
# numpy kütüphanesini içe aktaralım
import numpy as np
# resmi siyah beyaz olarak içeri aktaralım
img = cv2.imread("yayalar.jpg",0)
# resim üzerinde bulunan kenarları tespit edelim ve görselleştirelim.
img_resized = cv2.resize(img, (800,600))
edges = cv2.Canny(img_resized,threshold1=100,threshold2=200)
cv2.imshow("edges",edges)
# yüz tespiti için gerekli haar cascade'i içe aktaralım
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# yüz tespiti yapıp sonuçları görselleştirelim
face_rect = face_cascade.detectMultiScale(img,1.075,27)
for (x,y,w,h) in face_rect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
img_resized = cv2.resize(img, (800,600))
cv2.imshow("Taninmis Yuz",img_resized)

# HOG ilişkilendirelim insan tespiti algoritmamızı çağıralım ve svm'i set edelim.
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# resme insan tespiti algoritmamızı uygulayalım ve görselleştirelim.
(rects, wights) = hog.detectMultiScale(img,padding=(8,8),scale = 1.097)
for (x,y,w,h) in rects:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
img_resized = cv2.resize(img, (800,600))
cv2.imshow("Body Detected",img_resized)