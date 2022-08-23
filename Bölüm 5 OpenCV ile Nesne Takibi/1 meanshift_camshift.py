import cv2
import numpy as np
import time

# kamera aç
cap = cv2.VideoCapture(0)

"""
    Normalde frame okuma işlemini while döngüsü içerisinde yapıyorduk ama
şimdi takip işlemi yapacağımız için bizim amacımız önce yüzü tespit etmek
bunu bir kez gerçekleştireceğiz sonrasında takibi while döngüsü içerisinde 
gerçekleştireceğiz.
"""
# bir tane frame oku
ret, frame = cap.read()
if ret == False:
    print("********************Uyarı********************")
    
# detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0])
track_window = (face_x, face_y, w, h) # meanshift algoritması girdisi

"""
    Takip algoritması yukarda (face_rects) tespit ettiğimiz nesnenin
    kordinatlarına göre trackwindow'unu başlatacak
"""

# region of interest : tespit ettiğimiz kutucuğun içerisi (yüz)
roi = frame[face_y : face_y + h, face_x : face_x + w] # roi = face
# tespit ettiğimiz yüz face_y'den face_y+h'a ve face_x'ten face_x+w'e

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180]) # takip için histogram gerekli
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# takip için gerekli durdurma kriterleri
# count = hesaplanacak maksimum öğe sayısı
# eps = değişiklik
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)
# 5 yenileme veya 1 tane epsilon


while True:
    ret, frame = cap.read()
    
    if ret:
        hsv =cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # histogramı bir görüntüde bulmak için kullanıyoruz
        # piksel karşılaştırma
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        x, y, w2, h2 = track_window
        
        img2 = cv2.rectangle(frame, (x,y), (x+w2, y+h2), (0,0,255), 5)
        
        cv2.imshow("Takip", img2)
        

                
        if cv2.waitKey(1) & 0xFF == ord("q"): break
            
cap.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        