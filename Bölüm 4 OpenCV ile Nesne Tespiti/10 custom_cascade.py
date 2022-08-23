"""
1) veri seti:
    n (negatif resimler tam tersi herhangi bir resim), p (pozitif resimler, tespit etmek istediğimiz objemiz)
2) cascade programı indir
3) cascade oluştur
4) cascade kullanarak tespit algoritması yaz
"""

import cv2
import os

# resim depo klasörü
path = "images" # kendi kameramızda kaydettiğimiz resimler bu klasörün içinde bulunacak

# resim boyutu
imgWidth = 180 #◘px
imgHeight = 120 #px

# video capture
cap = cv2.VideoCapture(0)
cap.set(3,640) #♠ genişlik
cap.set(4,480) # boyut
cap.set(10, 180) # brigtness

global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder += 1
    os.makedirs(path + str(countFolder))
    
saveDataFunc()

count = 0
countSave = 0

while True:
    
    success, img = cap.read()
    
    if success:
        img = cv2.resize(img, (imgWidth, imgHeight))
        
        # her 5 resimden birini
        if count % 5 == 0:
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png", img)
            # buraya bu isimle resmimizi kaydedeceğiz
            countSave += 1
            print(countSave)
        count += 1
        
        cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()











































