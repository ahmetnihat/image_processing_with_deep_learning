import cv2
import os

files = os.listdir()
img_path_list = []

for f in files:
    if f.startswith("yaya"):
        img_path_list.append(f)
        
print(img_path_list)


# hog tanımlayıcısı : tespit algoritması
hog = cv2.HOGDescriptor()
# tanımlayıcıya SVM ekle
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_path_list:
    print(imagePath)
    
    image = cv2.imread(imagePath)
    
    (rects, wights) = hog.detectMultiScale(image, padding = (8,8), scale = 1.05)
    # padding: resmin etrafını 8,8 lik pencerelerle dolaşırken resmin etrafındaki
    # boşlukları 0 ile dolduruyor ve böylece resimde boyut kaybı yaşamamış oluyoruz.
    
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x,y), (x+y,y+h), (0,0,255),2)
    
    
    cv2.imshow("Yaya", image)
    
    if cv2.waitKey(0) & 0xFF == ord("q"): continue