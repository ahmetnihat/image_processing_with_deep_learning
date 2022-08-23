import cv2
import matplotlib.pyplot as plt

# içe aktar
einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap = "gray"),plt.axis("off"), plt.title("Einstein")

# sınıflandırıcı
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# einstein resminin içinde sınıflandırıcıyı kullanarak bir detect yap ve bunu bir rectangle içerisine at
face_rect = face_cascade.detectMultiScale(einstein)

for (x, y, w, h) in face_rect:
    cv2.rectangle(einstein, (x, y), (x+w, y+h), (255,255,255), 10)

plt.figure(), plt.imshow(einstein, cmap = "gray"),plt.axis("off"), plt.title("Einstein Detection")


# barce
# içe aktar
barce = cv2.imread("barcelona.jpeg", 0)
plt.figure(), plt.imshow(barce, cmap = "gray"),plt.axis("off"), plt.title("Barcelona")


face_rect = face_cascade.detectMultiScale(barce, minNeighbors = 13)

for (x, y, w, h) in face_rect:
    cv2.rectangle(barce, (x, y), (x+w, y+h), (255,255,255), 10)

plt.figure(), plt.imshow(barce, cmap = "gray"),plt.axis("off"), plt.title("Barcelona Detection")






# video

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 3)
        
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 3)

        cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()

