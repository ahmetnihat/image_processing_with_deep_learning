import cv2

# capture
cap = cv2.VideoCapture(0)
 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,height)

# video kaydet
writer = cv2.VideoWriter("video_kaydi.mp4",cv2.VideoWriter_fourcc(*"DIVX"),25,(width,height))
# fourcc 4 karakterli codec kodu windows için başka os'larda farklı, fps, video kaydedici boyut

while True:
    
    ret, frame = cap.read()
    cv2.imshow("Video",frame)
    
    # save
    writer.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
writer.release()
cv2.destroyAllWindows()