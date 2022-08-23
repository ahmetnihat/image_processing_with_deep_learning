from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

mon = {"top": 210, "left": 800, "width": 250, "height":180}
sct = mss()

width = 125
height = 50

# model yükle
model = model_from_json(open("model.json","r").read())
model.load_weights("trex_weight.h5")

# down = 0, rigt = 1, up = 2
labels = ["Down", "Right", "Up"]

framerate_time = time.time()
counter = 0
i = 0
delay = 0.4
key_down_pressed = False

while True:
    img = sct.grab(mon) # ekranı al
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255
    
    X = np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1)
    r = model.predict(X)
    # modelimizi kullanarak bir tahmin işlemi gerçekleştir ve bunu r'ye eşitle
    
    result = np.argmax(r)
    # labels listemizin içinde hangi değer en büyükse onu al demek [0,1,0] - 1
    
    if result == 0: # down = 0
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
        
    elif result == 2: # up = 2
        
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay+0.07)
        keyboard.press(keyboard.KEY_UP)
        if i < 1500: # 1500. frame'e kadar normal akıyor sonra hızlanıyor oyun.
            time.sleep(0.4)
        elif 1500 < i and i < 5000:
            time.sleep(0.3)
        else:
            time.sleep(0.2)
            
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
        
    counter += 1
    
    if (time.time() - framerate_time) > 1:
        counter = 0
        framerate_time = time.time()
        if i <= 15:
            delay -= 0.003
        else:
            delay -= 0.005
        if delay < 0:
            delay = 0
            
        print("--------------------------")
        print(f"Down: {r[0][0]} \nRight: {r[0][1]} \nUp: {r[0][2]} \n")
        i += 1
        
        
        