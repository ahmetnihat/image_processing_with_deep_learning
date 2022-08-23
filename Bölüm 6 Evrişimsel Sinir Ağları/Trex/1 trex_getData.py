import keyboard # klavye üzerindde tuşları kullanarak veri toplamamızı sağlar
import uuid # ekrandan kayıt alabileceğimiz kütüphane
import time
from PIL import Image
from mss import mss


mon = {"top": 210, "left": 800, "width": 250, "height":180}
sct = mss() # bize dict'teki ilgili alanı kesip frame haline dönüştürecek

i = 0

def record_screen(record_id, key):
    global i
    i += 1
    print(f"{key}: {i}")
    # key klavyemizdeki bastığımız tuş, i ise kaç kez klavyeye bastığımız
    img = sct.grab(mon) # ekranı al
    im = Image.frombytes("RGB", img.size, img.rgb)
    # RGB formatında okuyacağım, img.size'a göre okuyacağım
    im.save(f"./img/{key}_{record_id}_{i}.png")
    
is_exit = False

def exit():
    global is_exit
    is_exit = True
    
keyboard.add_hotkey("esc", exit)
# exit fonsiyonunu çağıracak esc tuşuna basıldığında

record_id = uuid.uuid4()

while True:
    
    if is_exit: break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.1)
    except RuntimeError: continue