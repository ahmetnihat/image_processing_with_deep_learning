import glob # resim ve klasörlere erişim sağlayacağız
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# Dense katmanlar, Dropout seyreltme, Flatten düzleştirme,
# Conv2D vrişim ağımız, MaxPooling2D piksel ekleme
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# LabelEncoder verimizi labellayacak 0,1,2,3 diye
#♦ OneHotEncoder etiketlenmiş verimizi kerasta eğitilebilir hale getirecek
from sklearn.model_selection import train_test_split
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
# Uyarıları kapatıyoruz.

imgs = glob.glob("./img/*.png")

width = 125
height = 50

X = []
Y = []

for img in imgs:
    filename = os.path.basename(img)
    # resimlerimizin isimlerini alıyoruz.
    label = filename.split("_")[0]
    # resim isimlerinden alt çizgi ile ayırıp up down rigtları alıyoruz.
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    # resmimizi açıyoruz ve size'ını değiştiriyoruz.
    im = im / 255 # 0 - 1 arasında değerler elde ediyoruz.
    X.append(im) # resimlerimiz
    Y.append(label) # etiketlerimiz
    
X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)
# X.shape[0]: kaç resim olduğunu buluyoruz. genişlik ve yükseklik
# 1: siyah beyaz a.tığımızı belirtiyouz resmi channel değeri keras istiyor.

# sns.countplot(Y)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # fit ne yapacağını öğreniyor, transform öğrendikten sonra dönüştürüyor.
    onehot_encoder = OneHotEncoder(sparse = False)
    # sparse default olarak True ama biz sparce matris elde etmek
    # istemediğimiz için False yapıyoruz.
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

Y = onehot_labels(Y)
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=2,)


# cnn model
model = Sequential()
# Layerlarımızın üzerine ekleyeceğimiz temel yapı.
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(width, height, 1)))
# 32 tane filtre kullanacağım, kullanacağım filtrelerin kernel_size (3,3) olacak
# relu aktivasyon formülünü kullanacağız, girdilerimizin boyutu
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
#♠ başlangıç layerında inputu belirtmemiz yeterli başlangıç layerının çıktısı
# sonraki layerların girdisi olacağı için bir kez daha belirtmemize gerek yok
model.add(MaxPooling2D(pool_size=(2,2)))
# piksel ekleme
model.add(Dropout(0.25))
# %25 olarak seyreltiyoruz.
model.add(Flatten()) # düzleştiriyoruz.
model.add(Dense(128, activation="relu"))
# 128 tane nöron olsun activasyon formülü relu olsun.
model.add(Dropout(0.4))
model.add(Dense(3, activation="softmax"))
# ikiden fazla çıktımız varsa softmax kullanıyoruz.

# if os.path.exists("./trex_weight.h5"):
#    model.load_weights("trex_weight.h5")
#    print("Weights yüklendi")


model.compile(loss = "categorical_crossentropy", optimizer="Adam", metrics = ["accuracy"])
# parametrelerimizi optimize ediyor, metrik modelimizin sonuçlarınıyorumlamamız
# için gerekli olan yapı, yüzde olarak gösterecek.

model.fit(train_X, train_y, epochs = 35, batch_size = 64)

score_train = model.evaluate(train_X, train_y)
print("Eğitim doğruluğu: %", score_train[1]*100)

score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %", score_test[1]*100)


open("model.json","w").write(model.to_json())
model.save_weights("trex_weight.h5")
























