import os
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization, LeakyReLU, MaxPooling2D
from sklearn.model_selection import train_test_split




path = "chest_xray_data_set"
files = os.listdir(path)


data = pd.read_csv(path + "/metadata/chest_xray_metadata.csv")
data = data.drop(data[data.Label_1_Virus_category == "Stress-Smoking"].index) #izbacili smo pusace
img_size = 150


def get_training_data(path, files):
    x = []
    y = []
    for row in data.itertuples():
        if row.Label.lower() == "normal":
            y.append(0)
        elif row.Label.lower() == "pnemonia":
            if row.Label_1_Virus_category.lower() == "bacteria":
                y.append(1)
            else:
                y.append(2)

        for file in files:
            if file == row.X_ray_image_name:
                img_arr = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Fiksiranje velicine slike
                x.append(resized_arr)

    return x, y


x, y = get_training_data(path, files)
#print(len(x))
#print(len(y))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


# Normalizacija podataka
X_train = np.array(X_train) / 255
X_val = np.array(X_val) / 255
X_test = np.array(X_test) / 255


# Prepavljanje velicine
X_train = X_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

X_val = X_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

X_test = X_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)


# Kreiranje modela
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(150,150,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()

# Fitovanje modela
model.fit(X_train, y_train, batch_size=32,epochs=15,verbose=1,validation_data=(X_val, y_val))
test_eval = model.evaluate(X_test, y_test)

#Ispisivanje rezultata
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

