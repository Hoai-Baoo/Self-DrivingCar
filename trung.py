# Import socket module
from keras import backend as keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from keras.layers import *
from keras.models import *
import tensorflow
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import backend as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import skimage.transform as trans
import skimage.io as io
import os
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
import socket
import cv2
import numpy as np
import math



def unet(pretrained_weights=None, input_size=(160, 640, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(32, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(8, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    # model.summary()

    return model

model = unet()

model.load_weights('Trungai')

global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('host.docker.internal', PORT))


def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed


if __name__ == "__main__":
    try:
        while True:

            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """

            message_getState = bytes("0", "utf-8")
            s.sendall(message_getState)
            state_date = s.recv(100)

            try:
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                ).split(' ')
            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)

            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                    ), -1
                )
                # them print
                print(sendBack_Speed, sendBack_angle)
                #
                print(current_speed, current_angle)
                # your process here

                # Fix lọc
                # image = cv2.GaussianBlur(image, (5, 5), 0)
                #
                image = image[120:, :]
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                dsize = cv2.resize(image, dsize=(640, 160))
                hsv = cv2.cvtColor(dsize, cv2.COLOR_BGR2HSV)
                hsv = hsv[None, :, :, :]
                y_pre = model.predict(hsv)[0]
                y_pre = y_pre.reshape(160, 640)
                y_pre = np.where(y_pre < 0.5, 0.0, 1.0) * 255
                y_pre = cv2.resize(y_pre, dsize=(640, 240))

                # Filter
                y_pre = cv2.GaussianBlur(y_pre, (5, 5), 0)
                y_pre = cv2.GaussianBlur(y_pre, (5, 5), 0)
                y_pre = cv2.GaussianBlur(y_pre, (5, 5), 0)
                y_pre = cv2.GaussianBlur(y_pre, (5, 5), 0)
                y_pre = cv2.GaussianBlur(y_pre, (5, 5), 0)
                #

                # fix choi
                # y_pre = np.where(y_pre < 127, 0.0, 255.0)
                #
                print(y_pre)
                arr = []
                # đổi từ 50 - >60
                line = y_pre[60, :]
                for x, y in enumerate(line):
                    if y == 255:
                        arr.append(x)
                arrmax = max(arr)
                arrmin = min(arr)
                center = int((arrmax+arrmin)/2)

                sendBack_angle = math.degrees(
                    math.atan((center - y_pre.shape[1]/2) / (y_pre.shape[0] - 60)))

                cv2.circle(y_pre, (arrmin, 60), 5, (255, 255, 255), 5)
                cv2.circle(y_pre, (arrmax, 60), 5, (255, 255, 255), 5)
                cv2.line(y_pre, (center, 60),
                         (int(y_pre.shape[1]/2), y_pre.shape[0]), (0, 0, 0), 5)

                sendBack_angle = (sendBack_angle*25)/90

                if -5 < sendBack_angle < 5:
                    # sendBack_Speed = 45
                    sendBack_Speed = 60
                elif -10 < sendBack_angle < 10:
                    sendBack_Speed = 5
                else:
                    sendBack_Speed = 1

                cv2.imshow('dsize', dsize)
                cv2.imshow('predict', y_pre)
                cv2.waitKey(1)
                # Control(angle, speed)
                Control(sendBack_angle, sendBack_Speed)

            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()


