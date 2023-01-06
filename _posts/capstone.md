---
layout: post
title: Capstone Project
author: [林恒旭]
category: [Lecture]
tags: [jekyll, ai]
---

空中手寫數字辨識 In-air Handwriting Digit Recognition

---
## 期末實作描述
透過**MediaPipe**在畫面上寫下數字，並透過**MNIST-CNN**辨識出寫下的數字
### 系統簡介及功能說明

伸出你的**食指**，在畫面中的紅框寫下你的數字!

![](https://github.com/willy610515/AI-course/blob/gh-pages/images/01.png)

當寫清楚後就可以馬上辨識出來所寫的數字

按下**R**鍵可以清除畫面上的筆跡

![](https://github.com/willy610515/AI-course/blob/gh-pages/images/02.png)

按下**Q**鍵可以離開程式
---
### 使用技術簡介

## MediaPipe
MediaPipe 是 Google Research 所開發的多媒體機器學習模型應用框架，支援 JavaScript、Python、C++ 等程式語言，可以運行在嵌入式平臺 ( 例如樹莓派等 )、移動設備 ( iOS 或 Android ) 或後端伺服器，目前如 YouTube、Google Lens、Google Home 和 Nest...等，都已和 MediaPipe 深度整合。

![](https://d1tlzifd8jdoy4.cloudfront.net/wp-content/uploads/2020/05/795316b92fc766b0181f6fef074f03fa-1-960x504.png)

官網: [MediaPipe](https://mediapipe.dev/)

## CNN

![](https://editor.analyticsvidhya.com/uploads/94787Convolutional-Neural-Network.jpeg)

卷積神經網路（Convolutional Neural Network, CNN）是一種前饋神經網路，它的人工神經元可以回應一部分覆蓋範圍內的周圍單元，[1]對於大型圖像處理有出色表現。

卷積神經網路由一個或多個卷積層和頂端的全連通層（對應經典的神經網路）組成，同時也包括關聯權重和池化層（pooling layer）。這一結構使得卷積神經網路能夠利用輸入資料的二維結構。與其他深度學習結構相比，卷積神經網路在圖像和語音辨識方面能夠給出更好的結果。這一模型也可以使用反向傳播演算法進行訓練。相比較其他深度、前饋神經網路，卷積神經網路需要考量的參數更少，使之成為一種頗具吸引力的深度學習結構。
[Wiki](https://zh.wikipedia.org/zh-tw/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)摘錄

## MNIST
想要訓練一個 Neural Network，不可缺少的是一大堆的訓練資料。針對「手寫數字圖像」分類問題，最有名的資料集是 MNIST 資料集 (MNIST Dataset)，MNIST 中包含了上萬張的手寫數字圖像以及每一張圖像正確的標籤 (Label)。

MNIST 是 Modified National Institute of Standards and Technology database 的縮寫，其為 NIST 機構所建立的兩個資料集的修改版本。

下圖為 MNIST 資料集中的幾張範例圖片：

![](https://datasciocean.tech/wp-content/uploads/2022/02/MNIST-Dataset.jpg)

---
### 製作步驟

1. 使用MNIST 資料集在Kaggle上訓練模型
2. Kaggle上測試模型
3. 開發在螢幕上以食指來繪製的程式
4. 加入在Kaggle上訓練好的模型
5. 辨識手寫數字

---
### 程式說明

## MNIST-CNN訓練模型
Kaggle:[MNIST-CNN](https://www.kaggle.com/code/willy610515/mnist-cnn/notebook)

```
#Load Dataset
from tensorflow.keras import datasets # boston_housing, cifar10, cifar100, fashion_mnist, imdb, mnist, reuters
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

#normalized image data
# data converted from integer to float
x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)
print(x_test.shape)

# reshape to add color channel into data
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print(x_train.shape)
print(x_test.shape)

#encoding label data
from tensorflow.keras import utils
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)

#Build Model
from tensorflow.keras import models, layers

inputs = layers.Input(shape=input_shape)
x = layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(inputs)
x = layers.MaxPool2D(pool_size = (2, 2))(x)
# 2nd Conv layer        
x = layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
x = layers.MaxPool2D(pool_size = (2, 2))(x)
# Fully Connected layer        
x = layers.Flatten()(x)
x = layers.Dense(128)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.summary()

# Compile Model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

#Train Model
history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test))

#Save Model
models.save_model(model, 'mnist_cnn.hdf5')

#Test Model
model = models.load_model('mnist_cnn.hdf5')
preds = model.predict(x_test[0].reshape(-1,28,28,1))
print(int(np.argmax(preds)))

# Evaluate Model
score = model.evaluate(x_test, y_test)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# Show Train History
keys=history.history.keys()
print(keys)

def show_train_history(hisData,train,test): 
    plt.plot(hisData.history[train])
    plt.plot(hisData.history[test])
    plt.title('Training History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history(history, 'loss', 'val_loss')
show_train_history(history, 'accuracy', 'val_accuracy')
dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
```

## 繪製程式

```
# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '1'
    else:
        return ''

cap = cv2.VideoCapture(0)            # 讀取攝影機
fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
lineType = cv2.LINE_AA               # 印出文字的邊框

# mediapipe 啟用偵測手掌
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    w, h = 540, 310                                        # 影像尺寸
    a, b, c, d = 350, 50, 110, 110                         # 定義擷取數字的區域位置和大小
    draw = np.zeros((h,w,4), dtype='uint8')                # 繪製全黑背景，尺寸和影像相同
    dots = []                                              # 使用 dots 空串列記錄繪圖座標點
    color = (0,0,255,255)                                  # 設定預設顏色為紅色
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (w,h))                       # 縮小尺寸，加快處理效率
        img = cv2.flip(img, 1)
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # 偵測手勢的影像轉換成 RGB 色彩
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)        # 畫圖的影像轉換成 BGRA 色彩
        results = hands.process(img2)                      # 偵測手勢
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = []                         # 記錄手指節點座標的串列
                for i in hand_landmarks.landmark:
                    # 將 21 個節點換算成座標，記錄到 finger_points
                    x = i.x*w
                    y = i.y*h
                    finger_points.append((x,y))
                if finger_points:
                    finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
                    text = hand_pos(finger_angle)            # 取得手勢所回傳的內容
                    if text == '1':
                        fx = int(finger_points[8][0])        # 如果手勢為 1，記錄食指末端的座標
                        fy = int(finger_points[8][1])
                        dots.append([fx,fy])                 # 記錄食指座標
                        dl = len(dots)
                        if dl>1:
                            dx1 = dots[dl-2][0]
                            dy1 = dots[dl-2][1]
                            dx2 = dots[dl-1][0]
                            dy2 = dots[dl-1][1]
                            cv2.line(draw,(dx1,dy1),(dx2,dy2),color,8)  # 在黑色畫布上畫圖
                    else:
                        dots = [] # 如果換成別的手勢，清空 dots
        
        # 將影像和黑色畫布合成
        for j in range(w):
            img[:,j,0] = img[:,j,0]*(1-draw[:,j,3]/255) + draw[:,j,0]*(draw[:,j,3]/255)
            img[:,j,1] = img[:,j,1]*(1-draw[:,j,3]/255) + draw[:,j,1]*(draw[:,j,3]/255)
            img[:,j,2] = img[:,j,2]*(1-draw[:,j,3]/255) + draw[:,j,2]*(draw[:,j,3]/255)
```
##辨識手寫數字
```
cnn = load_model('mnist_cnn.hdf5')               # 載入模型

img_num = np.zeros((h,w,4), dtype='uint8')       # 繪製全黑背景，尺寸和影像相同
        for j in range(w):
            img_num[:,j,0] = img_num[:,j,0]*(1-draw[:,j,3]/255) + draw[:,j,0]*(draw[:,j,3]/255)
            img_num[:,j,1] = img_num[:,j,1]*(1-draw[:,j,3]/255) + draw[:,j,1]*(draw[:,j,3]/255)
            img_num[:,j,2] = img_num[:,j,2]*(1-draw[:,j,3]/255) + draw[:,j,2]*(draw[:,j,3]/255)
        img_num = img_num[b:b+d, a:a+c]          # 擷取辨識的區域
        
        test=img_num
        test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)    # 顏色轉成灰階
        test = cv2.resize(test,(28,28))                  # 縮小成 28x28，和訓練模型對照
        test = test/255.0                                # 轉換格式
        test = test.reshape(-1,28,28,1)
        img_pre = cnn.predict(test[0].reshape(-1,28,28,1))  # 進行辨識
        num = str(int(np.argmax(img_pre)))                  # 取得辨識結果
        cv2.putText(img, num, (a,b-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, lineType) # 印出文字


        cv2.rectangle(img,(a,b),(a+c,b+d),(0,0,255),3)  # 標記辨識的區域
```
## 程式下載

Github: [https://github.com/willy610515/AI-course/tree/gh-pages/In-air%20Handwriting%20Digit%20Recognition](https://github.com/willy610515/AI-course/tree/gh-pages/In-air%20Handwriting%20Digit%20Recognition)
---
### 系統測試及成果展示

![](https://github.com/willy610515/AI-course/blob/gh-pages/images/test.gif)
---
### 參考資料

[Mediapipe 辨識手指，用手指在影片中畫圖](https://steam.oxxostudio.tw/category/python/ai/ai-mediapipe-finger-draw.html)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
