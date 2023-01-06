---
layout: post
title: Capstone Project
author: [林恒旭]
category: [Lecture]
tags: [jekyll, ai]
---

空中手寫數字識別 In-air Handwriting Digit Recognition

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
官網: [MediaPipe] (https://mediapipe.dev/)

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

1. 建立資料集 dataset
2. 移植程式to Kaggle
3. Kaggle上訓練模型
4. Kaggle上測試模型

---
### 系統測試及成果展示
<iframe width="956" height="538" src="https://www.youtube.com/embed/dQw4w9WgXcQ" title="Rick Astley - Never Gonna Give You Up (Official Music Video)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
