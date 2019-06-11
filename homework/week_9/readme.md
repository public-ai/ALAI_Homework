# Week 9 - Convolution Neural Network

CIFAR-10 데이터를 학습하고 모델의 정확도를 최종적으로 90% 이상으로 끌어올리는 여정을 시작합니다.

Readme 을 끝까지 읽어주세요!

이 번 Homework 부터 모델의 상세 조건들을 작성해주세요.

아래와 같이 표를 적상해주세요.

- 목표 Receptive Field : 28 <br>
- Convolution Phase 후  출력 크기  :  4 <br>
- Regularization  : L2
- Batch size : 120
- Learning rate : 0.0001
- Data normalization : min max normalization
- Standardization : None


| 층  | 종류|필터 갯수  | 필터 크기 | 스트라이드 | 패딩   | Dropout | output size |
|--- |--- |----|----|----|----|----| ---|
| c1 |conv| 64| 3x3| 1  | SAME | None| 32x32 |
| s2 |max-pooling| None| 3x3| 2  | SAME | None|16x16 |
| c3 |conv| 128| 3x3| 2  | SAME |NOne |16x16 |
| s4 |max-pooling| None| 3x3| 2  | SAME | None|8 x8 |
| c5 |conv| 128| 3x3| 2  | SAME | None |8 x8 |
| s6 |conv| 256| 3x3| 2  | SAME | None |4 x 4 |
| c7 |conv| 256| 1x1| 2  | SAME | None |4 x 4 |
| f8 ||| | FC 256  | |  ||
| f8 ||| | Dropout 0.7 | |  ||
| f9 ||| | FC 256  | |  ||
| f9 ||| | Dropout 0.6 | |  ||
| f10||| | FC 10   | |  ||

결과가 나오면 결과에 대해 해석하고 comment 을 달아주세요. 그리고 모델을 발전 시킬수 있는 방향을 적어주세요.

위 표와 똑같이 적을 필요는 없습니다. 하지만 중요한 정보들이 기록되고 그 기록을 바탕으로 모델에 대해 추론하고 발전 시킬수 있어야 합니다.

가령 위 표로 학습시켰을때 아래와 같은 결과가 나온다면

![Imgur](https://i.imgur.com/yqrIm5u.png)

'discussion : overfitting 이 심함. dropout, regularization, 데이터 수집 , 데이터 augmentation 등을 추가적으로 실시해야함.'

이후 discussion 에서 추론한 정보를 바탕으로 모델을 재설계 하고 다시 모델을 돌린후 결과를 수집합니다.

이런 작업을 10번 이상 진행해주세요.

각 report 는 각기 다른 readme 에 작성해주세요.

숙제 1차 마감 (~6/26)