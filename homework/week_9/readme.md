# Week 9 - Convolution Neural Network

CIFAR-10 데이터를 학습하고 모델의 정확도를 최종적으로 90% 이상으로 끌어올리는 여정을 시작합니다.

이 번 Homework 부터 모델의 상세 조건들을 작성해주세요.

아래와 같이 표를 적상해주세요.
똑같을 필요는 없습니다. 하지만 중요한 정보들이 기록되고 그 기록을 바탕으로 모델에 대해 추론하고 발전 시켜야 합니다


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

결과가 나오면 결과에 대해 해석하고 comment 을 달아주세요.