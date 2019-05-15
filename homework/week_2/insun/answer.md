#### 우리는 배우의 웃는 횟수가 60번이고, 
#### 우는 횟수가 0번이었던 영화의 카테고리를 예측해보고자 합니다.

```python3
classify_knn(inX,dataset,labels,4)
```

#### 우는 횟수가 단 한번도 발생하지 않았지만, 
#### 이를 drama로 판단하는 잘못된 결과를 반환하였습니다.

#### 로맨스 영화의 특징 상, 배우가 웃는 횟수가 배우가 우는 횟수보다 훨씬 많이 일어납니다. 그래서 웃는 횟수의 값의 범위와 우는 횟수의 값의 범위가 매우 상이합니다. 각 Feature의 범위가 매우 다르기 때문입니다.

#### 이러한 문제를 해결하기 위해서는 어떤 식으로 KNN을 수정해야 할까요?



### 수정한 코드
```python3
def classify_knn(inX, dataset, labels, K):
    ##########
    # CODE HERE!
    # 위의 결과가 COMEDY가 나올 수있도록 코드를 수정해 주세요!
    ##########    
    
    # (1-1) 우리가 분류항목을 알고자 하는 점 (inX)와 
    # 알고 있는 점들(dataset)과의 x, y 위치값 비율 체크    
    x = np.sum((inX[0]-dataset), axis=1)
    y = np.sum((inX[1]-dataset), axis=1)
    ratio = x / y
    
    
    # (2) 오름 차순으로 거리의 길이를 정렬
    sorted_index = ratio.argsort()
    
    
    # (3) 가능성이 높은 순서대로 K개의 아이템 추출
    sorted_labels = labels[sorted_index]
    K_nearest_labels = sorted_labels[:]   # 뒷부분부터 4개를 추출하려면 어떻게 해야 하나요?

    
    # (4) K개의 아이템에서 가장 많은 분류 항목 찾기
    _labels, count_labels = np.unique(K_nearest_labels,
          return_counts=True)
    
    # (5) 해당 항목 반환
    return _labels[count_labels.argmax()]
```

### 결과값을 확인하기 위하여 함수 호출
```python3
inX = [60, 0]
inX_lable = classify_knn(inX,dataset,labels,4)
print(inX_lable)
# 아래와 같은 결과가 나와야 합니다.
# >>> COMEDY
```

### [60,0] 위치값과 label을 기존 데이타 array에 자동으로 추가

```python3
dataset = np.array([
    [120, 3],
    [105, 2],
    [25, 12],
    [32, 15],
    [17, 9],
    [98, 5],
    [130, 1],
    [0, 16],
    [40, 20],
    [100, 10],
    [60, 0] ## 임의로 추가한 데이타
])

labels = np.array(["comedy", "comedy", "drama",
                   "drama","drama","comedy","comedy",
                   "drama","drama","comedy", 
                   "comedy"])  ## 임의로 추가한 데이타


# dataset과 label에 data를 추가하는 방법이 궁금합니다

#print(dataset.shape)
# inputX = np.array(inX).reshape(1,2)
#print(inputX.shape)
#dataset = np.stack([dataset, inputX], axis=0)
#print(dataset)

```

### 변경한 수식이 정상적으로 동작하는지 확인함

```python3

plt.title("The Category of Movie")
plt.scatter(dataset[labels=="comedy",0],dataset[labels=="comedy",1],
            label='comedy', c='g')
plt.scatter(dataset[labels=="drama",0],dataset[labels=="drama",1],
            label='drama', c='r')


plt.xlim(-10,140)
plt.ylim(-10,40)

plt.xlabel('The number of smile')
plt.ylabel('The number of cry')
plt.legend()
plt.show()
```