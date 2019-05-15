### Week 2. Homework 3. Pandas & Matplotlib_타이타닉 데이터셋 분석하기

#### Pandas를 통해 데이터 읽어오기
```python3
%matplotlib inline  
import matplotlib.pyplot as plt  

import pandas as pd
import numpy as np

!wget https://s3.ap-northeast-2.amazonaws.com/pai-datasets/alai-deeplearning/titanic_dataset.csv
df = pd.read_csv("./titanic_dataset.csv")

# 1. 나눔 폰트를 다운받기
!apt-get update -qq
!apt-get install fonts-nanum* -qq

import matplotlib.font_manager as fm
# 2. 나눔 폰트의 위치 가져오기 
system_font = fm.findSystemFonts() # 현재 시스템에 설치된 폰트
nanum_fonts = [font for font in system_font if "NanumBarunGothic.ttf" in font]
font_path = nanum_fonts[0] # 설정할 폰트의 경로

# 3. 나눔 폰트로 설정하기
font_name = fm.FontProperties(fname=font_path, size=10).get_name()
plt.rc("font",family=font_name)

# 4. 폰트 재설정하기
fm._rebuild()

# 5. (optional) minus 기호 깨짐 방지
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

```

#### (1) 각 열 별로 얼마나 결측(missing value)되어 있는지 확인하기
```python3
ataset = pd.DataFrame(df)

data_count = dataset.iloc[:,0].count()                              # 행방향 Data 개수
dataset.loc["결측 갯수"] = data_count - dataset.iloc[:].count().values   # data가 없는 개수

answer = pd.DataFrame(dataset.loc["결측 갯수"]).T
answer
```

#### (2) 생존자 수 확인
```python3
dataset = pd.DataFrame(df)
                      
# Mask(survived or dead)
mask_survived = dataset.loc[:,"Survived"] == 1
mask_dead = dataset.loc[:,"Survived"] == 0

survived = mask_survived.astype('int').sum()   
dead = mask_dead.astype('int').sum()     

# Another answer
# survived = df["Survived"].sum()
# dead = len(df["Survived"]) - df["Survived"].sum() 

# make pandas Data Frame
arr = np.array( [["사망", dead],
               ["생존", survived]])
columns = ["생존유무", "값"]
# index=["사망", "생존"]

answer = pd.DataFrame(arr, columns = columns)
answer
```

#### (3) 연령 추론
타이타닉 내 승객 이름에는 Title이 들어갑니다. 나이 정보가 없을 경우, 이 Title에 따라 대략적인 나이를 추론할 수 있습니다. 이번에는 각 Name 컬럼에서 Title을 추출 후, Title 별로 평균 나이가 어떻게 되는지 확인해 주세요.

# 진행중입니다!
```python3
dataset = pd.DataFrame(df)
#print(dataset)

data_count = dataset.iloc[:,0].count()                           

mask_no_age_info = (dataset.loc[:,"Age"]).values.astype('int') <= 0
# print(mask_no_age_info)

#dataset["GuessAge"] = dataset.loc[:,"Age"].values.astype('int') == True, "Name"

dataset["GuessAge"] = dataset.loc[ mask_no_age_info == True, "Name"]
print(dataset["GuessAge"])
```


#### (4) 나이 분포도 확인
```python3
df.plot(y='Age',kind='hist')
```


#### (5) 성별에 따른 생존율 비교
```python3

dataset = pd.DataFrame(df)

# Mask
mask_survived = dataset.loc[:,"Survived"] == 1  # 이것의 결과는 boolean이다.
mask_dead = dataset.loc[:,"Survived"] == 0

mask_man = dataset.loc[:,"Sex"] == "male"
mask_woman = dataset.loc[:,"Sex"] == "female"

# Caculate Value
survived_man = (mask_survived & mask_man).astype('int').sum()
survived_woman = (mask_survived & mask_woman).astype('int').sum()
dead_man = (mask_dead & mask_man).astype('int').sum()
dead_woman = (mask_dead & mask_woman).astype('int').sum()

# Make Pandas Data Frame
data = np.array( [[survived_woman, dead_woman],
                  [survived_man, dead_man]])

columns = ["생존", "사망"]
index = ["여자", "남자"]

answer = pd.DataFrame(data, columns = columns, index=index)
answer
```

# 질문 : Pandas DataFrame과 plt에서 column이름과 index이름은 다르게 사용할 수 있나요?

```python3
answer.plot(kind="bar")

plt.columns = ["Passengerld, Alive", "Passengerld, dead"]
plt.index = ["female", "male"]

plt.xlabel("sex")

plt.show()
```


#### (6) 연령에 따른 생존율 비교
```python3
dataset = pd.DataFrame(df)

# Mask
mask_survived = (dataset.loc[:,"Survived"] == 1)  # 이것의 결과는 boolean이다.
mask_dead = (dataset.loc[:,"Survived"] == 0)

# 나이에 대한 Mask 정보
mask_10 = (dataset.loc[:,"Age"] < 10) # 10살 밑
mask_20 = (dataset.loc[:,"Age"] < 20) & (dataset.loc[:,"Age"] >= 10)
mask_30 = (dataset.loc[:,"Age"] < 30) & (dataset.loc[:,"Age"] >= 20)  
mask_40 = (dataset.loc[:,"Age"] < 40) & (dataset.loc[:,"Age"] >= 30)   
mask_50 = (dataset.loc[:,"Age"] < 50) & (dataset.loc[:,"Age"] >= 40)   
mask_60 = (dataset.loc[:,"Age"] < 60) & (dataset.loc[:,"Age"] >= 50)  
mask_70 = (dataset.loc[:,"Age"] < 70) & (dataset.loc[:,"Age"] >= 60)  
mask_80 = (dataset.loc[:,"Age"] < 80) & (dataset.loc[:,"Age"] >= 70)  
mask_90 = (dataset.loc[:,"Age"] < 90) & (dataset.loc[:,"Age"] >= 80)  

# Caculate Value
survived_10 = (mask_survived & mask_10).astype('int').sum()
survived_20 = (mask_survived & mask_20).astype('int').sum()
survived_30 = (mask_survived & mask_30).astype('int').sum()
survived_40 = (mask_survived & mask_40).astype('int').sum()
survived_50 = (mask_survived & mask_50).astype('int').sum()
survived_60 = (mask_survived & mask_60).astype('int').sum()
survived_70 = (mask_survived & mask_70).astype('int').sum()
survived_80 = (mask_survived & mask_80).astype('int').sum()
survived_90 = (mask_survived & mask_90).astype('int').sum()

dead_10 = (mask_dead & mask_10).astype('int').sum()
dead_20 = (mask_dead & mask_20).astype('int').sum()
dead_30 = (mask_dead & mask_30).astype('int').sum()
dead_40 = (mask_dead & mask_40).astype('int').sum()
dead_50 = (mask_dead & mask_50).astype('int').sum()
dead_60 = (mask_dead & mask_60).astype('int').sum()
dead_70 = (mask_dead & mask_70).astype('int').sum()
dead_80 = (mask_dead & mask_80).astype('int').sum()
dead_90 = (mask_dead & mask_90).astype('int').sum()

# Make Pandas Data Frame
data = np.array( [[survived_10, dead_10],
                    [survived_20, dead_20],
                    [survived_30, dead_30],
                    [survived_40, dead_40],
                    [survived_50, dead_50],
                    [survived_60, dead_60],
                    [survived_70, dead_70],
                    [survived_80, dead_80],
                    [survived_90, dead_90]])

columns = ["생존자수", "사망자수"]
index = ["10대 미만", "10대", "20대", "30대", "40대", "50대", "60대", "70대", "80대"]

answer = pd.DataFrame(data, columns = columns, index=index)
answer
```

```python3
answer.columns.name = "생존/사망"
answer.index.name = "성별"

answer.plot(kind="bar")
plt.show()
```

#### (7) 각 Feature 간 상관계수 구하고, 생존에 가장 직결된 요소 파악하기

```python3
dataset = pd.DataFrame(df)
coll = dataset.corr()
sns.heatmap(coll, annot=True, cbar=True)
```