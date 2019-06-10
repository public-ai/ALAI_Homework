```python
# (1) 각 열 별로 얼마나 결측되어 있는지 확인하기
answer1 = pd.DataFrame(df.isna().sum()).T
print(answer1)
```

```python
# (2) 생존자 수 확인
survived = df["Survived"].sum()
dead = len(df["Survived"]) - df["Survived"].sum() 
answer = [["생존", survived],
         ["사망", dead]]
pd.DataFrame(answer, columns = ["생존유무", "값"])
```

```python
# (3) 연령추론
# 정답을 입력해 주세요!
age = []
for i in range(0,891):
    name_list = list(df.Name)[i].split(' ')
    if 'Mr.' in name_list:
        age.append(df.loc[i]["Age"])
answer = np.array(age)
np.delete(answer, "nan")
을 통해 age의 mean을 각각 title에 따라 구하려고 했으나 for 구문을 사용하는 것 자체가 비효율 적인 것 같다고 생각했습니다.
```

```python
# (4) 나이 분포도 확인
df.plot(y='Age',kind='hist')
```

```python
# (5) 성별에 따른 생존율 비교
# 정답을 입력해주세요 
a = np.array([
    [233, 81],
    [109, 468]
])
df1 = pd.DataFrame(a)
df1.columns = ["Passengerld, Alive", "Passengerld, dead"]
df1.index = ["female", "male"]
df1.plot(kind = "bar")
plt.xlabel("sex")
plt.show()
```

```python
# (6) 연령에 따른 생존율 비교
# 정답을 입력해 주세요!
a = np.array([
    [38, 24],
    [41, 61],
    [77, 143],
    [73, 94],
    [34, 55],
    [20, 28],
    [6, 13],
    [0, 6],
    [1, 0]
])
df2 = pd.DataFrame(a)
df2.columns = ["Passengerld, Alive", "Passengerld, dead"]
df2.index = ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"]
df2.plot(kind = "bar")
plt.xlabel("AgeGroup")

plt.show()
```

```python
# (7) 각 Feature간 상관계수 구하고, 생존에 가장 직결된 요소 파악하기
# 정답을 입력해 주세요
import seaborn as sns
answer = df.corr()
sns.heatmap(answer, annot=True, cbar=True)
```