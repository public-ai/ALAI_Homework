1.
(1)
<pre> <code>
missing_value = pd.DataFrame(df.PassengerId.count() - df.count())
missing_value_df = missing_value.T
missing_value_df.index = ['결측 갯수']
</code> </pre>


(2)
<pre> <code>
survived_group = pd.DataFrame(df.groupby('Survived')["PassengerId"].count())
survived_group.index = ['사망', '생존']
survived_group.columns = ['값']
</code> </pre>


(3)
<pre> <code>
default_titles = ['Mr', 'Mrs', 'Ms', 'Miss', 'Master', 'Maid', 'Madam', 'Don', 'Rev', 'Dr', 'Capt']
name_titles = []
for names in df["Name"]:
    names = names.split(',')[1].split('.')[0]
    names = names[1:]
    if names in default_titles:
           name_titles.append(names)
    else:
        name_titles.append("없음")

name_title_s = pd.Series(name_titles)
df_title_age = pd.DataFrame([name_title_s, df.Age])
df_title_age.index = ['name_title', 'Age']

column_ = []
values = []
for class_name, class_df in df_title_age.T.groupby('name_title'):
    column_.append(class_name)
    values.append(class_df["Age"].mean())

df_name_title = pd.DataFrame(values)
df_name_title = df_name_title.T
df_name_title.columns = column_
df_name_title.index = ["평균 연령"]
df_name_title = df_name_title.sort_index(axis=1)
df_name_title
</code> </pre>


(4)
<pre><code>
df.plot(kind='hist', y='Age')
</code></pre>


(5)
<pre><code>
# 정답을 입력해주세요 
alive = [0, 0]
dead = [0, 0]

for survived_value, survived_df in df.groupby('Survived')["Sex"]:
    for survived_one in survived_df:
        if survived_value == 1:
            if survived_one == "female":
                alive[0] += 1
            else:
                alive[1] += 1
        else:
            if survived_one == "female":
                dead[0] += 1
            else:
                dead[1] += 1

xs = ["female", 'male']
plt.bar(x=np.arange(0,2)-0.1, height=alive, width=0.2)
plt.bar(x=np.arange(0,2)+0.1, height=dead, width=0.2)
plt.xticks(ticks=np.arange(0,2), labels=xs)
plt.xlabel("Sex")
plt.legend(["(PassengerId, Alive)", "(PassengerId, dead)"])
plt.show()
</code></pre>


(6)
<pre><code>
import math

df_ageline = pd.DataFrame()
df_ageline["Age"] = df.Age
df_ageline["Survived"] = df.Survived
df_ageline = df_ageline.fillna(-1)
ageline = ['10대미만', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대']
column = []
for age in df_ageline["Age"]:
    if age == -1:
        column.append(-1)
        continue
    age = math.floor(age)
    if age < 10:
        column.append(0)
    elif age < 20:
        column.append(1)
    elif age < 30:
        column.append(2)
    elif age < 40:
        column.append(3)
    elif age < 50:
        column.append(4)
    elif age < 60:
        column.append(5)
    elif age < 70:
        column.append(6)
    elif age < 80:
        column.append(7)
    elif age < 90:
        column.append(8)    
df_ageline["연령대"] = column
del df_ageline["Age"]
sum_age = df_ageline.groupby("연령대").sum()
count_age = df_ageline.groupby("연령대").count()
df_survived_age = pd.merge(sum_age, count_age - sum_age, on='연령대', how='inner')
df_survived_age.columns = ["생존자수", "사망자수"]

df_survived_age = df_survived_age.drop(index=-1, axis=1)
df_survived_age.index = ageline

fig = plt.figure(figsize=(8, 4))

plt.bar(x=np.arange(0.0,9.0)-0.1, height = df_survived_age["생존자수"], width=0.2)
plt.bar(x=np.arange(0.0,9.0)+0.1, height = df_survived_age["사망자수"], width=0.2)
#df_survived_age.plot(kind='bar', x = np.arange(0, 9, 9))
plt.legend(["(PessengerId, Survived)", "(PassengerId, dead)"])
plt.xlabel('AgeGroup')
plt.xticks(np.arange(0.0,9.0), labels=["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8,0"])
plt.show()
</code></pre>


(7)
<pre><code>
import seaborn as sns

pierce = df.corr()

sns.heatmap(pierce, annot=True, cbar=True)
</code></pre>



2.
<pre><code>
%matplotlib inline  
import matplotlib.pyplot as plt  

import pandas as pd
import numpy as np

!wget https://s3.ap-northeast-2.amazonaws.com/pai-datasets/alai-deeplearning/titanic_dataset.csv
df = pd.read_csv("./titanic_dataset.csv")

parch_alive = df.groupby("Parch")["Survived"].sum()
parch_dead = df.groupby("Parch")["Survived"].count() - parch_alive

embark_alive = df.groupby("Embarked")["Survived"].sum()
embark_dead = df.groupby("Embarked")["Survived"].count() - embark_alive

parch_survived = pd.DataFrame()
parch_survived["alive"] = parch_alive
parch_survived["dead"] = parch_dead
parch_survived = parch_survived.T

embark_survived = pd.DataFrame()
embark_survived["alive"] = embark_alive
embark_survived["dead"] = embark_dead
embark_survived = embark_survived.T
column_list = ["Q", "S", "C"]
embark_survived = embark_survived[column_list]

fig = plt.figure(figsize=(28, 8))
ax = fig.add_subplot(1, 2, 1)
ax.bar(height = parch_survived[0], x = np.linspace(0,3,2), width=2.5, color='C0')
ax.bar(height = parch_survived[1], x = np.linspace(6,9,2), width=2.5, color='C1')
ax.bar(height = parch_survived[2], x = np.linspace(12,15,2), width=2.5, color='C2')
ax.bar(height = parch_survived[3], x = np.linspace(18,21,2), width=2.5, color='C3')
ax.bar(height = parch_survived[4], x = np.linspace(24,27,2), width=2.5, color='C4')
ax.bar(height = parch_survived[5], x = np.linspace(30,33,2), width=2.5, color='C5')
ax.bar(height = parch_survived[6], x = np.linspace(36,39,2), width=2.5, color='C6')
ax.set_xticks(np.linspace(0,39,14))
ax.set_xticklabels(["survived 0", "dead 0", "survived 1", "dead 1", "survived 2", "dead 2", "survived 3", "dead 3", "survived 4", "dead 4", "survived 5", "dead 5", "survived 6", "dead 6"])
ax.set_xlabel("# child or parents")
ax.set_ylabel("# people")
ax.set_title("Parch - Survived")
icount = 0
for x in parch_survived:
    ax.text(icount, parch_survived[x][0], str(parch_survived[x][0]))
    icount += 3
    ax.text(icount, parch_survived[x][1], str(parch_survived[x][1]))
    icount += 3


ax = fig.add_subplot(1, 2, 2)
ax.bar(height = embark_survived["Q"], x = np.linspace(0,3,2), width=2.5, color='C0')
ax.bar(height = embark_survived["S"], x = np.linspace(6,9,2), width=2.5, color='C1')
ax.bar(height = embark_survived["C"], x = np.linspace(12,15,2), width=2.5, color='C2')
ax.set_xticks(np.linspace(0,15,6))
ax.set_xticklabels(["survived Q", "dead Q", "survived S", "dead S", "survived C", "dead C"])
ax.set_xlabel("# child or parents")
ax.set_ylabel("# people")
ax.set_title("Embarked - Survived")
icount = 0
for x in embark_survived:
    ax.text(icount, embark_survived[x][0], str(embark_survived[x][0]))
    icount += 3
    ax.text(icount, embark_survived[x][1], str(embark_survived[x][1]))
    icount += 3

plt.show()
</code></pre>



3. permutation으로 섞은 데이터
<pre><code>
%matplotlib inline  
import matplotlib.pyplot as plt  

import pandas as pd
import numpy as np

df = pd.read_csv("./cancer_data.csv")

#셔플
def split_train_test(data, test_ratio, validation_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    validation_size = int(len(data) * validation_ratio)
    test_indices = shuffled_indices[:test_set_size]
    validation_indices = shuffled_indices[test_set_size:validation_size]
    train_indices = shuffled_indices[validation_size:]
    return data.iloc[train_indices], data.iloc[validation_indices], data.iloc[test_indices]

train_set, validation_set, test_set = split_train_test(df, 0.2, 0.4)

xs = ["mean", "std", "min", "max"]

fig = plt.figure(figsize=(12, 12))
ax = plt.subplot(3,3,1)
colors = ['r' if la==1 else 'b' for la in train_set.label]
plt.scatter(x=train_set.tumor_size, y=train_set.age, color=colors)
ax.set_title("train")
ax.legend(["Train data"])

ax = plt.subplot(3,3,2)
ax.set_title("age")
age_df = pd.DataFrame()
age_df["mean"] = [train_set.age.mean()]
age_df["std"] = [train_set.age.std()]
age_df["min"] = [train_set.age.min()]
age_df["max"] = [train_set.age.max()]
plt.bar(x=xs, height=np.array(age_df).reshape(-1), color='b')

ax = plt.subplot(3,3,3)
ax.set_title("tumor")
tumor_df = pd.DataFrame()
tumor_df["mean"] = [train_set.tumor_size.mean()]
tumor_df["std"] = [train_set.tumor_size.std()]
tumor_df["min"] = [train_set.tumor_size.min()]
tumor_df["max"] = [train_set.tumor_size.max()]
plt.bar(x=xs, height=np.array(tumor_df).reshape(-1), color='r')

ax = plt.subplot(3,3,4)
colors = ['r' if la==1 else 'b' for la in test_set.label]
plt.scatter(x=test_set.tumor_size, y=test_set.age, color=colors, marker='+')
ax.set_title("test")
ax.legend(["Test data"])

ax = plt.subplot(3,3,5)
age_df = pd.DataFrame()
age_df["mean"] = [test_set.age.mean()]
age_df["std"] = [test_set.age.std()]
age_df["min"] = [test_set.age.min()]
age_df["max"] = [test_set.age.max()]
plt.bar(x=xs, height=np.array(age_df).reshape(-1), color='b')

ax = plt.subplot(3,3,6)
tumor_df = pd.DataFrame()
tumor_df["mean"] = [test_set.tumor_size.mean()]
tumor_df["std"] = [test_set.tumor_size.std()]
tumor_df["min"] = [test_set.tumor_size.min()]
tumor_df["max"] = [test_set.tumor_size.max()]
plt.bar(x=xs, height=np.array(tumor_df).reshape(-1), color='r')

#Validation
ax = plt.subplot(3,3,7)
colors = ['r' if la==1 else 'b' for la in validation_set.label]
plt.scatter(x=validation_set.tumor_size, y=validation_set.age, color=colors, marker='*')
ax.set_title("Validation")
ax.legend(["Validation data"])

ax = plt.subplot(3,3,8)
age_df = pd.DataFrame()
age_df["mean"] = [validation_set.age.mean()]
age_df["std"] = [validation_set.age.std()]
age_df["min"] = [validation_set.age.min()]
age_df["max"] = [validation_set.age.max()]
plt.bar(x=xs, height=np.array(age_df).reshape(-1), color='b')

ax = plt.subplot(3,3,9)
tumor_df = pd.DataFrame()
tumor_df["mean"] = [validation_set.tumor_size.mean()]
tumor_df["std"] = [validation_set.tumor_size.std()]
tumor_df["min"] = [validation_set.tumor_size.min()]
tumor_df["max"] = [validation_set.tumor_size.max()]
plt.bar(x=xs, height=np.array(tumor_df).reshape(-1), color='r')

plt.show()
</code></pre>


3. 섞지 않은 데이터, validation data 구현x
<pre><code>
%matplotlib inline  
import matplotlib.pyplot as plt  

import pandas as pd
import numpy as np

df = pd.read_csv("./cancer_data.csv")

#not 셔플
def split_train_test_noshuffle(data, test_ratio, validation_ratio):
    indices = np.arange(len(data))
    test_set_size = int(len(data) * test_ratio)
    validation_size = int(len(data) * validation_ratio)
    test_indices = indices[:test_set_size]
    validation_indices = indices[test_set_size:validation_size]
    train_indices = indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[validation_indices], data.iloc[test_indices]

train_set, validation_set, test_set = split_train_test_noshuffle(df, 0.2, 0.4)

xs = ["mean", "std", "min", "max"]

fig = plt.figure(figsize=(12, 12))
ax = plt.subplot(3,3,1)
colors = ['r' if la==1 else 'b' for la in train_set.label]
plt.scatter(x=train_set.tumor_size, y=train_set.age, color=colors)
ax.set_title("train")
ax.legend(["Train data"])

ax = plt.subplot(3,3,2)
ax.set_title("age")
age_df = pd.DataFrame()
age_df["mean"] = [train_set.age.mean()]
age_df["std"] = [train_set.age.std()]
age_df["min"] = [train_set.age.min()]
age_df["max"] = [train_set.age.max()]
plt.bar(x=xs, height=np.array(age_df).reshape(-1), color='b')

ax = plt.subplot(3,3,3)
ax.set_title("tumor")
tumor_df = pd.DataFrame()
tumor_df["mean"] = [train_set.tumor_size.mean()]
tumor_df["std"] = [train_set.tumor_size.std()]
tumor_df["min"] = [train_set.tumor_size.min()]
tumor_df["max"] = [train_set.tumor_size.max()]
plt.bar(x=xs, height=np.array(tumor_df).reshape(-1), color='r')

ax = plt.subplot(3,3,4)
colors = ['r' if la==1 else 'b' for la in test_set.label]
plt.scatter(x=test_set.tumor_size, y=test_set.age, color=colors, marker='+')
ax.set_title("test")
ax.legend(["Test data"])

ax = plt.subplot(3,3,5)
age_df = pd.DataFrame()
age_df["mean"] = [test_set.age.mean()]
age_df["std"] = [test_set.age.std()]
age_df["min"] = [test_set.age.min()]
age_df["max"] = [test_set.age.max()]
plt.bar(x=xs, height=np.array(age_df).reshape(-1), color='b')

ax = plt.subplot(3,3,6)
tumor_df = pd.DataFrame()
tumor_df["mean"] = [test_set.tumor_size.mean()]
tumor_df["std"] = [test_set.tumor_size.std()]
tumor_df["min"] = [test_set.tumor_size.min()]
tumor_df["max"] = [test_set.tumor_size.max()]
plt.bar(x=xs, height=np.array(tumor_df).reshape(-1), color='r')

#Validation
ax = plt.subplot(3,3,7)
colors = ['r' if la==1 else 'b' for la in validation_set.label]
plt.scatter(x=validation_set.tumor_size, y=validation_set.age, color=colors, marker='*')
ax.set_title("Validation")
ax.legend(["Validation data"])

ax = plt.subplot(3,3,8)
age_df = pd.DataFrame()
age_df["mean"] = [validation_set.age.mean()]
age_df["std"] = [validation_set.age.std()]
age_df["min"] = [validation_set.age.min()]
age_df["max"] = [validation_set.age.max()]
plt.bar(x=xs, height=np.array(age_df).reshape(-1), color='b')

ax = plt.subplot(3,3,9)
tumor_df = pd.DataFrame()
tumor_df["mean"] = [validation_set.tumor_size.mean()]
tumor_df["std"] = [validation_set.tumor_size.std()]
tumor_df["min"] = [validation_set.tumor_size.min()]
tumor_df["max"] = [validation_set.tumor_size.max()]
plt.bar(x=xs, height=np.array(tumor_df).reshape(-1), color='r')

plt.show()
</code></pre>
