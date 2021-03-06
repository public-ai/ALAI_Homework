```python3
#1번
df = pd.read_csv("./titanic_dataset.csv")

df_isnan = df.isna().sum(axis=0)

df_isnan_ = pd.DataFrame(df_isnan).T

df_miss = pd.DataFrame({"컬럼이름" : ["결측 갯수"]})

df_con = pd.concat([df_miss,df_isnan_],axis =1)

df_con
```

```python3
#2번
people_count=df.Survived.value_counts(dropna=False).sum()

alive=df.Survived.sum(axis=0)

die=people_count - alive

df_2 = pd.DataFrame({"생존유무" : ["사망" , "생존"],
                      "값" : [die, alive]},
                   columns = ["생존유무", "값"])

df_2

```
```python3
#3번
Name_ = df.Name.str.partition(',')[2].str.partition('.')[0]
df["Title"] = Name_
df2 = pd.DataFrame([df.groupby("Title")['Age'].mean()])
df2


```
```python3
#4번
df.Age.plot(kind='hist',legend=True)
```
```python3
#5번

fe_cnt = df.Survived[df.Sex == 'female'].value_counts()

ma_cnt =  df.Survived[df.Sex == 'male'].value_counts().sort_values(ascending=True)

df_cnt = pd.DataFrame([fe_cnt, ma_cnt], index = ["female", "male"] )
df_cnt.columns = ["alive", "dead"] 

df_cnt.index.name = "Sex"
df_cnt.columns.name = "None,survived"
    
print(df_cnt)
df_cnt.plot(kind='bar') #series단위
```
```python3
#6번

df_sam = df
df_sam

df_age = df_sam.sort_values('Age')

count_0 = df_age[df_age['Age']<10].Survived.value_counts()
count_1 = df_age[(df_age['Age']>=10) & (df_age['Age']<20)].Survived.value_counts()
count_2 = df_age[(df_age['Age']>=20) & (df_age['Age']<30)].Survived.value_counts()
count_3 = df_age[(df_age['Age']>=30) & (df_age['Age']<40)].Survived.value_counts()
count_3_ = df_age[(df_age['Age']>=40) & (df_age['Age']<50)].Survived.value_counts()
count_4 = df_age[(df_age['Age']>=50) & (df_age['Age']<60)].Survived.value_counts()
count_5 = df_age[(df_age['Age']>=60) & (df_age['Age']<70)].Survived.value_counts()
count_6 = df_age[(df_age['Age']>=70) & (df_age['Age']<80)].Survived.value_counts()
count_7 = df_age[(df_age['Age']>=80)].Survived.value_counts()

count_val = pd.DataFrame([count_0, count_1, count_2, count_3, count_3_, count_4, count_5, count_6, count_7], index = ["10미만", "10대", "20대", "30대", "40대", "50대", "60대", "70대", "80대"])
count_val.columns.name = ["연령대"]
count_val.columns = ["사망자수", "생존자수"]

count_val.plot(kind='bar')
```
```python3
#7번
import seaborn as sns
df_ = df

df_cor = df_.corr()

sns.heatmap(df_cor, annot=True, cbar=True,)
```

```python3
#8번
df_sam = df

df_cnt = df_sam.groupby("Parch").Survived.value_counts(ascending=False, sort=False, dropna=False)

df_cnt2 = df_sam.groupby("Embarked").Survived.value_counts(ascending=False, sort=False, dropna=False)

df_cnt = np.array(df_cnt)

df_cnt2 = np.array(df_cnt2)

x_tic_val = ["dead","survived","dead","survived","dead",
             "survived","dead","survived","dead","dead","survived","dead"]

fig = plt.figure(figsize= (15,5))

ax = fig.add_subplot(1,2,1, facecolor= 'white')
ax.set_xticks( ticks = np.arange(0,12) )
ax.set_xticklabels( labels =x_tic_val ,rotation = 45)
ax.set_title("Parch-Survived")
ax.set_xlabel("# child or parents")
ax.set_ylabel("# people")
#ax.text(x=0, y=450, s="sdfsdf") # 질문1 : bar의 숫자위치
#질문2 : Q,S등 label의 label..?
color_ = ["#cea2fd","#cea2fd","pink","pink",
          "y","y","#39ad48","#39ad48","xkcd:sky blue","xkcd:sky blue"
          ,"xkcd:light green","xkcd:light green"]

i=0
for col, df_ in zip(color_, df_cnt) :
    bar = ax.bar(x = np.arange(i, i+1),
       height = df_,
       width = 0.8,
       color = [col])
    i += 1

ax = fig.add_subplot(1,2,2, facecolor= 'white')
ax.set_xticks(np.arange(0,6))
ax.set_xticklabels(["survived", "dead", "survived", "dead", "survived", "dead"])
ax.set_xlabel("# child or parents")
ax.set_ylabel("# people")
ax.set_title("Embarked-Survived")
color_2 = ["#cea2fd","#cea2fd", "#39ad48","#39ad48","xkcd:sky blue","xkcd:sky blue"]

i=0
for col2, df_ in zip(color_2, df_cnt2) :
    ax.bar(x = np.arange(i,i+1),
          height = df_,
          width = 0.8,
          color = [col2])
    i+=1
```
```python3
#9번
df_2 = pd.read_csv("./cancer_data.csv")
#----------------------------------Shuffle-------------------------------------
idx_ = np.arange(0,100)
np.random.shuffle(idx_)
df_2["idx"] = idx_
df_2 = df_2.sort_values('idx')
train, test, vali  = df_2[:60], df_2[60:80] , df_2[80:]
#----------------------------------Figure--------------------------------------
fig = plt.figure(figsize =(10,10))
plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
#------------------------------------------------------------------------------
#train_set
#---------------(1,1)------------------
colors = list(map(lambda x : 'r' if x == 1 else 'b', train.label))
ax = fig.add_subplot(3,3,1)
ax.set_title("train")
ax.set_xlabel("Tumor size")
ax.set_ylabel("age")
ax.scatter(x=train.tumor_size, y=train.age, color = colors )
#---------------(1,2)------------------
ax = fig.add_subplot(3,3,2)
train_mean = train.describe().iloc[1,0]
train_std = train.describe().iloc[2,0]
train_min = train.describe().iloc[3,0]
train_max = train.describe().iloc[7,0]
train_describe = np.array([train_mean,train_std,train_min,train_max])
ax.set_title("age")
ax.bar(x=np.arange(0,4),
       height=train_describe,
       color='b')
ax.set_xticks(ticks = np.arange(0,4))
ax.set_xticklabels(labels = ["mean","std","min","max"])
#---------------(1,3)------------------
ax = fig.add_subplot(3,3,3)
train_mean = train.describe().iloc[1,1]
train_std = train.describe().iloc[2,1]
train_min = train.describe().iloc[3,1]
train_max = train.describe().iloc[7,1]
train_describe = np.array([train_mean,train_std,train_min,train_max])
ax.set_title("tumor")
ax.bar(x=np.arange(0,4),
       height=train_describe,
       color='r')
ax.set_xticks(ticks = np.arange(0,4))
ax.set_xticklabels(labels = ["mean","std","min","max"])

#------------------------------------------------------------------------------
#test_set
#---------------(2,1)------------------
colors = list(map(lambda x : 'r' if x == 1 else 'b', test.label))
ax = fig.add_subplot(3,3,4)
ax.set_title("test")
ax.set_xlabel("Tumor size")
ax.set_ylabel("age")
ax.scatter(x=test.tumor_size,
            y=test.age,
          color = colors)
#---------------(2,2)------------------
ax = fig.add_subplot(3,3,5)
test_mean = test.describe().iloc[1,0]
test_std = test.describe().iloc[2,0]
test_min = test.describe().iloc[3,0]
test_max = test.describe().iloc[7,0]
test_describe = np.array([test_mean,test_std,test_min,test_max])
#print(test_describe)
ax.set_title("age")
ax.bar(x=np.arange(0,4),
       height=test_describe,
       color='b')
ax.set_xticks(ticks = np.arange(0,4))
ax.set_xticklabels(labels = ["mean","std","min","max"])
#---------------(2,3)------------------
ax = fig.add_subplot(3,3,6)
test_mean = test.describe().iloc[1,1]
test_std = test.describe().iloc[2,1]
test_min = test.describe().iloc[3,1]
test_max = test.describe().iloc[7,1]
test_describe = np.array([test_mean,test_std,test_min,test_max])
ax.set_title("tumor")
ax.bar(x=np.arange(0,4),
       height=test_describe,
       color='r')
ax.set_xticks(ticks = np.arange(0,4))
ax.set_xticklabels(labels = ["mean","std","min","max"])

#------------------------------------------------------------------------------
#vali_set
#---------------(3,1)------------------
colors = list(map(lambda x : 'r' if x == 1 else 'b', vali.label))
ax = fig.add_subplot(3,3,7)
ax.set_title("vali")
ax.set_xlabel("Tumor size")
ax.set_ylabel("age")
ax.scatter(x=vali.tumor_size,
            y=vali.age,
          color = colors)
#---------------(3,2)------------------
ax = fig.add_subplot(3,3,8)
vali_mean = vali.describe().iloc[1,0]
vali_std = vali.describe().iloc[2,0]
vali_min = vali.describe().iloc[3,0]
vali_max = vali.describe().iloc[7,0]
test_describe = np.array([vali_mean,vali_std,vali_min,vali_max])
#print(vali_describe)
ax.set_title("age")
ax.bar(x=np.arange(0,4),
       height=test_describe,
       color='b')
ax.set_xticks(ticks = np.arange(0,4))
ax.set_xticklabels(labels = ["mean","std","min","max"])
#---------------(3,3)------------------
ax = fig.add_subplot(3,3,9)
vali_mean = vali.describe().iloc[1,1]
vali_std = vali.describe().iloc[2,1]
vali_min = vali.describe().iloc[3,1]
vali_max = vali.describe().iloc[7,1]
vali_describe = np.array([vali_mean,vali_std,vali_min,vali_max])
#print(vali_describe)
ax.set_title("age")
ax.bar(x=np.arange(0,4),
       height=vali_describe,
       color='r')
ax.set_xticks(ticks = np.arange(0,4))
ax.set_xticklabels(labels = ["mean","std","min","max"])
```

