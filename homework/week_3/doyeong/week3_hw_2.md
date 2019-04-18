```python
# Parch
a = df[['Parch', 'Survived']]
b = a.groupby(['Parch'], as_index=True)
c = a[a.Survived > 0]
d = c.groupby(['Parch'], as_index=True)
alive = d.count()
e = a[a.Survived == 0]
f = e.groupby(['Parch'], as_index=True)
dead = f.count()
answer = pd.concat([alive,dead],axis=1).T
answer.index = ["Survived", "Dead"]
answer.plot(kind = "bar")
plt.title("Parch - Survived")
plt.xlabel("# child or parents")
plt.ylabel("# people")
plt.show()
```

```python
# Embarked
a = df[['Embarked', 'Survived']]
b = a.groupby(['Embarked'], as_index=True)
c = a[a.Survived > 0]
d = c.groupby(['Embarked'], as_index=True)
alive = d.count()
e = a[a.Survived == 0]
f = e.groupby(['Embarked'], as_index=True)
dead = f.count()
answer = pd.concat([alive,dead],axis=1).T
answer.index = ["Survived", "Dead"]
answer.plot(kind = "bar")
plt.title("Embarked - Survived")
plt.xlabel("# child or parents")
plt.ylabel("# people")
plt.show()
```

