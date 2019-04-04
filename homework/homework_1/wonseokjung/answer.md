## 1
```python3
i = input()
A, B = map(int, i.split(', '))

print(A-B)
```

## 2
```python3
print('힘내라 친구')
```

## 3
```python3
class Calculator:
    def __init__(self, x):
        self.x = x
    def __add__(self, other):
        return self.x + other.x

    def __sub__(self, other):
        return self.x - other.x

    def __mul__(self, other):
        return self.x * other.x

    def __truediv__(self, other):
        return self.x / other.x

    def __mod__(self, other):
        return self.x % other.x
```

## 4
```python3
s = input()
len(s.split())
```

## 5
```python3
l = list(input())

result = 0
local_result = 0
for i in l:
    if i == 'O':
        local_result += 1
    elif i == 'X':
        local_result = 0
                
    result += local_result

print(result)
```

## 6
```python3
N, X = map(int, input().split())
l = list(map(int, input().split()))

print(list(filter(lambda x : x < X, l)))
```

## 7
```python3
l = list(map(int, input().split()))
print(sorted(l, reverse=True))
```

## 8
```python3
```
