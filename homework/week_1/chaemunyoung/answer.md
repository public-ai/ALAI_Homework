### 1번 문제
~~~
A = int(input())
B = int(input())
print(A - B)
~~~
### 2번
~~~
print("힘내라 친구")
~~~
### 3번
~~~
class Calculaor:  
    def __init__(self, number = 0):
      self.number = number
    def __add__(self, other):
      return self.number + other.number
    def __sub__(self, other):
      return self.number - other.number
    def __mul__(self, other):
      return self.number * othter.number
    def __truediv__(self, other):
      return self.number / other.number
~~~
### 4번
~~~
input = "The Curious Case of Benjamin Button"
len(input.split(" "))

input = "Mazatneunde Wae Teullyeoyo"
len(input.split(" "))

input = "Teullinika Teullyeotzi"
len(input.split(" "))
~~~
### 5번
~~~
input = "OOXXOXXOOO"
lst = input.split("X")
result = 0
while len(lst):
    for count in range(len(lst.pop()) + 1):
        result += count
print(result)
~~~
### 6번
~~~
N = 10
A = [1, 10, 4, 9, 2, 3, 8, 5, 7, 6]
X = 5
for value in A:
    if value < X:
        print(value)
~~~
### 7번
~~~
N = 2143
N = str(N)
N = sorted(N, reverse = True)
for value in N:
    print(value, end ="")
~~~
### 8번
~~~
N = 110

def checkHansu(number):
    if number < 100:
        return True
    else:
        num1 = number / 100
        num2 = (number / 10) % 10
        num3 = number % 10
        if num3 - num2 == num2 - num1 :
            return True
        else:
            return False
result = list(filter(checkHansu, [x for x in range(1, N + 1)]))
print(len(result))
~~~
