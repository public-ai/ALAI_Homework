# 1
> 문제 <br>
두 정수 A와 B를 입력받은 다음, A-B를 출력하는 프로그램을 작성하시오.

```python3
A, B = map(int,input().split(','))
print(A-B)
```
> 예시 <br>
입력   : 출력 <br>
3, 2  : 1


# 2
> 문제 <br>
ACM-ICPC 인터넷 예선, Regional, 그리고 World Finals까지 이미 2회씩 진출해버린 kriii는 미련을 버리지 못하고 왠지 모르게 올 해에도 파주 World Finals 준비 캠프에 참여했다.
대회를 뜰 줄 모르는 지박령 kriii를 위해서 격려의 문구를 출력해주자.

> 입력은 없습니다 <br>

> 출력 예시 <br>
힘내라 친구

```python3
print("힘내라 친구")
```

# 3
> 문제 두 자연수 A와 B가 주어진다. 이때, A+B, A-B, A*B, A/B(몫), A%B(나머지)를 출력하는 프로그램을 작성하시오.


조건
- class 로 만드세요 , class 이름은 Calculator 입니다
- +, - , *, / 은 모두 magic method 로 만드세요.

```python3
class Calculator :
    def __init__(self, value) :
        self.val = value
    def __add__(self, other) :
        return print(self.val + other.val)
    def __sub__(self, other) :
        return print(self.val - other.val)
    def __mul__(self, other) :
        return print(self.val * other.val)
    def __truediv__(self, other) :
        return print(self.val / other.val)
    def __mod__(self, other) :
        return print(self.val % other.val)    

```
```python3
cal1 = Calculator(3)
cal2 = Calculator(5)
```

# 4
> 문제
영어 대소문자와 띄어쓰기만으로 이루어진 문자열이 주어진다. 이 문자열에는 몇 개의 단어가 있을까? 이를 구하는 프로그램을 작성하시오. 단, 한 단어가 여러 번 등장하면 등장한 횟수만큼 모두 세어야 한다.

> 입력 : 출력 <br>
The Curious Case of Benjamin Button : 6 <br>
Mazatneunde Wae Teullyeoyo : 3 <br>
Teullinika Teullyeotzi : 2 <br>

```pythone3
val = input("입력: ")
val = val.split(" ")
print(len(val))
```

# 5
> 문제 <br>
"OOXXOXXOOO"와 같은 OX퀴즈의 결과가 있다.<br>
 O는 문제를 맞은 것이고, X는 문제를 틀린 것이다. <br>
 문제를 맞은 경우 그 문제의 점수는 그 문제까지 연속된 O의 개수가 된다. <br>
 예를 들어, 10번 문제의 점수는 3이 된다.<br>
 "OOXXOXXOOO"의 점수는 1+2+0+0+1+0+0+1+2+3 = 10점이다.<br>
OX퀴즈의 결과가 주어졌을 때, 점수를 구하는 프로그램을 작성하시오.


> 예시 입력 : 출력 <br>
OOXXOXXOOO : 10 <br>
OOXXOOXXOO : 9 <br>
OXOXOXOXOXOXOX : 7 <br>
OOOOOOOOOO : 55 <br>
OOOOXOOOOXOOOOX : 30 <br>

```python3
val = input("입력 : ")
i = 0
totalsum = 0
cnt = 0

while i < len(val) :
    if val[i] == "O" :
        cnt += 1
        totalsum += cnt
    else :
        cnt = 0
    i += 1
    
print(totalsum)
```

# 6
> 문제<br>
 정수 N개로 이루어진 수열 A와 정수 X가 주어진다. 이때, A에서 X보다 작은 수를 모두 출력하는 프로그램을 작성하시오.

> 입력<br>
첫째 줄에 N과 X가 주어진다. (1 ≤ N, X ≤ 10,000)
둘째 줄에 수열 A를 이루는 정수 N개가 주어진다. 주어지는 정수는 모두 1보다 크거나 같고, 10,000보다 작거나 같은 정수이다.


> 출력<br>
X보다 작은 수를 입력받은 순서대로 공백으로 구분해 출력한다. X보다 작은 수는 적어도 하나 존재한다.

> 예시<br>
N: 10  , A : [ 1 10 4 9 2 3 8 5 7 6 ] , X : 5 <br>
출력 : 1 4 2 3 <br>

```python3
n = int(input("n = "))
a = input("A = ")
x = int(input('X = '))

A = []
A = a.split(" ")
print(A)

i = 0
result = []
while i < n :
    if int(A[i]) < x :
        result.append(A[i])
    i += 1
        
print(result)
```

# 7
> 문제
배열을 정렬하는 것은 쉽다. 수가 주어지면, 그 수의 각 자리수를 내림차순으로 정렬해보자.

> 입력
첫째 줄에 정렬하고자하는 수 N이 주어진다. N은 1,000,000,000보다 작거나 같은 자연수이다.

> 출력
첫째 줄에 자리수를 내림차순으로 정렬한 수를 출력한다.


> 예시 <br>
입력 : 출력 <br>
2143 : 4321 <br>

```python3
val = input("입력 : ")
val = sorted(val, reverse=True)
answer = ''.join(val)
print(answer)

```


# 8
> 문제
어떤 양의 정수 X의 자리수가 등차수열을 이룬다면, 그 수를 한수라고 한다. 등차수열은 연속된 두 개의 수의 차이가 일정한 수열을 말한다. N이 주어졌을 때, 1보다 크거나 같고, N보다 작거나 같은 한수의 개수를 출력하는 프로그램을 작성하시오.

> 입력
첫째 줄에 1,000보다 작거나 같은 자연수 N이 주어진다.


> 출력
첫째 줄에 1보다 크거나 같고, N보다 작거나 같은 한수의 개수를 출력한다.


> 입력 : 출력
110 : 99