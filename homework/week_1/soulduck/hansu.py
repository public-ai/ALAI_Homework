def hansu(x):
    count = 0
    for i in range(1,x+1):
        numbers = list(map(int, list(str(i))))
        if len(numbers)==1 or len(numbers)==2:
            count +=1
        else:
            diff = list(map(lambda ind: numbers[ind] - numbers[ind + 1], range(len(numbers[:-1]))))
            print(numbers ,diff)
            if len(set(diff)) == 1:
                print(diff)
                count +=1
    return count

hansu(1000)
