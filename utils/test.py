with open('result.txt', 'w+') as file:
    file.write('11111\n')

for i in range(10):
    with open('result.txt', 'a+') as file:
        file.write('2222\n')


j =1

while True:
    if j/10 == 1:
        print(1)
        j=0
    else:
        print(2)
        j = j + 1
