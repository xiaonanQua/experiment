
x = 0

def ss(x):
    for i in range(100):
        x += 1
        print(x)
    return x

for j in range(100):
    x = ss(x)

print(x)