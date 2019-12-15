import json

file = open('test.json', 'w')

data = {'a':1, 'b':2}
json.dump(data, file)
file.close()

file = open('test.json', 'r')
data = json.load(file)
print(data)


