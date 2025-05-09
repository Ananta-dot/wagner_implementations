arrH = " 4 10  7 10  5  6  3  7  8  1  1  2  8  3  9  4  2  5  6  9"
arrV = " 5  3 10  9  8  1  7  5  7  4  4 10  2  9  6  6  3  1  2  8"

arH = []
arV = []

arrH = arrH.split(" ")

for i in arrH:
    if i != "":
        arH.append(int(i))

arrV = arrV.split(" ")

for i in arrV:
    if i != "":
        arV.append(int(i))

print(arH)
print(arV)

h = []
v = [] 
for i in arH:
    if i != 3 and i < 3:
        h.append(i)
    elif i != 3 and i > 3:
        h.append(i-1)

for i in arV:
    if i != 3 and i < 3:
        v.append(i)
    elif i != 3 and i > 3:
        v.append(i-1)

print(h)
print(v)

finh = []
finv = []

for i in h:
    if i != 8 and i < 8:
        finh.append(i)
    elif i != 8 and i > 8:
        finh.append(i-1)

for i in v:
    if i != 8 and i < 8:
        finv.append(i)
    elif i != 8 and i > 8:
        finv.append(i-1)

print(finh)
print(finv)