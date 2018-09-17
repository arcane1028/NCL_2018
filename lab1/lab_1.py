a = ['Life', 'is', 'too', 'short', 'you', 'need', 'python']
print(a)

b: str = a[0][1:3]
c: str = a[0][-1] + a[0][0:3].lower()
a.append(b)
a.append(c)

print(a)