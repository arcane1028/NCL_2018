def my_graph(num: int):
    for i in range(num):
        for j in range(0, num - i - 1):
            print(",", end="")
        for k in range(0, i + 1):
            print("*", end="")
        print("")

    for i in range(num - 1):
        for j in range(0, i + 1):
            print(",", end="")
        for k in range(0, num - 1 - i):
            print("*", end="")
        print("")


my_graph(3)
print("")
my_graph(4)