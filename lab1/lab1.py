import tensorflow as tf
import datetime as dt

"""

a = tf.constant('hello tensorflow')
sess = tf.Session()
print(sess.run(a))

"""
tf.ConfigProto(inter_op_parallelism_threads=8)

a = ['Life', 'is', 'too', 'short', 'you', 'need', 'python']
print(a)

b: str = a[0][1:3]
c: str = a[0][-1] + a[0][0:3].lower()
a.append(b)
a.append(c)

print(a)

student = {"kim": {"kor": 70,
                   "eng": 65,
                   "math": 43},
           "choi": {"kor": 99,
                    "eng": 87,
                    "math": 65}
           }
print(student["kim"]["kor"])


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
my_graph(4)

input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)

a1 = input1 + input2
a2 = input1 * input2
b1 = a1 + a2

sess = tf.Session()
print(sess.run(b1, feed_dict={input1: 5, input2: 3}))

month = input("month : ")
day = input("day : ")
year = 2018

weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

week = weekDays[
    dt.date(year=year,
            month=int(month),
            day=int(day)).weekday()]

print(week)

a = tf.constant([1, 2, 3], tf.int32)
b = tf.constant([3, 2, 1], tf.int32)

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)

mul1 = tf.multiply(a, b)
mul2 = tf.multiply(a, b)
mul3 = tf.multiply(a, b)

add1 = tf.add(mul2, mul3)
add2 = tf.add(mul1, mul3)
add3 = tf.add(mul1, mul2)

add4 = tf.add(add1, add2)
add5 = tf.add(add2, add3)

add6 = tf.add(add4, add5)

print(sess.run(add6, feed_dict={a: [[1, 2, 3], [2, 3, 4]], b: [[3, 2, 1], [4, 3, 2]]}))

sess.close()
