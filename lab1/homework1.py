import tensorflow as tf


def get_day(month, day):
    hs = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    q = int(day)
    K = int(18)
    J = int(20)
    m = int(month)
    if m in [1, 2]:
        m = int(12 + int(month))
        K = K - 1

    h = (q + int(((m + 1) * 13) / 5) + K + int(K / 4) + int(J / 4) - 2 * J) % 7
    return hs[h]


month_in = input("month : ")
day_in = input("day : ")
year = 2018

print(get_day(month_in, day_in))

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
sess = tf.Session()

print(sess.run(add6, feed_dict={a: [[1, 2, 3], [2, 3, 4]], b: [[3, 2, 1], [4, 3, 2]]}))
print(sess.run(add6, feed_dict={a: [[1, 5, 6], [3, 4, 5]], b: [[5, 4, 3], [3, 5, 6]]}))

sess.close()
