import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 1-1
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * w + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, w, b, train],
                                         feed_dict={X: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                    Y: [2.2, 5.2, 6.1, 7.9, 10.5, 11.8, 15, 16, 18.2, 20]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
sess.close()

# 1-2
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [2.2, 5.2, 6.1, 7.9, 10.5, 11.8, 15, 16, 18.2, 20]

W = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
b_val = []
cost_val = []
for wi in np.arange(1, 3, 0.1):
    for bi in np.arange(0, 2, 0.1):
        curr_cost, curr_W, curr_b = sess.run([cost, W, b], feed_dict={W: wi, b: bi})
        W_val.append(curr_W)
        b_val.append(curr_b)
        cost_val.append(curr_cost)

        print("cost {:3.7f} w = {:1.1f} b = {:1.1f}".format(curr_cost, curr_W, curr_b))

mean_cost = 10000
mean_w = 0
mean_b = 0
for ci, wi, bi in zip(cost_val, W_val, b_val):
    if ci < mean_cost:
        mean_w = wi
        mean_b = bi
        mean_cost = ci

print("mean cost {:3.7f} w = {:1.1f} b = {:1.1f}".format(mean_cost, mean_w, mean_b))

# 2
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [2.2, 5.2, 6.1, 7.9, 10.5, 11.8, 15, 16, 18.2, 20]

W = tf.placeholder(tf.float32)
b = tf.constant(0.5, dtype=tf.float32)
hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for wi in np.arange(0,4, 0.01):
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: wi})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()

# 3
tf.set_random_seed(777)

xy = np.loadtxt("data.csv", delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_train, x_test = np.split(x_data, [int(len(x_data) * 0.8)])
y_train, y_test = np.split(y_data, [int(len(y_data) * 0.8)])

print(x_train.shape)
print(y_train.shape)

X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_train, Y: y_train})
    if step % 100 == 0:
        print(step, "Cost :", cost_val, "\nPrediction:\n", hy_val)

cost_val, hy_val = sess.run([cost, hypothesis], feed_dict={X: x_test, Y: y_test})
print("Cost :\n", y_test[0:5], "\nPrediction:\n", hy_val[0:5])
print("mean: ", cost_val)
