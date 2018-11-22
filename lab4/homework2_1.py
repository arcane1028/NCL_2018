import tensorflow as tf
import numpy as np

# 실습 2-1
tf.set_random_seed(777)

nb_classes = 7 # 0 ~ 6
learning_late = 0.05
steps = 10001

xy = np.loadtxt("image.csv", delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# 데이터 나눔
x_train, x_test = np.split(x_data, [int(len(x_data) * 0.8)])
y_train, y_test = np.split(y_data, [int(len(y_data) * 0.8)])
print(x_train.shape)

X = tf.placeholder(tf.float32, shape=[None, 19])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# hidden 1층
W1 = tf.Variable(tf.random_normal([19, 15]), name='weight1')
b1 = tf.Variable(tf.random_normal([15]), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# hidden 2층
W2 = tf.Variable(tf.random_normal([15, 11]), name='weight2')
b2 = tf.Variable(tf.random_normal([11]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
# hidden 3층
W3 = tf.Variable(tf.random_normal([11, 7]), name='weight3')
b3 = tf.Variable(tf.random_normal([7]), name='bias3')
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
# 출력층
W4 = tf.Variable(tf.random_normal([7, nb_classes]), name='weight4')
b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias4')
logits = tf.matmul(L3, W4) + b4
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_late).minimize(cost)
predict = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(predict, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 2000 == 0:
            print(step, cost_val)

    _, _, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})
    print("\nAccuracy: ", a)
    sess.close()

# hidden 1층
W1 = tf.Variable(tf.random_normal([19, 11]), name='weight2-1')
b1 = tf.Variable(tf.random_normal([11]), name='bias2-1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# 출력층
W2 = tf.Variable(tf.random_normal([11, nb_classes]), name='weight2-2')
b2 = tf.Variable(tf.random_normal([nb_classes]), name='bias2-2')
logits = tf.matmul(L1, W2) + b2
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_late).minimize(cost)
predict = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(predict, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 2000 == 0:
            print(step, cost_val)

    _, _, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})
    print("\nAccuracy: ", a)
    sess.close()

