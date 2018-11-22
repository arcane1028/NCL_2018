import tensorflow as tf
import numpy as np

# 실습 2-1
tf.set_random_seed(777)

nb_classes = 7  # 0 ~ 6
learning_late = 0.01
steps = 20001

xy = np.loadtxt("image.csv", delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# 데이터 나눔
x_train, x_test = np.split(x_data, [int(len(x_data) * 0.8)])
y_train, y_test = np.split(y_data, [int(len(y_data) * 0.8)])

print(xy.shape)

X = tf.placeholder(tf.float32, shape=[None, 19])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W1 = tf.Variable(tf.random_normal([19, 15]), name='weight1')
b1 = tf.Variable(tf.random_normal([15]), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([15, 11]), name='weight2')
b2 = tf.Variable(tf.random_normal([11]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([11, 7]), name='weight3')
b3 = tf.Variable(tf.random_normal([7]), name='bias3')
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.Variable(tf.random_normal([7, nb_classes]), name='weight4')
b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias4')
logits = tf.matmul(L3, W4) + b4
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
train = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(cost)
predict = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(predict, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):

        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step%2000 == 0:
            print(step, cost_val)

    _, _, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})
    print("\nAccuracy: ", a)
    sess.close()

# xavier
W1 = tf.get_variable('weight1', shape=[19, 15], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([15]), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable('weight2', shape=[15, 11], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([11]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable('weight3', shape=[11, 7], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([7]), name='bias3')
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable('weight4', shape=[7, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias4')
logits = tf.matmul(L3, W4) + b4
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
train = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(cost)
predict = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(predict, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step%2000 == 0:
            print(step, cost_val)

    _, _, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})
    print("\nAccuracy: ", a)
    sess.close()
