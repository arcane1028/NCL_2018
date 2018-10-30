import tensorflow as tf
import numpy as np

"""
# 실습 1 linear function
tf.set_random_seed(777)
xy_test = np.loadtxt('magic_test.csv', delimiter=',', dtype=np.float32)
x_test = xy_test[:, 0:-1]
y_test = xy_test[:, [-1]]

xy_train = np.loadtxt('magic_train.csv', delimiter=',', dtype=np.float32)
x_train = xy_train[:, 0:-1]
y_train = xy_train[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 10])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([10, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 2000 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})
    print("\nHypothesis: ", h, "\nCorrect (Y):", c, "\nAccuracy: ", a)

# 실습 2 logistic function

X = tf.placeholder(tf.float32, shape=[None, 10])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([10, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 2000 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})
    print("\nHypothesis: ", h, "\nCorrect (Y):", c, "\nAccuracy: ", a)


"""
# minist

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 10
batch_size = 100
for learning_rate in [0.01, 0.1, 0.5]:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    epoch_val = []
    train_val = []
    validation_val = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += c / total_batch
            print('\nEpoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
            epoch_val.append(epoch)
            train_val.append(accuracy.eval(session=sess, feed_dict={X: mnist.train.images, Y: mnist.train.labels}))
            validation_val.append(
                accuracy.eval(session=sess, feed_dict={X: mnist.validation.images, Y: mnist.validation.labels}))
            print("Train Accuracy: ", train_val[epoch])
            print("Validation Accuracy: ", validation_val[epoch])
            plt.plot(epoch_val, train_val)
            plt.plot(epoch_val, validation_val)


        print("\nTest Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
plt.show()
