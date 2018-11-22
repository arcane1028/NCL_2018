import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

models = []
num_models = 3
nb_classes = 3
learning_late = 0.0001
training_epochs = 100
batch_size = 100


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3072])
            X_img = tf.reshape(self.X, [-1, 32, 32, 3])
            self.Y = tf.placeholder(tf.int32, [None, 1])

            Y_one_hot = tf.one_hot(self.Y, nb_classes)
            Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

            W1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01))

            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))

            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            L2 = tf.reshape(L2, [-1, 8 * 8 * 64])

            # W3 = tf.get_variable("W3", shape=[8 * 8 * 64, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
            W3 = tf.Variable(tf.random_normal([8 * 8 * 64, nb_classes], stddev=0.01))
            b = tf.Variable(tf.random_normal([nb_classes]))

            self.logits = tf.matmul(L2, W3) + b
            hypothesis = tf.nn.softmax(self.logits)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=Y_one_hot))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(self.cost)
            predict = tf.argmax(hypothesis, 1)
            correct_prediction = tf.equal(predict, tf.argmax(Y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        """
        self.logits = tf.matmul(L2, W3) + b
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(self.cost)

        predictions = tf.equal(tf.arg_max(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
        """

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})


if __name__ == "__main__":
    trainX = np.load('trainX.npy')
    trainY = np.load('trainY.npy')
    testX = np.load('testX.npy')
    testY = np.load('testY.npy')

    print(np.shape(trainX))
    print(np.shape(trainY))
    print(np.shape(testX))
    print(np.shape(testY))

    with tf.Session() as sess:
        m1 = Model(sess, "m1")
        print("learning start")
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0

            total_batch = int(len(trainX) / batch_size)
            for i in range(total_batch):
                batch_xs = trainX[i * batch_size:(i + 1) * batch_size]
                batch_ys = trainY[i * batch_size:(i + 1) * batch_size]
                c, _ = m1.train(batch_xs, batch_ys)
                avg_cost += c / total_batch
                # print("cost", c)
            print(epoch, avg_cost, "acc: ", m1.get_accuracy(testX, testY))
        print("learning end")
        a = m1.get_accuracy(testX, testY)
        print("Accuracy: ", a)

        sess.close()