import tensorflow as tf
import numpy as np

models = []
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

            # conv layer 1
            W1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01))
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            # result (?, 32, 32, 32)

            # max pooling layer 1
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # result (?, 16, 16, 32)

            # conv layer 2
            W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            # result (?, 16, 16, 64)

            # max pooling layer 2
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # result (?, 8, 8, 64)

            # fully connected layer
            L2 = tf.reshape(L2, [-1, 8 * 8 * 64])
            W3 = tf.Variable(tf.random_normal([8 * 8 * 64, nb_classes], stddev=0.01))
            b = tf.Variable(tf.random_normal([nb_classes]))

            self.logits = tf.matmul(L2, W3) + b
            hypothesis = tf.nn.softmax(self.logits)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=Y_one_hot))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(self.cost)
            predict = tf.argmax(hypothesis, 1)
            correct_prediction = tf.equal(predict, tf.argmax(Y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

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
        print("Learning Start")
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0

            total_batch = int(len(trainX) / batch_size)
            for i in range(total_batch):
                batch_xs = trainX[i * batch_size:(i + 1) * batch_size]
                batch_ys = trainY[i * batch_size:(i + 1) * batch_size]
                c, _ = m1.train(batch_xs, batch_ys)
                avg_cost += c / total_batch
            if epoch % 10 == 0:
                print(epoch, avg_cost, "acc: ", m1.get_accuracy(testX, testY))
        print("Learning End")
        a = m1.get_accuracy(testX, testY)
        print("Accuracy: ", a)

        sess.close()
