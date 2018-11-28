import tensorflow as tf
import numpy as np

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

            W1 = tf.get_variable("W1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)

            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            W2 = tf.get_variable("W2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)

            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            L2 = tf.reshape(L2, [-1, 8 * 8 * 64])
            W3 = tf.get_variable("W3", shape=[8 * 8 * 64, nb_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[nb_classes], initializer=tf.contrib.layers.xavier_initializer())

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
        models = []
        for m in range(num_models):
            models.append(Model(sess, "model" + str(m)))

        print("Learning Start")
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost_list = np.zeros(len(models))

            total_batch = int(len(trainX) / batch_size)
            for i in range(total_batch):
                batch_xs = trainX[i * batch_size:(i + 1) * batch_size]
                batch_ys = trainY[i * batch_size:(i + 1) * batch_size]

                for m_idx, m in enumerate(models):
                    c, _ = m.train(batch_xs, batch_ys)
                    avg_cost_list[m_idx] += c / total_batch
            if epoch % 10 == 0 :
                acc_list = np.zeros(num_models)
                for m_idx, m in enumerate(models):
                    a = m.get_accuracy(testX, testY)
                    acc_list[m_idx] = a

                print(epoch, avg_cost_list, acc_list)

        print("Learning End")

        test_size = len(testY)
        predictions = np.zeros(test_size * nb_classes).reshape(test_size, nb_classes)
        for m_idx, m in enumerate(models):
            print(m_idx, m.get_accuracy(testX, testY))
            p = m.predict(testX)
            predictions += p

        Y_one_hot = tf.one_hot(testY, nb_classes)
        Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
        ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y_one_hot, 1))
        ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
        print("Ensemble Accuracy: ", sess.run(ensemble_accuracy))
        sess.close()
