from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import shutil

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
learning_rate = 0.5
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('Layer1'):
    W1 = tf.Variable(tf.random_normal([784, 256]), name='weight1')
    b1 = tf.Variable(tf.random_normal([256]), name='bias1')
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    W1_hist = tf.summary.histogram('weights1', W1)
    b1_hist = tf.summary.histogram('biases1', b1)
    L1_hist = tf.summary.histogram('L1', L1)

with tf.name_scope('Layer2'):
    W2 = tf.Variable(tf.random_normal([256, 256]), name='weight2')
    b2 = tf.Variable(tf.random_normal([256]), name='bias2')
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    W2_hist = tf.summary.histogram('weights2', W2)
    b2_hist = tf.summary.histogram('biases2', b2)
    L2_hist = tf.summary.histogram('L2', L2)

with tf.name_scope('Hypothesis'):
    W3 = tf.Variable(tf.random_normal([256, 10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    hypothesis = tf.matmul(L2, W3) + b3
    W3_hist = tf.summary.histogram('weights3', W3)
    b3_hist = tf.summary.histogram('biases3', b3)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)

with tf.name_scope('Train'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    cost_scalar = tf.summary.scalar('cost', cost)

with tf.name_scope('Predict'):
    prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    acc_scalar = tf.summary.scalar('accuracy', accuracy)

training_epochs = 10
batch_size = 100

with tf.Session() as sess:
    shutil.rmtree("./homework1", ignore_errors=True)
    writer = tf.summary.FileWriter("./homework1")
    summary = tf.summary.merge_all()
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            s, c, _ = sess.run([summary, cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
            writer.add_summary(s, global_step=epoch)

        print('\nEpoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Learning Finished!")
    sess.close()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("\nTest Accuracy: ", accuracy_test.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

writer.close()
