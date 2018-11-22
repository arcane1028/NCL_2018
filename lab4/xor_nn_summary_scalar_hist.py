import tensorflow as tf
import numpy as np
import shutil


tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

with tf.name_scope('Layer1'):
    W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
    b1 = tf.Variable(tf.random_normal([10]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    W1_hist = tf.summary.histogram('weights1',W1)
    b1_hist = tf.summary.histogram('biases1',b1)
    layer1_hist = tf.summary.histogram('layer1',layer1)

with tf.name_scope('Layer2'):
    W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
    b2 = tf.Variable(tf.random_normal([10]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    W2_hist = tf.summary.histogram('weights2',W2)
    b2_hist = tf.summary.histogram('biases2',b2)
    layer2_hist = tf.summary.histogram('layer2',layer2)

with tf.name_scope('Layer3'):
    W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
    W3_hist = tf.summary.histogram('weight3',W3)
    b3_hist = tf.summary.histogram('biases3',b3)
    layer3_hist = tf.summary.histogram('layer3',layer3)

with tf.name_scope('Sigmoid'):
    W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
    W4_hist = tf.summary.histogram('weightss4',W4)
    b4_hist = tf.summary.histogram('biases4',b4)
    hypothesis_hist = tf.summary.histogram('hypothesis',hypothesis)

# cost/loss function
with tf.name_scope('Train'):
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    cost_scalar = tf.summary.scalar('cost',cost)
    
with tf.name_scope('Predict'):
    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    acc_scalar = tf.summary.scalar('accuracy',accuracy)



# Launch graph
with tf.Session() as sess:
    shutil.rmtree("./xor_nn_summary_scalar_hist", ignore_errors=True)
    writer = tf.summary.FileWriter("./xor_nn_summary_scalar_hist")
    summary = tf.summary.merge_all()
    writer.add_graph(sess.graph)
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        s,_ = sess.run([summary,train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(s,global_step=step)
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

writer.close()
