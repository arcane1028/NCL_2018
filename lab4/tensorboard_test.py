import tensorflow as tf

a = tf.constant(1, name="input_a")
b = tf.constant(2, name="input_b")
c = tf.add(a,b,name='sum_c')

sess = tf.Session()
writer = tf.summary.FileWriter("./board1")
writer.add_graph(sess.graph)

writer.close()