import tensorflow as tf
import numpy as np

#define a layer
def add_layer(inputs, in_size, out_size, activation = None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation is None:
		outputs = Wx_plus_b
	else:
		outputs = activation(Wx_plus_b)
	return outputs

#make up datas
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise -0.5

#set placeholder
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#define layers
hidden_layer = add_layer(xs, 1, 10, activation = tf.nn.relu)
prediction = add_layer(hidden_layer, 10, 1, activation = None)

#loss function
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		sess.run(train, feed_dict={xs:x_data, ys:y_data})
		if i % 50 == 0:
			print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))