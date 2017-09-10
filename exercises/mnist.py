from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/roger/WorkSpace/deeplearning/tensorflow/DeepLearning_Demo/siamese_tf_mnist/MNIST_data", one_hot=True)

print("Training data size:", mnist.train.num_examples)
print(mnist.train.images.shape)

IINPUT_NODE = 784
OUTPUT_NODE = 10


LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, bias1, weights2, bias2):
	if avg_class == None:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + bias1)
		return tf.matmul(layer1, weights2) + bias2

	else:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(bias1))
		return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(bias2)

def train(mnist):
	x = tf.placeholder(tf.float32, [None, IINPUT_NODE], name = "x-input")
	y_ = tf.placeholder(tf.float32,,[None, OUTPUT_NODE], name="y-input")
	weights1 = tf.Variable(
		tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
	bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE], stddev = 0.1))
	bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))


y = inference(x, None,weights1, bias1, weights2, bias2)

global_step = tf.Variable(0, trainable=False)

variable_averages = tf.train.ExponentailMovingAverage(MOVING_AVERAGE_DECAY, global_step)

variable_averages_op = variable_averages.apply(tf.trainable_variables())

average_y = inference(x, variable_averages, weights1, bias1, weights2, bias2)

cross_entropy = tf.nn.sparse_softmax_cross_entopy_with_logits(y, tf.argmax(y_, 1))


cross_entropy_mean = tf.reduce_mean(cross_entropy)

regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

regularization = regularizer(weights2) + regularizer(weights1)

loss = cross_entropy_mean + regularization

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)

train_step = tf.train.GradientDescentOptimizer(learning_rate)\
	.minimize(loss, global_step = global_step)

with tf.control_dependencies([train_step, variable_averages_op]):
	train_op = tf.no_op(name = "train")
cross_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1),)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	tf.initialize_all_variables().run() 
	validate_feed = {x:mnist.validation.images. y_:mnist.validation.labels}
	test_feed={x:mnist.test.images, y}
