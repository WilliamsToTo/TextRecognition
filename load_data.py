# Load training and eval data
#mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#train_data = mnist.train.images  # Returns np.array
trainDataPath = '../train_data_resized/'
(train_data, train_labels) = data_provider.provide_data(trainDataPath)

#print(type(train_data), np.shape(train_data))
#train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#print(type(train_labels), np.shape(train_labels), train_labels[8])
  
evalDataPath = '../test_data_resized/'
(eval_data, eval_labels) = data_provider.provide_data(evalDataPath)
#eval_data = mnist.test.images  # Returns np.array
#eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

