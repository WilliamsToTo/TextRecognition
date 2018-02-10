from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import data_provider

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=62)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
#  
## Load training and eval data
##mnist = tf.contrib.learn.datasets.load_dataset("mnist")
##train_data = mnist.train.images  # Returns np.array
#trainDataPath = '../train_data_resized/'
#(train_data, train_labels) = data_provider.provide_data(trainDataPath)
#
##print(type(train_data), np.shape(train_data))
##train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
##print(type(train_labels), np.shape(train_labels), train_labels[8])
#  
#evalDataPath = '../test_data_resized/'
#(eval_data, eval_labels) = data_provider.provide_data(evalDataPath)
##eval_data = mnist.test.images  # Returns np.array
##eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
#"/tmp/mnist_convnet_model"
# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
x={"x": train_data},
y=train_labels,
batch_size=1000,
num_epochs=None,
shuffle=True)
mnist_classifier.train(
input_fn=train_input_fn,
steps=20000,
hooks=[logging_hook])

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
x={"x": eval_data},
y=eval_labels,
num_epochs=1,
shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
  
  
  
  
#def main(unused_argv):
#  # Load training and eval data
#  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#  #train_data = mnist.train.images  # Returns np.array
#  trainDataPath = '../train_data_resized/'
#  (train_data, train_labels) = data_provider.provide_data(trainDataPath)
#  #print(type(train_data), np.shape(train_data))
#  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#  #print(type(train_labels), np.shape(train_labels), train_labels[8])
#  
#  evalDataPath = '../test_data_resized/'
#  (eval_data, eval_labels) = data_provider.provide_data(evalDataPath)
#  #eval_data = mnist.test.images  # Returns np.array
#  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#  # Create the Estimator
#  mnist_classifier = tf.estimator.Estimator(
#      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
#
#  # Set up logging for predictions
#  # Log the values in the "Softmax" tensor with label "probabilities"
#  tensors_to_log = {"probabilities": "softmax_tensor"}
#  logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=50)
#
#  # Train the model
#  train_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": train_data},
#      y=train_labels,
#      batch_size=100,
#      num_epochs=None,
#      shuffle=True)
#  mnist_classifier.train(
#      input_fn=train_input_fn,
#      steps=10,
#      hooks=[logging_hook])
#
#  # Evaluate the model and print results
#  eval_input_fn = tf.estimator.inputs.numpdef main(unused_argv):
#  # Load training and eval data
#  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#  #train_data = mnist.train.images  # Returns np.array
#  trainDataPath = '../train_data_resized/'
#  (train_data, train_labels) = data_provider.provide_data(trainDataPath)
#  #print(type(train_data), np.shape(train_data))
#  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#  #print(type(train_labels), np.shape(train_labels), train_labels[8])
#  
#  evalDataPath = '../test_data_resized/'
#  (eval_data, eval_labels) = data_provider.provide_data(evalDataPath)
#  #eval_data = mnist.test.images  # Returns np.array
#  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#  # Create the Estimator
#  mnist_classifier = tf.estimator.Estimator(
#      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
#
#  # Set up logging for predictions
#  # Log the values in the "Softmax" tensor with label "probabilities"
#  tensors_to_log = {"probabilities": "softmax_tensor"}
#  logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=50)
#
#  # Train the model
#  train_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": train_data},
#      y=train_labels,
#      batch_size=100,
#      num_epochs=None,
#      shuffle=True)
#  mnist_classifier.train(
#      input_fn=train_input_fn,
#      steps=10,
#      hooks=[logging_hook])
#
#  # Evaluate the model and print results
#  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": eval_data},
#      y=eval_labels,
#      num_epochs=1,
#      shuffle=False)
#  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#  print(eval_results)
#
#
#if __name__ == "__main__":
#    tf.app.run()y_input_fn(
#      x={"x": eval_data},
#      y=eval_labels,
#      num_epochs=1,
#      shuffle=False)
#  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#  print(eval_results)
#
#
#if __name__ == "__main__":
#    tf.app.run()









#placeholders and input
#classes = ['0', '4', '8', '_b', '_d' , '_f', '_h', '_j',  
#           '_l',  '_n',  '_p',  '_r', '_t',  '_v',  '_x',  '_z', 
#           '1', '5', '9', 'B', 'D', 'F',   'H', 'J', 'L', 'N', 'P',
#           'R', 'T', 'V', 'X' ,'Z', '2','6', '_a', '_c', '_e', '_g', 
#           '_i', '_k', '_m' , '_o', '_q', '_s', '_u', '_w', '_y', '3', 
#           '7', 'A', 'C', 'E', 'G', 'I', 'K', 'M' , 'O' , 'Q' , 'S' ,
#           'U',  'W' , 'Y']
#img_height = 28
#img_width = 28
#num_channel = 1
#num_classes = len(classes)
#inputs = tf.placeholder(tf.float32,[None, img_height, img_width, num_channel], 'inputs')
#targets = tf.placeholder(tf.float32, [None, num_classes], 'targets')
