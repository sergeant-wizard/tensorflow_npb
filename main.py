import tensorflow as tf
import numpy

SCORE_SIZE = 33
HIDDEN_UNIT_SIZE = 1024
HIDDEN_UNIT_SIZE = 1024

raw_input = numpy.loadtxt(open("input.csv"), delimiter=",")
[salary, score]  = numpy.hsplit(raw_input, [1])

[salary_train, salary_test] = numpy.vsplit(salary, [50])
[score_train, score_test] = numpy.vsplit(score, [50])

def inference(score_placeholder):
  with tf.name_scope('hidden1') as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([SCORE_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(score_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope('hidden2') as scope:
    hidden2_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="hidden2_weight")
    hidden2_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="hidden2_bias")
    hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, hidden2_weight) + hidden2_bias)
  with tf.name_scope('output') as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, 1], stddev=0.1), name="output_weight")
    output_bias = tf.Variable(tf.constant(0.1, shape=[1]), name="output_bias")
    output = tf.matmul(hidden2_output, output_weight) + output_bias
  return tf.nn.l2_normalize(output, 0)

def loss(output, salary_placeholder):
  with tf.name_scope('loss') as scope:
    loss = tf.nn.l2_loss(output - tf.nn.l2_normalize(salary_placeholder, 0))
    tf.scalar_summary("loss", loss)
  return loss

def training(loss):
  with tf.name_scope('training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step


with tf.Graph().as_default():
  salary_placeholder = tf.placeholder("float", [None, 1], name="salary_placeholder")
  score_placeholder = tf.placeholder("float", [None, SCORE_SIZE], name="score_placeholder")

  train_feed_dict={salary_placeholder: salary_train, score_placeholder: score_train}
  test_feed_dict={salary_placeholder: salary_test, score_placeholder: score_test}

  output = inference(score_placeholder)
  loss = loss(output, salary_placeholder)
  training_op = training(loss)

  summary_op = tf.merge_all_summaries()

  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
    sess.run(init)

    for step in range(100):
      sess.run(training_op, feed_dict=train_feed_dict)
      if step % 10 == 0:
        print "train loss:"
        print sess.run(loss, feed_dict=train_feed_dict)
        print "test loss:"
        print sess.run(loss, feed_dict=test_feed_dict)
        summary_str = sess.run(summary_op, feed_dict=train_feed_dict)
        summary_writer.add_summary(summary_str, step)
