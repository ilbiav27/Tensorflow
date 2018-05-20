import tensorflow as tf
import os
import numpy as np

from datetime import datetime
import time
from six.moves import xrange    #...?
import math
tf.set_random_seed(777)  # reproducibility

#filenamequeue = tf.train.string_input_producer([os.path.join('.','cifar_data','data_batch_1.bin')])
#testnamequeue = tf.train.string_input_producer([os.path.join('.','cifar_data','test_batch.bin')])

imagesize = 32
image_NUM = 50000
test_NUM = 10000
MAX_STEPS = 1000
CLASS_NUM = 10
learning_rate = 0.001
training_epochs = 15
batch_SIZE = 100
data_dir = os.path.join('.','cifar_data')
train_dir = os.path.join('.','cifar_train')
eval_dir = os.path.join('.','cifar_eval')

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
    tf.summary.image('images', images)
    # 배치 과정을 거친 이미지와 라벨의 최종 shape는 각각 [100, 32, 32, 3]과 [100]
    return images, tf.reshape(label_batch, [batch_size])

def read_cifar10(filename_queue):
 class CIFAR10_record(object):
  pass
 result = CIFAR10_record()
 image_bytes = imagesize * imagesize * 3
 reader = tf.FixedLengthRecordReader(record_bytes=(1 + image_bytes))
 result.key, value = reader.read(filename_queue)
 record_bytes = tf.decode_raw(value,tf.uint8)
 result.label = tf.cast(tf.strided_slice(record_bytes,[0],[1]),tf.int32)
 depth_major = tf.reshape(tf.strided_slice(record_bytes,[1],[1+ image_bytes]),[3,imagesize,imagesize])
 result.uint8image = tf.cast(tf.transpose(depth_major,[1,2,0]), tf.float32)
 result.uint8image = tf.image.per_image_standardization(result.uint8image)
 result.uint8image.set_shape([imagesize, imagesize, 3])     #32x32x3
 result.label.set_shape([1])    #1x10000
 return result.uint8image, result.label

class Model:
    def inference(images):
        with tf.variable_scope('conv1')as scope:
            W1 = tf.get_variable(name='w1',shape=[3,3,3,32],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32))
            L1 = tf.nn.conv2d(images, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1,name=scope.name)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides = [1, 2, 2, 1], padding = 'SAME')

        with tf.variable_scope('conv2')as scope:
            W2 = tf.get_variable('w2',shape=[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2,name=scope.name)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides = [1, 2, 2, 1], padding = 'SAME')

        with tf.variable_scope('conv3')as scope:
            W3 = tf.get_variable('w3',shape=[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3,name=scope.name)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],strides = [1, 2, 2, 1], padding = 'SAME')

        with tf.variable_scope('Fc1')as scope:
            F1 = tf.reshape(L3,shape=[-1,128*4*4],name='F1')
            W4 = tf.get_variable('W4',shape=[128*4*4,625],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32))
            bias1 = tf.zeros([625],dtype=tf.float32)
            L4 = tf.nn.relu(tf.matmul(F1,W4)+ bias1,name=scope.name)

        with tf.variable_scope('Fc2')as scope:
            W5 = tf.get_variable('W5',shape=[625,CLASS_NUM],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32))
            bias2 = tf.zeros([CLASS_NUM])
            L5 = tf.add(tf.matmul(L4, W5), bias2,name=scope.name)
            L5 = tf.cast(L5,tf.float32)
        return L5

def lossfunc(logits, labels):
     labels_ = tf.cast(labels, tf.int64)
     #Computes sparse softmax cross entropy between logits and labels
     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
         labels=labels_, logits=logits, name='cross_entropy_per_example')
     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
     #labels_n = tf.one_hot(labels, CLASS_NUM,dtype=tf.int64)
     '''    optimizer = tf.train.AdamOptimizer(
         learning_rate=learning_rate).minimize(cross_entropy_mean)
     '''
     tf.add_to_collection('losses', cross_entropy_mean)

     # The total loss is defined as the cross entropy loss plus all of the weight
     # decay terms (L2 loss).
     return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.

  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss, global_step):
  num_batches_per_epoch = image_NUM / batch_SIZE
  decay_steps = int(num_batches_per_epoch * 350.0)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(0.1,
                                  global_step,
                                  decay_steps,
                                  0.1,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  results = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([results]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      0.9999, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op



def training():
    with tf.Graph().as_default():
         global_step = tf.train.get_or_create_global_step()

         # Get images and labels for CIFAR-10.
         # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
         # GPU and resulting in a slow down.

         with tf.device('/cpu:0'):
             filenamequeue = tf.train.string_input_producer([os.path.join(data_dir, 'data_batch_%d.bin'%i)
                                                             for i in xrange(1, 6)])
             images, labels = read_cifar10(filenamequeue)
             images, labels = _generate_image_and_label_batch(images, labels, int(image_NUM * 0.4), batch_SIZE,
                                         shuffle=False)

         # Build a Graph that computes the logits predictions from the
         # inference model.
         logits = Model.inference(images)

         # Calculate loss.
         loss = lossfunc(logits, labels)
         # Build a Graph that trains the model with one batch of examples and
         # updates the model parameters.
         train_op = train(loss, global_step)

         class _LoggerHook(tf.train.SessionRunHook):
             """Logs loss and runtime."""

             def begin(self):
                 self._step = -1
                 self._start_time = time.time()

             def before_run(self, run_context):
                 self._step += 1
                 return tf.train.SessionRunArgs(loss)  # Asks for loss value.

             def after_run(self, run_context, run_values):
                 #FLAG.log_frequency
                 if self._step % 10 == 0:
                     current_time = time.time()
                     duration = current_time - self._start_time
                     self._start_time = current_time

                     loss_value = run_values.results
                     examples_per_sec = 10 * batch_SIZE/ duration
                     sec_per_batch = float(duration / 10)

                     format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                   'sec/batch)')
                     print(format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))

         with tf.train.MonitoredTrainingSession(
                 checkpoint_dir=train_dir,
                 hooks=[tf.train.StopAtStepHook(last_step=MAX_STEPS),
                        tf.train.NanTensorHook(loss),
                        _LoggerHook()],
                 config=tf.ConfigProto(
                     log_device_placement=False)) as mon_sess:
             while not mon_sess.should_stop():
               mon_sess.run(train_op)
             #if mon_sess.should_stop():
             #    evaluate()

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      num_iter = int(math.ceil(test_NUM / batch_SIZE))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * batch_SIZE
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    testnamequeue = tf.train.string_input_producer([os.path.join(data_dir, 'test_batch.bin')])
    images, labels = read_cifar10(testnamequeue)
    images, labels = _generate_image_and_label_batch(images, labels,4000, batch_SIZE,False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = Model.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      time.sleep(60 * 5)

def main(argv=None):  # pylint: disable=unused-argument
  training()

if __name__ == '__main__':
    print("Let's get started")
    print(datetime.now())
    tf.app.run()

'''
with tf.Session() as sess:
    with tf.device('/cpu:0'):
        images, labels = read_cifar10(filenamequeue)

    batch = int(image_NUM/batch_SIZE)
    print(datetime.now(),'Start to learn!')
      for i in range(training_epochs):
        true_count = 0
        avg = 0
        for j in range(batch):
            images_d, labels_d = _generate_image_and_label_batch(images, labels, int(image_NUM * 0.4), batch_SIZE,
                                                                 shuffle=False)
            logits= Model.inference(images_d, labels_d)
            top_k_op = tf.nn.in_top_k(logits, labels_d, 1)
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, collection=tf.GraphKeys.QUEUE_RUNNERS, coord=coord)
            true_count += np.sum(sess.run([top_k_op]))
        coord.request_stop()
        coord.join(threads)
        print(datetime.now(),'cost = ',cost,'accuracy = ',true_count/image_NUM)
    '''
