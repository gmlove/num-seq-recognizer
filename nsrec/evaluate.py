
"""Evaluate the model.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""

import math
import os.path
import time


import numpy as np
import tensorflow as tf

from nsrec.cnn_model import CNNModelConfig, CNNEvalModel

FLAGS = tf.flags.FLAGS

current_dir = os.path.dirname(os.path.abspath(__file__))
default_metadata_file_path = os.path.join(current_dir, '../data/test/metadata.pickle')
tf.flags.DEFINE_string("metadata_file_path", default_metadata_file_path,
                       "Meta data file path.")

default_checkpoint_dir = os.path.join(current_dir, '../output/train')
tf.flags.DEFINE_string("checkpoint_dir", default_checkpoint_dir,
                       "Directory containing model checkpoints.")

default_eval_dir = os.path.join(current_dir, '../output/eval')
tf.flags.DEFINE_string("eval_dir", default_eval_dir, "Directory to write event logs.")

tf.flags.DEFINE_integer("eval_interval_secs", 60,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 10132,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 500,
                        "Minimum global step to run evaluation.")


def evaluate_model(sess, model, global_step, summary_writer, summary_op):
  """
  Args:
    sess: Session object.
    model: Instance of CNNEvalModel; the model to evaluate.
    global_step: Integer; global step of the model checkpoint.
    summary_writer: Instance of SummaryWriter.
    summary_op: Op for generating model summaries.
  """
  # Log model summaries on a single batch.
  summary_str = sess.run(summary_op)
  summary_writer.add_summary(summary_str, global_step)

  # Compute accuracy over the entire dataset.
  num_eval_batches = int(
    math.ceil(FLAGS.num_eval_examples / model.config.batch_size))

  start_time = time.time()
  correct_count = 0
  for i in range(num_eval_batches):
    correct_count += model.correct_count(sess)
    if not i % 10:
      tf.logging.info("Computed accuracy for %d of %d batches.", i + 1, num_eval_batches)
  eval_time = time.time() - start_time

  accuracy = correct_count / (num_eval_batches * model.config.batch_size)
  tf.logging.info("Accuracy = %f (%.2g sec)", accuracy, eval_time)

  # Log accuracy to the SummaryWriter.
  summary = tf.Summary()
  value = summary.value.add()
  value.simple_value = accuracy
  value.tag = "Accuracy"
  summary_writer.add_summary(summary, global_step)

  # Write the Events file to the eval directory.
  summary_writer.flush()
  tf.logging.info("Finished processing evaluation at global step %d.",
                  global_step)


def run_once(model, saver, summary_writer, summary_op):
  """Evaluates the latest model checkpoint.

  Args:
    model: Instance of CNNEvalModel; the model to evaluate.
    saver: Instance of tf.train.Saver for restoring model Variables.
    summary_writer: Instance of SummaryWriter.
    summary_op: Op for generating model summaries.
  """
  model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  if not model_path:
    tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                    FLAGS.checkpoint_dir)
    return

  with tf.Session() as sess:
    # Load model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", model_path)
    saver.restore(sess, model_path)
    global_step = tf.train.global_step(sess, model.global_step.name)
    tf.logging.info("Successfully loaded %s at global step = %d.",
                    os.path.basename(model_path), global_step)
    if global_step < FLAGS.min_global_step:
      tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                      FLAGS.min_global_step)
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Run evaluation on the latest checkpoint.
    try:
      evaluate_model(
        sess=sess,
        model=model,
        global_step=global_step,
        summary_writer=summary_writer,
        summary_op=summary_op)
    except Exception as e:  # pylint: disable=broad-except
      tf.logging.error("Evaluation failed.")
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.
  eval_dir = FLAGS.eval_dir
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  g = tf.Graph()
  with g.as_default():
    # Build the model for evaluation.
    model_config = CNNModelConfig()
    model_config.metadata_file_path = FLAGS.metadata_file_path
    model = CNNEvalModel(model_config)
    model.build()

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    # Create the summary operation and the summary writer.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(eval_dir)

    g.finalize()

    # Run a new evaluation run every eval_interval_secs.
    while True:
      start = time.time()
      tf.logging.info("Starting evaluation at " + time.strftime(
        "%Y-%m-%d-%H:%M:%S", time.localtime()))
      run_once(model, saver, summary_writer, summary_op)
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


def main(unused_argv):
  run()


if __name__ == "__main__":
  tf.app.run()
