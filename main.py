"""
Distributed Tensorflow example
The original code was in @ischlag, but the distributed architecture is quite
different.
The code runs on TF 1.1. 
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 
The code requires 'tmux'.
The code runs on the local server only.

Run like this: 
$ bash run.sh

Then, by using ctrl+b+(window number, e.g., 0, 1, 2), 
you can change the terminal.

"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import signal, sys
from worker import Worker
from utils import *

# Define flags.
flags = tf.app.flags

flags.DEFINE_string('job_name', 'ps', "Either 'ps' or 'worker'")
flags.DEFINE_integer('task_index', 0, "Index of task within the job")
flags.DEFINE_integer('batch_size', 100, "Batch size")
flags.DEFINE_float('learning_rate', 0.0005, "Learning rate")
flags.DEFINE_integer('training_epochs', 20, "Training epochs")
flags.DEFINE_string('logdir', './tmp/mnist/1', "Log directory")
flags.DEFINE_integer('num_workers', 2, "Number of workers")
flags.DEFINE_integer('num_gpus', 1,
        "Number of gpus, less than or equal to num_workers")

FLAGS = flags.FLAGS

def main():
    # Load MNIST dataset.
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Cluster specification
    spec = cluster_spec(FLAGS.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec)

    # Signal
    def shutdown(signal, frame):
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Set GPU memory fraction.
    process_per_memory =\
            np.ceil(float(FLAGS.num_workers)/float(FLAGS.num_gpus))
    fraction = 0.99 / process_per_memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction)

    if FLAGS.job_name == 'ps':
        config = tf.ConfigProto(device_filters=['/job:ps'])
        server = tf.train.Server(cluster, job_name='ps',
                task_index=FLAGS.task_index, config=config)
        while True:
            time.sleep(1000)

    elif FLAGS.job_name == 'worker':
        config = tf.ConfigProto(gpu_options=gpu_options,
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=2)
        server = tf.train.Server(cluster, job_name='worker',
                task_index=FLAGS.task_index, config=config)
        worker = Worker(FLAGS.job_name, FLAGS.task_index, server)
        worker.learn(mnist)

if __name__ == '__main__':
    main()
