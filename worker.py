import tensorflow as tf
import time
from utils import *

FLAGS = tf.app.flags.FLAGS

class Worker(object):
    def __init__(self, job_name, task_index, server):
        self.job_name = job_name
        self.task_index = task_index
        self.server = server

        # For shared parameters, including global step.
        global_device = '/job:{}/task:{}/cpu:0'.format(job_name, task_index)

        # For local computations,
        """
        The gradient is computed at each "local_device".
        Since CUDA_VISIBLE_DEVICES for each worker process allocates single
        gpu, '/gpu:0' is used.
        """
        local_device = '/job:{}/task:{}/gpu:0'.format(job_name, task_index)

        with tf.device(tf.train.replica_device_setter(1,
            worker_device=global_device)):

            with tf.variable_scope('global'):
                self.build_net()
                self.global_step = tf.get_variable('global_step', [], tf.int32,
                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                        trainable=False)
                self.counter_op = self.global_step.assign_add(1)

        with tf.device(local_device):
            with tf.variable_scope('local'):
                self.build_net()
                self.build_loss()
                self.build_sync_op()
                self.build_train_op()
                self.build_summary_op()

        self.build_init_op()
        self.build_saver()


    def build_net(self):
        self.x = tf.placeholder(tf.float32, [None, 784])

        def _net(inputs):
            net = tf.layers.dense(inputs, 100, activation=tf.nn.sigmoid,
                    kernel_initializer=tf.random_normal_initializer())
            logits = tf.layers.dense(net, 10,
                    kernel_initializer=tf.random_normal_initializer())
            net = tf.nn.softmax(logits)
            return net, logits

        self.net, self.logits = _net(self.x)

    def build_loss(self):
        self.y = tf.placeholder(tf.float32, [None, 10])

        def _loss(labels, logits):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
            
            return tf.reduce_mean(cross_entropy)
        
        self.loss = _loss(self.y, self.logits)

    def build_train_op(self):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        gvs = optimizer.compute_gradients(self.loss,
                var_list=get_vars('local'))
        
        global_gvs = []
        for v, gv in zip(get_vars('global'), gvs):
            global_gvs.append((gv[0], v))
        
        self.train_op = optimizer.apply_gradients(global_gvs)
 
    def build_sync_op(self):
        local_vars = get_vars('local')
        global_vars = get_vars('global')
        self.sync_op = tf.group(*[v1.assign(v2)\
                for v1, v2 in zip(local_vars, global_vars)])

    def build_summary_op(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.net, 1), tf.argmax(self.y, 1))
            accuracy = self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', accuracy)

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(FLAGS.logdir + '_%d' % self.task_index)

    def build_init_op(self):
        self.global_init_op = tf.variables_initializer(get_vars('global', False))
        self.local_init_op = tf.variables_initializer(get_vars('local', False))

    def build_saver(self):
        self.saver = FastSaver(get_vars('global', False))

    def learn(self, dataset):

        sv = tf.train.Supervisor(is_chief=(self.task_index==0),
                                 logdir=FLAGS.logdir,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 ready_op=tf.report_uninitialized_variables(
                                     get_vars('global', False)),
                                 global_step=self.global_step,
                                 save_model_secs=30,
                                 save_summaries_secs=30,
                                 init_op=self.global_init_op,
                                 local_init_op=self.local_init_op)

        config = tf.ConfigProto(allow_soft_placement=True,
                log_device_placement=False)

        with sv.managed_session(self.server.target, config=config) as sess, sess.as_default():

            begin_time = time.time()
            start_time = time.time()
            start_step = 0

            while not sv.should_stop():

                sess.run(self.sync_op)

                batch_x, batch_y = dataset.train.next_batch(FLAGS.batch_size)           		
                
                _, cost, summary_str, step = sess.run(
                        [self.train_op, self.loss, self.summary_op, self.global_step], 
                        feed_dict={self.x: batch_x, self.y: batch_y})

                self.summary_writer.add_summary(summary_str, step)
                
                sess.run(self.counter_op)

                if step % 100 == 0 and step != 0:
                    elapsed_time = time.time() - start_time
                    print("step: {}\t| cost: {}\t| speed: {}step/sec".format(
                        step, cost, float((step - start_step) / elapsed_time)))
                    start_time = time.time()
                    start_step = step
               
                if step % 10000 == 0:
                    print("test accuracy: {}".format(
                        sess.run(self.accuracy,
                            {self.x: dataset.test.images, self.y: dataset.test.labels})
                        ))

                
                        
                if step >= FLAGS.training_steps:
                    break
