import tensorflow as tf

def get_vars(scope, trainable=True):
    if trainable:
        keys = tf.GraphKeys.TRAINABLE_VARIABLES
    else:
        keys = tf.GraphKeys.GLOBAL_VARIABLES
    return tf.get_collection(keys, scope)

def cluster_spec(num_workers, num_ps):
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix='meta', write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step,
                latest_filename, meta_graph_suffix, False)
