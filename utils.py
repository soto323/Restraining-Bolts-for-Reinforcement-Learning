import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

def load_checkpoint(ckpt_dir_or_file, session, var_list=None):
    """
    Load checkpoint and return epoch_to_restore
    """
    if os.path.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    restorer = tf.train.Saver(var_list)
    restorer.restore(session, ckpt_dir_or_file)
    print('[*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)
    epoch_to_restore = int(ckpt_dir_or_file.split("/")[-1].split("_")[1].split(".")[0])
    print("episode: {}".format(epoch_to_restore))
    return epoch_to_restore + 1


def summary(tensor_collection,
            summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram'],
            scope=None):
    """Summary.

    usage:
        1. summary(tensor)y
        2. summary([tensor_a, tensor_b])
        3. summary({tensor_a: 'a', tensor_b: 'b})
    """
    def _summary(tensor, name, summary_type):
        """Attach a lot of summaries to a Tensor."""
        if name is None:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
            name = re.sub(':', '-', name)

        summaries = []
        if len(tensor.shape) == 0:
            summaries.append(tf.summary.scalar(name, tensor))
        else:
            if 'mean' in summary_type:
                mean = tf.reduce_mean(tensor)
                summaries.append(tf.summary.scalar(name + '/mean', mean))
            if 'stddev' in summary_type:
                mean = tf.reduce_mean(tensor)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                summaries.append(tf.summary.scalar(name + '/stddev', stddev))
            if 'max' in summary_type:
                summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
            if 'min' in summary_type:
                summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
            if 'sparsity' in summary_type:
                summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
            if 'histogram' in summary_type:
                summaries.append(tf.summary.histogram(name, tensor))
        return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]

    with tf.name_scope(scope, 'summary'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)
