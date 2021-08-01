import os
import tensorflow as tf

class RecordLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, logdir):
        super(RecordLearningRate, self).__init__()
        self.logdir = logdir
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        with tf.summary.create_file_writer(os.path.join(self.logdir, 'train')).as_default():
            tf.summary.scalar("epoch_learning_rate", lr, step=epoch)
