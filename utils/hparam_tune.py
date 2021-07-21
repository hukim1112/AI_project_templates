import tensorflow as tf
import tensorflow_addons as tfa
from .file import make_directory
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay



def make_hparam_list():
    #하이퍼파라미터 설정
    HP_optimizer = hp.HParam('optimizer', hp.Discrete(["RMSPROP", "ADAM_W"]))
    HP_scheduler = hp.HParam('lr scheduler', hp.Discrete(['constant', 'piecewide decay',
                                                            'linear decay', 'cosine decay restart']))
    HP_batch = hp.HParam('batch size', hp.Discrete([32, 64, 128]))
    HP_weight_decay = hp.HParam('weight_decay', hp.Discrete([1E-5, 5E-5, 1E-4]))
    #batch norm : momentum=0.99, epsilon 1E-3 (keras default settings)
    return [HP_init, HP_optimizer, HP_scheduler, HP_batch, HP_weight_decay]

def make_scheduler(name, init_lr):
    if name == 'constant':
        lr_fn = init_lr
    elif name == 'piecewise decay':
        lr_fn = PiecewiseConstantDecay(
                boundaries=[20*steps_per_epoch,
                            50*steps_per_epoch,
                            100*steps_per_epoch,
                            150*steps_per_epoch,
                            200*steps_per_epoch,
                            250*steps_per_epoch],
                values=[init_lr,
                        init_lr * 0.8, #50
                        init_lr * 0.6, #50
                        init_lr * 0.4, #100
                        init_lr * 0.3, #150
                        init_lr * 0.2, #200
                        init_lr * 0.1]) #250
    elif name == 'linear decay':
        pass
    elif name == '':
        first_decay_steps = 1000
        lr_decayed_fn = (
          tf.keras.optimizers.schedules.CosineDecayRestarts(
              initial_learning_rate,
              first_decay_steps))



class tuner():
    def __init__(self, log_dir, hparams, initial_model, metric):
        make_directory(log_dir)
        self.hparams = hparams
        self.initial_model = initial_model
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams_config(
            hparams=make_hparam_list(),
            metrics=[hp.Metric(metric, display_name=metric)],
          )
    def build_model(self):
        return tf.keras.models.load_model(self.initial_model)
    def write_log(self):
        pass
    def run(self, init_lr, train_step, test_step, params):
        if params["optimizer"] == "RMSPROP":
            optimizer = tf.keras.optimizers.RMSprop()




        for epoch in range(300):
            train_step
