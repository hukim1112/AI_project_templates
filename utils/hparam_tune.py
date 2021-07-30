import tensorflow as tf
import tensorflow_addons as tfa
from .file import make_directory
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay, PolynomialDecay, CosineDecayRestarts
import math
from tensorboard.plugins.hparams import api as hp

def make_hparam_list():
    #하이퍼파라미터 설정
    HP_init = hp.HParam('init_lr', hp.Discrete([1E-5, 1E-4, 1E-3]))
    HP_optimizer = hp.HParam('optimizer', hp.Discrete(["RMSPROP", "ADAM_W"]))
    HP_scheduler = hp.HParam('lr_scheduler', hp.Discrete(['constant', 'piecewise_decay',
                                                            'linear_decay', 'cosine_decay_restart']))
    HP_batch = hp.HParam('batch_size', hp.Discrete([32, 64, 128]))
    HP_weight_decay = hp.HParam('weight_decay', hp.Discrete([1E-5, 5E-5, 1E-4]))
    #batch norm : momentum=0.99, epsilon 1E-3 (keras default settings)
    return [HP_init, HP_optimizer, HP_scheduler, HP_batch, HP_weight_decay]
def make_scheduler(name, init_lr, steps_per_epoch):
    if name == 'constant':
        lr_fn = init_lr
    elif name == 'piecewise_decay':
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
    elif name == 'linear_decay':
        decay_steps = steps_per_epoch * 300 # defaultly during 300 epoch, learning rate goes down.
        lr_fn = PolynomialDecay(initial_learning_rate=init_lr, decay_steps=decay_steps, end_learning_rate=init_lr*0.1, power=1.0)
    elif name == 'cosine_decay_restart':
        first_decay_steps = 1000
        lr_fn = CosineDecayRestarts(initial_learning_rate=init_lr, first_decay_steps=first_decay_steps, t_mul=2.0)
    print(name, lr_fn)
    return lr_fn

def custom_model(model):
    # your loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_tracker = tf.keras.metrics.Mean(name="loss")
    acc = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    class CustomModel(tf.keras.Model):
        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute our own loss
                loss = loss_fn(y, y_pred)
                l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
                l2_loss = self.l2_reg * tf.math.reduce_sum(l2_loss)
                total_loss = loss + l2_loss
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Compute our own metrics
            loss_tracker.update_state(loss)
            acc.update_state(y, y_pred)
            return {"loss": loss_tracker.result(), "acc": acc.result()}
        def test_step(self, data):
            # Unpack the data
            x, y = data
            # Compute predictions
            y_pred = self(x, training=False)
            # Updates the metrics tracking the loss
            loss = loss_fn(y, y_pred)

            # Update the metrics.
            loss_tracker.update_state(loss)
            acc.update_state(y, y_pred)

            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            return {m.name: m.result() for m in self.metrics}
        @property
        def metrics(self):
            # We list our `Metric` objects here so that `reset_states()` can be
            # called automatically at the start of each epoch
            # or at the start of `evaluate()`.
            # If you don't implement this property, you have to call
            # `reset_states()` yourself at the time of your choosing.
            return [loss_tracker, acc]
    return CustomModel(model.input, model.output)


class Keras_Tuner():
    def __init__(self, log_dir, ckpt_dir, initial_model_path, train_ds, val_ds, metric):
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        make_directory(log_dir)
        make_directory(ckpt_dir)
        self.metric = metric
        self.initial_model_path = initial_model_path
        self.train_ds = train_ds
        self.val_ds = val_ds

        # with tf.summary.create_file_writer(log_dir).as_default():
        #     hp.hparams_config(
        #     hparams=make_hparam_list(),
        #     metrics=[hp.Metric(metric, display_name=metric)],
        #   )
    def build_model(self, initial_model_path):
        model = tf.keras.models.load_model(initial_model_path)
        return custom_model(model)
    def write_log(self):
        pass

    def train_test_model(self, hparams, run_log_dir, run_ckpt_dir):
        hparams = {h.name: hparams[h] for h in hparams}
        model = self.build_model(self.initial_model_path)
        model.l2_reg = hparams["weight_decay"]
        steps_per_epoch = int(math.ceil(1.0*len(self.train_ds)/hparams["batch_size"]))
        lr_scheduler = make_scheduler(hparams["lr_scheduler"], hparams["init_lr"], steps_per_epoch)
        if hparams["optimizer"] == "RMSPROP":
            optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
        elif hparams["optimizer"] == "ADAM_W":
            optimizer = tfa.optimizers.AdamW(lr_scheduler)
        model.compile(optimizer)

        callbacks = [tf.keras.callbacks.EarlyStopping(
                    # Stop training when `val_loss` is no longer improving
                    monitor="val_loss",
                    # "no longer improving" being defined as "no better than 1e-2 less"
                    min_delta=0,
                    # "no longer improving" being further defined as "for at least 2 epochs"
                    patience=10,
                    verbose=1),
                    tf.keras.callbacks.ModelCheckpoint(
                        # Path where to save the model
                        # The two parameters below mean that we will overwrite
                        # the current checkpoint if and only if
                        # the `val_loss` score has improved.
                        # The saved model name will include the current epoch.
                        filepath=run_ckpt_dir + "/model-{epoch}.h5",
                        save_best_only=True,  # Only save a model if `val_loss` has improved.
                        monitor="val_loss",
                        verbose=1),
                    tf.keras.callbacks.TensorBoard(run_log_dir)
                ]
        train_ds = self.train_ds.shuffle(2000).batch(hparams["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = self.val_ds.batch(hparams["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)
        model.fit(train_ds, validation_data=val_ds, epochs=300, callbacks=callbacks)
        loss, acc = model.evaluate(val_ds)
        return acc

    def run(self, session_num, hparams):
        run_name = "run-%d" % session_num
        run_log_dir = self.log_dir + '/' + run_name
        run_ckpt_dir = self.ckpt_dir + '/' + run_name
        make_directory(run_ckpt_dir)
        with tf.summary.create_file_writer(run_log_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = self.train_test_model(hparams, run_log_dir, run_ckpt_dir)
            tf.summary.scalar(self.metric, accuracy, step=1)

    def tuning(self):
        #하이퍼파라미터 설정
        HP_init = hp.HParam('init_lr', hp.Discrete([1E-5, 1E-4, 1E-3]))
        HP_optimizer = hp.HParam('optimizer', hp.Discrete(["RMSPROP", "ADAM_W"]))
        HP_scheduler = hp.HParam('lr_scheduler', hp.Discrete(['constant', 'piecewise_decay',
                                                                'linear_decay', 'cosine_decay_restart']))
        HP_batch = hp.HParam('batch_size', hp.Discrete([32, 64, 128]))
        HP_weight_decay = hp.HParam('weight_decay', hp.Discrete([1E-5, 5E-5, 1E-4]))
        session_num = 0

        for init_lr in HP_init.domain.values:
            for optimizer in HP_optimizer.domain.values:
                for lr_scheduler in HP_scheduler.domain.values:
                    for batch_size in HP_batch.domain.values:
                        for weight_decay in HP_weight_decay.domain.values:
                            hparams = {
                                HP_init : init_lr,
                                HP_optimizer: optimizer,
                                HP_scheduler: lr_scheduler,
                                HP_batch: batch_size,
                                HP_weight_decay : weight_decay
                            }
                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            self.run(session_num, hparams)
                            session_num += 1
