import tensorflow as tf
from tensorflow.keras.layers import Conv2D


class CAModel(tf.keras.Model):

    def __init__(self, channel_n=19, fire_rate=0.5,
                 add_noise=True):
        # CHANNEL_N does *not* include the greyscale channel.
        # but it does include the 10 possible outputs.
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.add_noise = add_noise

        self.perceive = tf.keras.Sequential([
            Conv2D(80, 3, activation=tf.nn.relu, padding="SAME"),
        ])

        self.dmodel = tf.keras.Sequential([
            Conv2D(80, 1, activation=tf.nn.relu),
            Conv2D(self.channel_n, 1, activation=None,
                   kernel_initializer=tf.zeros_initializer),
        ])

        self(tf.zeros([1, 3, 3, channel_n + 1]))  # dummy calls to build the model

    @tf.function
    def call(self, x, fire_rate=None, manual_noise=None):
        gray, state = tf.split(x, [1, self.channel_n], -1)
        ds = self.dmodel(self.perceive(x))
        if self.add_noise:
            if manual_noise is None:
                residual_noise = tf.random.normal(tf.shape(ds), 0., 0.02)
            else:
                residual_noise = manual_noise
            ds += residual_noise

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        living_mask = gray > 0.1
        residual_mask = update_mask & living_mask
        ds *= tf.cast(residual_mask, tf.float32)
        state += ds

        return tf.concat([gray, state], -1)

    @tf.function
    def initialize(self, images):
        state = tf.zeros([tf.shape(images)[0], 28, 28, self.channel_n])
        images = tf.reshape(images, [-1, 28, 28, 1])
        return tf.concat([images, state], -1)

    @tf.function
    def classify(self, x):
        # The last 10 layers are the classification predictions, one channel
        # per class. Keep in mind there is no "background" class,
        # and that any loss doesn't propagate to "dead" pixels.
        return x[:, :, :, -10:]


CAModel().perceive.summary()
CAModel().dmodel.summary()