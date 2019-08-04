import tensorflow as tf


class Barney(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        pass

    def encoder_model(self, activation_function: tf.nn = tf.nn.elu, last_layer_activation: tf.nn = None):
        input_layer = tf.keras.layers.Input(shape=(self.xs, self.ys, self.channels), name="input_1")

        x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation=activation_function,
                                   kernel_regularizer=tf.keras.regularizers.l2())(input_layer)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation=activation_function,
                                   kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation=activation_function,
                                   kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, activation=activation_function,
                                   kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), strides=2, activation=activation_function,
                                   kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, activation=activation_function,
                                   kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, activation=activation_function,
                                   kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, activation=activation_function,
                                   kernel_regularizer=tf.keras.regularizers.l2())(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.last_layer_units*2, activation=last_layer_activation,
                                  kernel_regularizer=tf.keras.regularizers.l2())(x)

        model = tf.keras.models.Model(input_layer, x)
        model.summary()

        return model

    def decoder_model(self, activation_function: tf.nn = tf.nn.elu, last_layer_activation: tf.nn = None):
        input_layer = tf.keras.layers.Input(
            shape=(self.last_layer_units, ), name="input_2"
        )

        x = tf.keras.layers.Dense(512, activation=activation_function,
                                  kernel_regularizer=tf.keras.regularizers.l2())(input_layer)
        x = tf.keras.layers.Reshape((1, 1, 512))(x)

        x = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=2, activation=activation_function,
                                            kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=1, activation=activation_function,
                                            kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, activation=activation_function, padding="same",
                                            kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation=activation_function, padding="same",
                                            kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=1, activation=activation_function, padding="same",
                                            kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=1, activation=activation_function, padding="same",
                                            kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation=activation_function, padding="same",
                                            kernel_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Conv2DTranspose(self.channels, (3, 3), strides=1, activation=last_layer_activation,
                                            padding="same")(x)

        model = tf.keras.models.Model(input_layer, x)
        model.summary()

        return model

    def save_models(self):
        self.encoder.save(self.file_path.replace(".h5", "_encoder.h5"))
        self.decoder.save(self.file_path.replace(".h5", "_decoder.h5"))

    def __init__(self, image_shape: tuple, file_path: str, last_layer_units: int = 1024, lr: float = 0.001):
        super(Barney, self).__init__()
        self.xs, self.ys, self.channels = image_shape
        self.file_path = file_path

        self.last_layer_units, self.lr = last_layer_units, lr

        tf.compat.v1.gfile.MakeDirs("".join(self.file_path.split("/")[:-1]))

        try:
            self.encoder = tf.keras.models.load_model(self.file_path.replace(".h5", "_encoder.h5"),
                                                      custom_objects={"leaky_relu": tf.nn.leaky_relu})
        except OSError:
            self.encoder = self.encoder_model(
                activation_function=tf.nn.leaky_relu,
                last_layer_activation=None
            )

        try:
            self.decoder = tf.keras.models.load_model(self.file_path.replace(".h5", "_decoder.h5"),
                                                      custom_objects={"leaky_relu": tf.nn.leaky_relu})
        except OSError:
            self.decoder = self.decoder_model(
                activation_function=tf.nn.leaky_relu,
                last_layer_activation=None
            )

        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)

    def encode(self, x: tf.Tensor):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z: tf.Tensor, apply_sigmoid: bool = False):
        logits = self.decoder(z)
        if apply_sigmoid:
            logits = tf.sigmoid(logits)

        return logits

    @staticmethod
    def reparameterize(mean: tf.Tensor, logvar: float):
        return tf.random.normal(shape=mean.shape) * tf.exp(logvar * .5) + mean

    def generate_sample(self, eps: tf.Tensor):
        return self.decode(eps, apply_sigmoid=True)

    @staticmethod
    def log_normal_pdf(sample: float, mean: float, logvar: float, raxis: float = 1):
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + 1.837877), axis=raxis)

    @tf.function
    def compute_loss(self, x: tf.Tensor):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x), x_logit

    @tf.function
    def train_step(self, x: tf.Tensor):
        with tf.GradientTape() as tape:
            loss, outputs = self.compute_loss(x)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, outputs


if __name__ == '__main__':
    barney = Barney(
        image_shape=(128, 128, 3),
        file_path="models/my_model.h5",
        lr=0.0001
    )
