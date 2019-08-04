import VAEmodel
import data_loader
import tensorflow as tf
import matplotlib.pyplot as plt


class Robin:
    def __init__(self, marshall_pipeline: data_loader.Marshall, barney_model: VAEmodel.Barney, epochs: int = 10,
                 batch_size: int = 32):
        self.marshall_data = marshall_pipeline
        self.barney_model = barney_model
        self.batch_size, self.epochs = batch_size, epochs

        self.file_writer = tf.summary.create_file_writer("graphs/")
        self.marshall_data.dataset = self.marshall_data.dataset.batch(self.batch_size)

    def save_images_to_tensorboard(self, epoch: int, real_ex: tf.Tensor, regenerated_ex: tf.Tensor):
        samples_from_random = self.barney_model.generate_sample(tf.random.normal(
                shape=(self.batch_size, self.barney_model.last_layer_units,)
            ))

        with tf.device("/cpu:0"):
            with self.file_writer.as_default():
                tf.summary.image("real images", real_ex.numpy(), step=epoch, max_outputs=self.batch_size,
                                 description="real images, no effect from Barney!")

                tf.summary.image("regenerated images", regenerated_ex.numpy(), step=epoch, max_outputs=self.batch_size,
                                 description="regenerated images, encoded and decoded by Barney!")

                tf.summary.image("decoded images", samples_from_random.numpy(), step=epoch, max_outputs=self.batch_size,
                                 description="decoded images, generated from random bottleneck, decoded by Barney!")

    def train_model(self):
        x = regenerated_images = None
        q = int(tf.data.experimental.cardinality(self.marshall_data.dataset))

        for epoch in range(self.epochs):
            loss_metric = tf.keras.metrics.Mean()
            bar = tf.keras.utils.Progbar(target=q, stateful_metrics=["loss"])

            for i, x in enumerate(self.marshall_data.dataset):
                loss_value, regenerated_images = self.barney_model.train_step(x=x)
                loss_metric(loss_value)

                loss_result = round(float(loss_metric.result()), 5)

                with self.file_writer.as_default():
                    tf.summary.scalar("loss", loss_result, step=(q*epoch)+i, description="Barney's Loss")

                bar.update(current=int(i+1), values=[["loss", loss_result]])

            self.save_images_to_tensorboard(
                epoch=epoch,
                real_ex=x,
                regenerated_ex=regenerated_images
            )

            self.barney_model.save_models()

    def generate_random_images(self, number_of_images: int = 100):
        samples_from_random = self.barney_model.generate_sample(tf.random.normal(
            shape=(number_of_images, self.barney_model.last_layer_units,)
        ))

        c = r = int(tf.sqrt(float(number_of_images)).numpy())
        fig = plt.figure(figsize=(64, 64))

        for i in range(int(c*r)):
            fig.add_subplot(r, c, i+1)
            plt.axis("off")
            plt.imshow(samples_from_random[i])

        plt.savefig("results.png")
        plt.show()


if __name__ == '__main__':
    marshall = data_loader.Marshall(
        main_path="cartoonset100k",
        image_shape=(128, 128, 3),
        random_flip=False
    )

    barney = VAEmodel.Barney(
        image_shape=(128, 128, 3),
        file_path="models/my_model.h5",
        last_layer_units=256,
        lr=0.0001
    )

    robin = Robin(
        marshall_pipeline=marshall,
        barney_model=barney,
        epochs=10,
        batch_size=32,
    )

    robin.generate_random_images(number_of_images=100)
