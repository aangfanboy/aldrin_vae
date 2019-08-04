import tensorflow as tf
import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm


class Marshall:
    def load_image(self, x):
        img = tf.io.read_file(x)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.central_crop(img, 0.825)
        img = tf.image.resize(img, (self.xs, self.ys), method="nearest")

        if self.random_flip:
            img = tf.image.random_flip_left_right(img)

        return tf.cast(img, tf.float32) / 255.

    def __init__(self, main_path: str, image_shape: tuple, random_flip: bool = True):
        self.main_path = main_path
        self.xs, self.ys, self.channels = image_shape
        self.random_flip = random_flip

        self.x_data, self.y_data = self.read_all_data()
        self.x_data = tf.convert_to_tensor(self.x_data)

        self.dataset = tf.data.Dataset.from_tensor_slices(self.x_data).shuffle(len(self.x_data))
        self.dataset = self.dataset.map(self.load_image)
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def read_all_data(self):
        try:
            labels = np.load("processed_data/labels.npy", allow_pickle=True)
            paths = np.load("processed_data/paths.npy", allow_pickle=True)
        except FileNotFoundError:
            labels = []
            paths = []

            for path in tqdm(glob(f"{self.main_path.rstrip('/')}/*/*.*")):
                if path.endswith(".csv"):
                    csv_path = path
                    image_path = path.rstrip(".csv")+".png"

                    df = pd.read_csv(csv_path)
                    label = df.values[:, 1]

                    paths.append(image_path)
                    labels.append(label)

            paths, labels = np.array(paths), np.array(labels)

            tf.compat.v1.gfile.MakeDirs("processed_data")
            np.save("processed_data/labels.npy", labels)
            np.save("processed_data/paths.npy", paths)

        return paths, labels


if __name__ == '__main__':
    marshall = Marshall(
        main_path="cartoonset100k",
        image_shape=(182, 182, 3),
        random_flip=False
    )
