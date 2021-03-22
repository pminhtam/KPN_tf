from typing import List
from .common import *


class DataLoader:

    def __init__(self, images_noise: List[str], images_clean: List[str]):
        self.images_noise = images_noise
        self.images_clean = images_clean

    def __len__(self):
        assert len(self.images_noise) == len(self.images_noise)
        return len(self.images_noise)

    def build_dataset(self, image_crop_size: int, batch_size: int, is_dataset_train: bool):
        low_light_dataset = read_images(self.images_noise)
        high_light_dataset = read_images(self.images_clean)
        dataset = tf.data.Dataset.zip((low_light_dataset, high_light_dataset))
        dataset = dataset.map(apply_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda low, high: random_crop(low, high, image_crop_size, image_crop_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        if is_dataset_train:
            dataset = dataset.map(random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


