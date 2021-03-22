import os
import tensorflow as tf
from typing import List
from utils.utils import psnr
from .kpn import kpn
from loss.losses import charbonnier_loss
from data.data_provider import DataLoader


class DenoiseTrainer:

    def __init__(self):
        self.model = None
        self.crop_size = None
        self.train_dataset = None
        self.valid_dataset = None
        # self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        # if len(tf.config.list_physical_devices('GPU')) > 1:
        #     self.strategy = tf.distribute.MirroredStrategy()

    def build_dataset(
            self, train_noise_images: List[str], train_clean_images: List[str],
            valid_noise_images: List[str], valid_clean_images: List[str],
            crop_size: int, batch_size: int):
        self.crop_size = crop_size
        self.train_dataset = DataLoader(
            images_noise=train_noise_images,
            images_clean=train_clean_images
        ).build_dataset(
            image_crop_size=crop_size, batch_size=batch_size, is_dataset_train=True)
        self.valid_dataset = DataLoader(
            images_noise=valid_noise_images,
            images_clean=valid_clean_images
        ).build_dataset(
            image_crop_size=crop_size, batch_size=batch_size, is_dataset_train=False)

    def compile(self, learning_rate=1e-4):
        # self.model = kpn(self.crop_size, num_rrg, num_mrb, channels)
        input_tensor = tf.keras.Input(shape=[128, 128, 3])
        output_tensor = kpn(input_tensor)
        self.model = tf.keras.Model(input_tensor, output_tensor)
        loss_function = charbonnier_loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=[psnr])

    def train(self, epochs: int, checkpoint_dir: str):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val",
                patience=10
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val', factor=0.5,
                patience=5, verbose=1, min_delta=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(checkpoint_dir, 'low_light_weights_best.h5'),
                monitor="val_psnr", save_weights_only=True,save_freq='epoch'
            )
        ]
        history = self.model.fit(
            self.train_dataset, validation_data=self.valid_dataset,
            epochs=epochs, callbacks=callbacks, verbose=1,validation_steps=10
        )
        return history