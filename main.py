from glob import glob
from model.train import DenoiseTrainer
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

train_low_light_images = glob('/home/dell/Downloads/noise/0001_NOISY_SRGB/*.PNG')
train_high_light_images = glob('/home/dell/Downloads/gt/0001_GT_SRGB/*.PNG')
valid_low_light_images = glob('/home/dell/Downloads/noise/0001_NOISY_SRGB/*.PNG')
valid_high_light_images = glob('/home/dell/Downloads/gt/0001_GT_SRGB/*.PNG')

trainer = DenoiseTrainer()
trainer.build_dataset(
    train_low_light_images, train_high_light_images,
    valid_low_light_images, valid_high_light_images,
    crop_size=128, batch_size=2
)

trainer.compile()

trainer.train(epochs=100, checkpoint_dir='./checkpoints')