import os
import tensorflow as tf
from model.kpn import kpn
from glob import glob
from PIL import Image
import numpy as np
import imageio

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


input_tensor = tf.keras.Input(shape=[1024, 1024, 3])
output_tensor = kpn(input_tensor)
model = tf.keras.Model(input_tensor, output_tensor)
model.load_weights(os.path.join('low_light_weights_best.h5'))
valid_low_light_images = glob('/home/dell/Downloads/noise/0001_NOISY_SRGB/*.PNG')
valid_high_light_images = glob('/home/dell/Downloads/gt/0001_GT_SRGB/*.PNG')


# train_dataset = DataLoader(
#             images_noise=valid_low_light_images,
#             images_clean=valid_high_light_images
#         ).build_dataset(
#             image_crop_size=128, batch_size=2, is_dataset_train=True)
# print(next(iter(train_dataset)))
# exit()
for img_path in valid_low_light_images:
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = img[1024:1024+1024,1024:1024+1024,:]
    img = np.array([img])/255.0
    image_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    print(img)
    result = model(img)
    imageio.imwrite("img.png",result[0])
    print(result)
    exit()