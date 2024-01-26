import ASPIREv as ges
import matplotlib.pyplot as plt
import tensorflow as tf

plt.close('all')

SAT_image, meta_data = ges.io.open_geoTiFF("_data/WorldView_area_training_RGB.tif")
X = ges.dataset._create_dataset(SAT_image, meta_data, patch_number = 1000, patch_radius = 40)
# Convert NumPy arrays to TensorFlow tensors
X = tf.convert_to_tensor(X, dtype=tf.uint8)

dataset = tf.data.Dataset.from_tensor_slices(X)


def augment_image(image, delta = 0.5):
    image = tf.image.adjust_brightness(image, delta=tf.random.uniform([], -delta, delta))    
    return image


dataset = dataset.map(augment_image)

batch_size = 32  # You can choose an appropriate batch size
# dataset = dataset.batch(batch_size)
# dataset = dataset.shuffle(buffer_size=len(X))
# dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


num_images_to_display = 1  # Adjust the number of images you want to display

for image in dataset.take(num_images_to_display):
    image = image.numpy()  # Convert the TensorFlow tensor to a NumPy array
    plt.figure()
    plt.imshow(image)
    plt.show()