import tensorflow as tf
import matplotlib as plt

# Charger Fashion-MNIST
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# Normaliser les pixels entre 0 et 1
x_train = x_train / 255.0

# Ajouter le canal des niveaux de gris
x_train = x_train[..., tf.newaxis]

# Sélectionner une seule image
image = x_train[:1]

print("Forme de l'image :", image.shape)

# Créer une couche de convolution
conv = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu"
)

# Appliquer la convolution
sortie = conv(image)

print("Forme de la sortie :", sortie.shape)
