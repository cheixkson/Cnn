
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Charger le dataset Fashion-MNIST

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Afficher les dimensions des données
print("Images d'entraînement :", x_train.shape)
print("Étiquettes d'entraînement :", y_train.shape)
print("Images de test :", x_test.shape)

# 3. Afficher une image
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Classe : {y_train[0]}")
plt.axis("off")
plt.show()



# Normaliser les images d'entraînement
x_train = x_train/255.0

# Normaliser les images de test
x_test = x_test/255.0

# Vérifier le résultat
print("Minimum :", x_train.min())
print("Maximum :", x_train.max())
print("Dimensions :", x_train.shape)



model = tf.keras.Sequential([
    # Transformer l'image 28 x 28 en vecteur
    tf.keras.layers.Flatten(input_shape=(28,28)),

    # Couche cachée de 128 neurones
    tf.keras.layers.Dense(128, activation="relu"),

    # Couche de sortie : 10 classes
    tf.keras.layers.Dense(10)
])

# Afficher la structure du modèle
model.summary()

  

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32
)



test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Loss sur le test :", test_loss)
print("Accuracy sur le test :", test_accuracy)
print("Pourcentage de bonnes prédictions :", test_accuracy * 100)





# Courbe de la loss
plt.plot(history.history["loss"])
plt.title("Loss pendant l'entraînement")
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.show()

# Courbe de l'accuracy
plt.plot(history.history["accuracy"])
plt.title("Accuracy pendant l'entraînement")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
#plt.show()





test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Loss test :", test_loss)
print("Accuracy test :", test_accuracy)

print("Loss finale entraînement :", history.history["loss"][-1])
print("Accuracy finale entraînement :", history.history["accuracy"][-1])

# 1. Charger l'image
image = Image.open("test.jpg")

# 2. Convertir en niveaux de gris
image = image.convert("L")

# 3. Redimensionner en 28 x 28 pixels
image = image.resize((28, 28))

# 4. Convertir en tableau NumPy
image = np.array(image)

# 5. Normaliser les pixels entre 0 et 1
image = image / 255.0

# 6. Ajouter la dimension du lot
image = image.reshape(1, 28, 28)

print("Forme de l'image :", image.shape)

predictions = model.predict(image)

print(predictions.shape)
print(predictions[0])