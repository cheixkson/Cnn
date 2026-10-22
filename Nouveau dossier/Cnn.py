import tensorflow as tf
import numpy as np
# 1. Charger les données
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# 2. Normaliser les pixels entre 0 et 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Ajouter le canal des images en niveaux de gris
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 4. Construire le CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(
        64, kernel_size=(3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation="relu")
    
])

# 5. Préparer l'entraînement
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)

# 6. Afficher les dimensions et les paramètres
model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Loss de test :", test_loss)
print("Accuracy de test :", test_accuracy)






# Noms des 10 catégories Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Sélectionner une image du jeu de test
image = x_test[1:2]

# Calculer les 10 logits
logits = model.predict(image, verbose=0)

# Trouver la classe ayant le logit le plus élevé
predicted_class = np.argmax(logits[0])

# Vraie étiquette
true_class = y_test[1]

print("Logits :", logits[0])
print("Classe prédite :", class_names[predicted_class])
print("Vraie classe :", class_names[true_class])

