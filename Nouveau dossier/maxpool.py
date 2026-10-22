import tensorflow as tf

# Exemple : une image, une carte de caractéristiques
x = tf.constant([[
    [[1.0], [3.0], [2.0], [4.0]],
    [[5.0], [6.0], [1.0], [2.0]],
    [[2.0], [4.0], [8.0], [7.0]],
    [[3.0], [1.0], [6.0], [9.0]]
]])

pool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2)
)

y = pool(x)

print(y.numpy())
print("Forme :", y.shape)