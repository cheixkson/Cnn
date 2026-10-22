import tensorflow as tf
import matplotlib.pyplot as plt


train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(256, 256),
    batch_size=32,
    color_mode="rgb",
    label_mode="int",
    shuffle=True
)
print(train_ds.class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=(256, 256),
    batch_size=32,
    color_mode="rgb",
    label_mode="int",
    shuffle=False
)
print(val_ds.class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)),

    tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(2)
])

model.summary()
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
