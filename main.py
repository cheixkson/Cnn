import tensorflow as tf
import numpy as np 
from model_ia import model
import model_ia

image_path = "images.jfif"

image = tf.keras.utils.load_img(
    image_path,
    target_size=(256, 256),
    color_mode="rgb"
)

image_array = tf.keras.utils.img_to_array(image)

# Ajouter la dimension du lot
image_array = np.expand_dims(image_array, axis=0)

print("Forme de l'image :", image_array.shape)

predictions = model.predict(image_array)
predicted_index = np.argmax(predictions[0])
predicted_class = main.train_ds.class_names[predicted_index]
print("Scores bruts :", predictions)
print("Classe prédite :", predicted_class)



