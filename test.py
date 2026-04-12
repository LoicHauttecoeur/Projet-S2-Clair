import tensorflow as tf
from keras import layers, models
import os 
import numpy as np


dossier_actuel = os.path.dirname(__file__)
chemin_dataset = os.path.join(dossier_actuel, "dataset")

hauteur = 128
largeur = 128
trait = 32 

print(f"Je cherche les images ici : {chemin_dataset}")

#chargement des données
print("--- Chargement des données d'entraînement ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    chemin_dataset,          
    validation_split=0.2,    
    subset="training",       
    seed=123,                
    image_size=(hauteur, largeur),
    batch_size=trait
)

print("\n--- Chargement des données de validation ---")
val_ds = tf.keras.utils.image_dataset_from_directory(
    chemin_dataset,          
    validation_split=0.2,
    subset="validation",     
    seed=123,
    image_size=(hauteur, largeur),
    batch_size= trait
)


class_names = train_ds.class_names
print(f"\nLes classes trouvées sont : {class_names}")

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

#creation du modele
num_classes = 3  

model = models.Sequential([
 
  data_augmentation,
  
  
  layers.Rescaling(1./255),
  

  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  
  
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Modèle construit !")

#train

epochs = 10 # le nombre de fois ou il va étudier tout le dataset
print(f"Début de l'entraînement pour {epochs} époques...")

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



print("\n--- TEST DE RECONNAISSANCE ---")


for images, labels in val_ds.take(1):
    
    predictions = model.predict(images)
    score = tf.nn.softmax(predictions[0]) 
    
    vrai_nom = class_names[labels[0]]
    nom_predit = class_names[np.argmax(score)]
    confiance = 100 * np.max(score)

    print(f"L'image est réellement un : {vrai_nom}")
    print(f"L'IA pense que c'est un   : {nom_predit} (à {confiance:.2f}% de sûreté)")
    
    if vrai_nom == nom_predit:
        print("--> BRAVO ! C'est gagné.")
    else:
        print("--> Oups, encore des progrès à faire.")
chemin_sauvegarde = os.path.join(dossier_actuel, 'mon_pokedex.keras')

model.save(chemin_sauvegarde)
print(f"Le modèle est sauvegardé ici : {chemin_sauvegarde}")