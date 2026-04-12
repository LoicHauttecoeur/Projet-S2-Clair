import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import numpy as np
import matplotlib.pyplot as plt
import time                        # ← AJOUT pour le chronomètre
 
# ─────────────────────────────────────────────
#  PARAMÈTRES
# ─────────────────────────────────────────────
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
chemin_dataset  = os.path.join(dossier_actuel, "dataset")
chemin_modele   = os.path.join(dossier_actuel, "mon_pokedex.keras")
chemin_graphe   = os.path.join(dossier_actuel, "courbe_entrainement.png")
 
HAUTEUR      = 128
LARGEUR      = 128
TAILLE_BATCH = 32
EPOCHS       = 30
 
print(f"Dossier dataset : {chemin_dataset}\n")
 
# ─────────────────────────────────────────────
#  CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
print("--- Chargement des données d'entraînement ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    chemin_dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(HAUTEUR, LARGEUR),
    batch_size=TAILLE_BATCH
)
 
print("\n--- Chargement des données de validation ---")
val_ds = tf.keras.utils.image_dataset_from_directory(
    chemin_dataset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(HAUTEUR, LARGEUR),
    batch_size=TAILLE_BATCH
)
 
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nClasses détectées ({num_classes}) : {class_names}")
 
# Optimisation : mise en cache + préchargement
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
 
# ─────────────────────────────────────────────
#  AUGMENTATION DES DONNÉES
# ─────────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")
 
# ─────────────────────────────────────────────
#  CONSTRUCTION DU MODÈLE CNN
# ─────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(HAUTEUR, LARGEUR, 3)),
    data_augmentation,
    layers.Rescaling(1. / 255),
 
    # Bloc 1
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
 
    # Bloc 2
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
 
    # Bloc 3
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
 
    # Bloc 4
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
 
    # Tête de classification
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(num_classes)
], name="PokeDex_CNN")
 
model.summary()
 
# ─────────────────────────────────────────────
#  COMPILATION
# ─────────────────────────────────────────────
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
 
# ─────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────
liste_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=chemin_modele,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]
 
# ─────────────────────────────────────────────
#  ENTRAÎNEMENT
# ─────────────────────────────────────────────
print(f"\nDébut de l'entraînement (max {EPOCHS} époques, EarlyStopping actif)...\n")
 
temps_debut = time.time()          # ← CHRONO DÉMARRE ICI
 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=liste_callbacks
)
 
# ─────────────────────────────────────────────
#  COURBES D'ENTRAÎNEMENT
# ─────────────────────────────────────────────
def afficher_courbes(history, chemin_sauvegarde):
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
    ax1.plot(epochs_range, acc,     label='Entraînement', color='steelblue')
    ax1.plot(epochs_range, val_acc, label='Validation',   color='orange')
    ax1.set_title('Précision par époque')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Précision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    ax2.plot(epochs_range, loss,     label='Entraînement', color='steelblue')
    ax2.plot(epochs_range, val_loss, label='Validation',   color='orange')
    ax2.set_title('Perte par époque')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Perte')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
 
    plt.suptitle("Courbes d'entraînement – PokéDex CNN", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(chemin_sauvegarde, dpi=150)
    plt.show()
    print(f"Graphe sauvegardé : {chemin_sauvegarde}")
 
afficher_courbes(history, chemin_graphe)
 
# ─────────────────────────────────────────────
#  RÉSUMÉ FINAL POUR LE RAPPORT
# ─────────────────────────────────────────────
temps_total = time.time() - temps_debut
minutes     = int(temps_total // 60)
secondes    = int(temps_total % 60)
 
meilleure_val_acc  = max(history.history['val_accuracy']) * 100
derniere_train_acc = max(history.history['accuracy'])     * 100
nb_epoques         = len(history.history['accuracy'])
 
print("\n" + "="*55)
print("  RESUME --- A COPIER DANS LE RAPPORT")
print("="*55)
print(f"  Précision entraînement  : {derniere_train_acc:.1f} %")
print(f"  Précision validation    : {meilleure_val_acc:.1f} %")
print(f"  Époques effectuées      : {nb_epoques} / {EPOCHS}")
print(f"  Temps d'entraînement    : {minutes} min {secondes} sec")
print("="*55)
 
# ─────────────────────────────────────────────
#  TEST SUR TOUTES LES CLASSES
# ─────────────────────────────────────────────
print("\n--- TEST DE RECONNAISSANCE SUR LE LOT DE VALIDATION ---")
 
exemples_par_classe = {}
 
for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    for i in range(len(images)):
        classe_reelle = class_names[labels[i]]
        if classe_reelle not in exemples_par_classe:
            score      = tf.nn.softmax(predictions[i])
            nom_predit = class_names[np.argmax(score)]
            confiance  = 100 * np.max(score)
            exemples_par_classe[classe_reelle] = (nom_predit, confiance)
        if len(exemples_par_classe) == len(class_names):
            break
    if len(exemples_par_classe) == len(class_names):
        break
 
for vrai_nom in sorted(exemples_par_classe.keys()):
    nom_predit, confiance = exemples_par_classe[vrai_nom]
    statut = "OK" if vrai_nom == nom_predit else "ERREUR"
    print(f"{statut}  Réel : {vrai_nom:<15} Prédit : {nom_predit:<15} ({confiance:.1f}%)")
 
print(f"\nModèle sauvegardé ici : {chemin_modele}")