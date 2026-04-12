import tensorflow as tf
import numpy as np
import os
import sys

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
chemin_modele  = os.path.join(dossier_actuel, 'mon_pokedex.keras')

# Toutes les classes possibles (doit correspondre à l'ordre alphabétique du dataset)
# Starter seuls  → ['Bulbasaur', 'Charmander', 'Squirtle']
# Avec évolutions → ajoutez les noms en respectant l'ordre alphabétique
CLASS_NAMES = [
    'Blastoise',   # B avant C
    'Bulbasaur',
    'Charizard',
    'Charmander',
    'Charmeleon',
    'Ivysaur',
    'Squirtle',
    'Venusaur',
    'Wartortle',
]

HAUTEUR = 128
LARGEUR = 128

# ─────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────
if not os.path.exists(chemin_modele):
    print(f"ERREUR : Modèle introuvable → {chemin_modele}")
    print("Lancez d'abord entrainement.py pour créer le modèle.")
    sys.exit(1)

print(f"Chargement du modèle...")
model = tf.keras.models.load_model(chemin_modele)
print("Modèle chargé avec succès !\n")

# ─────────────────────────────────────────────
#  FONCTION DE PRÉDICTION
# ─────────────────────────────────────────────
def predire(chemin_image: str) -> None:
    """Analyse une image et affiche le Pokémon reconnu avec le top-3."""
    if not os.path.exists(chemin_image):
        print(f"ERREUR : image introuvable → '{chemin_image}'")
        return

    # Chargement et prétraitement
    img       = tf.keras.utils.load_img(chemin_image, target_size=(HAUTEUR, LARGEUR))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)   # batch de 1

    # Prédiction
    predictions = model.predict(img_array, verbose=0)
    scores      = tf.nn.softmax(predictions[0]).numpy()

    # Résultat principal
    idx_max      = np.argmax(scores)
    pokemon_top1 = CLASS_NAMES[idx_max]
    confiance    = 100 * scores[idx_max]

    print(f"{'─'*45}")
    print(f"  Image analysée : {os.path.basename(chemin_image)}")
    print(f"{'─'*45}")
    print(f"  🏆  {pokemon_top1:<15}  →  {confiance:.1f} %")

    # Top-3
    top3_idx = np.argsort(scores)[::-1][:3]
    print(f"\n  Classement complet :")
    medailles = ["🥇", "🥈", "🥉"]
    for rang, i in enumerate(top3_idx):
        print(f"    {medailles[rang]}  {CLASS_NAMES[i]:<15} {100*scores[i]:5.1f} %")
    print(f"{'─'*45}\n")

# ─────────────────────────────────────────────
#  UTILISATION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Mode ligne de commande : python prediction.py mon_image.jpg ──
    if len(sys.argv) > 1:
        for fichier in sys.argv[1:]:
            chemin = os.path.join(dossier_actuel, fichier)
            predire(chemin)

    # ── Mode interactif (aucun argument) ──
    else:
        print("=== PokéDex – Reconnaissance de Pokémon ===\n")
        print("Entrez le nom du fichier image (ex: pikachu.jpg)")
        print("Tapez 'quitter' pour terminer.\n")
        while True:
            nom = input("Image > ").strip()
            if nom.lower() in ("quitter", "q", "exit"):
                print("Au revoir !")
                break
            chemin = os.path.join(dossier_actuel, nom)
            predire(chemin)