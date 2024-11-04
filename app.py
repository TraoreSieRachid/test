# app.py
def predire_prix( superficie, nb_chambres ):
    # Implémentation fictive de la prédiction
    return superficie * 3000 + nb_chambres * 20000

if __name__ == "__main__":
    prix = predire_prix(100, 3)
    print(f"Le prix prédit est : {prix}")
