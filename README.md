# Classifieur d'images en trois classes

Ce projet entraîne un modèle à partir d'un dossier d'images déjà réparties entre trois classes :

- `0` / `footage` : prise de vue réelle ;
- `1` / `graphic` : visuel entièrement graphique ;
- `2` / `mixture` : prise de vue réelle contenant des éléments graphiques importants.

## Exemples

| `footage` | `graphic` | `mixture` |
|:---:|:---:|:---:|
| ![Exemple de la classe footage](docs/readme/footage.jpg) | ![Exemple de la classe graphic](docs/readme/graphic.jpg) | ![Exemple de la classe mixture](docs/readme/mixture.jpg) |

## Organisation du jeu de données

L'utilisateur fournit directement le dossier suivant :

```text
data/
  frames/
    dataset/
      footage/
      graphic/
      mixture/
```

Le dossier contenant une image définit sa classe. Les formats acceptés sont JPEG, PNG, BMP et WebP. Chaque classe doit contenir au moins dix images.

Un autre emplacement peut être utilisé, à condition de conserver exactement les trois sous-dossiers `footage`, `graphic` et `mixture`.

## Entraînement

Pour entraîner un modèle avec le dossier par défaut :

```powershell
python src/model_training/01_train.py
```

Pour utiliser un autre jeu d'images :

```powershell
python src/model_training/01_train.py --data chemin/vers/les/images
```

Le classifieur peut être choisi explicitement :

```powershell
python src/model_training/01_train.py --classifier logreg
python src/model_training/01_train.py --classifier mlp --epochs 30 --batch-size 32
```

Le script :

1. charge les images des trois classes ;
2. crée une répartition stratifiée `80 % / 10 % / 10 %` pour l'entraînement, la validation et le test ;
3. calcule les représentations DINOv2 et CLIP des images ;
4. entraîne le classifieur choisi ;
5. affiche les métriques pour les trois classes ;
6. enregistre le modèle et son journal dans `models/`.

Par défaut, le modèle est enregistré sous la forme :

```text
models/image_classifier_YYYY-MM-DD_HH-MM-SS.joblib
```

## Structure du projet

```text
data/frames/dataset/            # images classées fournies par l'utilisateur
docs/readme/                    # exemples des trois classes
models/                         # modèles et journaux d'entraînement
src/model_training/
  01_train.py                   # entraînement
  embeddings.py                 # représentations DINOv2 et CLIP
  mlp_classifier.py             # classifieur MLP optionnel
src/utils.py                    # fonctions communes
```
