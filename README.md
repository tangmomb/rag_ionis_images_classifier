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

## Embeddings

Chaque image est transformée en deux vecteurs par des backbones gelés, donc non réentraînés :

| Backbone | Représentation utilisée | Dimension |
|---|---|---:|
| DINOv2 Base (`facebook/dinov2-base`) | token de classification de la dernière couche | 768 |
| CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | sortie visuelle projetée par `visual_projection` | 512 |

Les deux vecteurs sont normalisés séparément avec une norme L2, puis concaténés dans cet ordre :

```text
embedding(image) = concat(DINOv2[768], CLIP[512])
```

Chaque image est donc représentée par un unique vecteur `float32` de **1 280 dimensions**, utilisé comme entrée du classifieur.

## Entraînement

Pour entraîner un modèle avec le dossier par défaut :

```powershell
python src/model_training/01_train.py
```

Pour utiliser un autre jeu d'images :

```powershell
python src/model_training/01_train.py --data chemin/vers/les/images
```

Sans l'option `--classifier`, le script demande quel classifieur utiliser et propose `logreg` par défaut. Le choix peut aussi être fourni directement :

```powershell
python src/model_training/01_train.py --classifier logreg
python src/model_training/01_train.py --classifier mlp
```

### Régression logistique (`logreg`)

| Option | Valeur par défaut | Rôle |
|---|---:|---|
| `--max-iter` | `1000` | Nombre maximal d'itérations de l'optimiseur. |
| `--c` | `1.0` | Inverse de l'intensité de la régularisation : une valeur plus faible applique une régularisation plus forte. |

Exemple avec des valeurs personnalisées :

```powershell
python src/model_training/01_train.py --classifier logreg --max-iter 2000 --c 0.5
```

### Réseau de neurones (`mlp`)

| Option | Valeur par défaut | Rôle |
|---|---:|---|
| `--epochs` | `25` | Nombre de passages complets sur le jeu d'entraînement. |
| `--batch-size` | `16` | Nombre d'images traité avant chaque mise à jour des poids. |
| `--hidden-dim` | `256` | Nombre de neurones dans la couche cachée. |
| `--lr` | `0.001` | Taux d'apprentissage. |

Exemple avec des valeurs personnalisées :

```powershell
python src/model_training/01_train.py --classifier mlp --epochs 30 --batch-size 32 --hidden-dim 512 --lr 0.0005
```

L'option `--batch-size` contrôle également la taille des lots utilisés pour calculer les embeddings DINOv2 et CLIP, quel que soit le classifieur choisi.

Le script :

1. charge les images des trois classes ;
2. crée une répartition stratifiée `80 % / 10 % / 10 %` pour l'entraînement, la validation et le test ;
3. calcule et concatène les représentations DINOv2 et CLIP en un vecteur de 1 280 dimensions par image ;
4. entraîne le classifieur choisi ;
5. affiche les métriques pour les trois classes ;
6. enregistre le modèle et son journal dans `models/`.

Par défaut, le modèle est enregistré sous la forme :

```text
models/frame_filter_YYYY-MM-DD_HH-MM-SS.joblib
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
