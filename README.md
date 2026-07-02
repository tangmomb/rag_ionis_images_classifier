# Frame Filter: ajout montage vs contenu filme

Projet Python pour entrainer et utiliser un detecteur de frames a filtrer:

- `0` / `footage`: contenu filme a garder
- `1` / `graphic`: ajout graphique ou montage a filtrer

Exemples `graphic`: intro/outro, carton titre, logo centre, fond graphique, slide generee au montage.  
Exemples `footage`: logo filme sur ecran, slide filmee, personne, salle, batiment, contenu camera.

## Installation

Creer et activer le venv dedie:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pour CUDA sur RTX 2080, installez une version PyTorch compatible avec votre driver depuis les instructions officielles PyTorch si la roue par defaut n'inclut pas CUDA.

## Structure

```text
data/
  raw_videos/
  frames/
    initial_sort_via_script/
      footage/
      graphic/
    corrected_sort/
      footage/
      graphic/
models/
outputs/
  video_urls.txt
  extracted_frames/
  clusters/
src/
  data_prep/
    run_pipeline.py
    01_fetch_youtube_list.py
    02_download_youtube_videos.py
    03_extract_youtube_frames_ffmpeg.py
    04_cluster_extracted_frames.py
    youtube_pipeline.py
  model_training/
    train.py
    predict_video.py
    embeddings.py
    temporal_smoothing.py
    export_review_frames.py
```

## Workflow YouTube

### Pipeline complet

Pour lancer toute la preparation des donnees en une commande:

```powershell
python src/data_prep/run_pipeline.py --limit 20 --fps 1 --clusters 8
```

Ce pipeline execute les etapes `01` a `04`: liste YouTube via API officielle, telechargement, extraction ffmpeg, puis clustering des frames.
Si vous lancez `run_pipeline.py` sans `--limit`, `--all` ou `--video-url`, il pose une seule question: tapez `all`, un nombre de videos, ou un lien YouTube precis. Il demande ensuite le nombre de clusters prealables si `--clusters` n'est pas fourni.
Par defaut, chaque lancement efface le run precedent et repart de zero:

```text
outputs/video_urls.txt
data/raw_videos/
outputs/extracted_frames/
outputs/clusters/
data/frames/initial_sort_via_script/footage/
data/frames/initial_sort_via_script/graphic/
```

Options utiles:

```powershell
python src/data_prep/run_pipeline.py --all --fps 1 --clusters 8
python src/data_prep/run_pipeline.py --video-url https://www.youtube.com/watch?v=VIDEO_ID --fps 1 --clusters 8
python src/data_prep/run_pipeline.py --limit 20 --cluster-copy-mode hardlink --cluster-sample-every-n 2
python src/data_prep/run_pipeline.py --limit 20 --cluster-mixed-target graphic
python src/data_prep/run_pipeline.py --limit 20 --no-apply-labels
python src/data_prep/run_pipeline.py --limit 20 --no-clean
python src/data_prep/run_pipeline.py --skip-fetch --skip-download --fps 1 --clusters 12
```

### 1. Configurer l'API officielle YouTube

Copiez `.env.example` vers `.env` et renseignez:

```text
YOUTUBE_API_KEY=...
```

### 2. Recuperer la liste des videos de la chaine

Par defaut, le script cible `https://www.youtube.com/@IONIS-STM/videos`.

```powershell
python src/data_prep/01_fetch_youtube_list.py --limit 20
```

Pour une autre chaine:

```powershell
python src/data_prep/01_fetch_youtube_list.py --channel https://www.youtube.com/@NOM/videos --limit 20
```

La liste est ecrite dans:

```text
outputs/video_urls.txt
```

### 3. Telecharger les videos avec yt-dlp

```powershell
python src/data_prep/02_download_youtube_videos.py --limit 20
```

Les videos sont rangees directement dans:

```text
data/raw_videos/
```

### 4. Extraire 1 image par seconde avec ffmpeg

```powershell
python src/data_prep/03_extract_youtube_frames_ffmpeg.py --fps 1
```

Les frames extraites avant tri sont ecrites dans une zone temporaire hors des labels:

```text
outputs/extracted_frames/<video_id>/
```

### 5. Grouper les frames pour degrossir le tri

Avant l'annotation manuelle, vous pouvez creer des clusters visuels rapides avec des features OpenCV et k-means:

```powershell
python src/data_prep/04_cluster_extracted_frames.py --input outputs/extracted_frames --out outputs/clusters --clusters 8
```

Le script cree d'abord des copies de controle dans:

```text
outputs/clusters/<video_id>/
  graphic_candidate/
  footage_candidate/
  mixed/
```

Chaque cluster contient aussi un `contact_sheet.jpg` et un `manifest.json`.

Ensuite, par defaut, il deplace les frames sources vers:

```text
data/frames/initial_sort_via_script/footage/
data/frames/initial_sort_via_script/graphic/
```

Les clusters `mixed` sont envoyes dans `footage` par defaut. Vous pouvez changer ce comportement:

Options utiles:

```powershell
python src/data_prep/04_cluster_extracted_frames.py --clusters 12 --copy-mode hardlink --sample-every-n 2
python src/data_prep/04_cluster_extracted_frames.py --mixed-target graphic
python src/data_prep/04_cluster_extracted_frames.py --no-apply-labels
```

### 6. Annoter manuellement

Verifiez ensuite le premier tri et copiez/deplacez les images corrigees dans les dossiers d'entrainement:

```text
data/frames/corrected_sort/footage/
data/frames/corrected_sort/graphic/
```

### 7. Entrainer

Par defaut, `train.py` lit le jeu corrige `data/frames/corrected_sort`.

```powershell
python src/model_training/train.py --data data/frames/corrected_sort --model-out models/frame_filter.joblib --crop-bottom 0.20
```

## Extraction video optionnelle hors YouTube

Le projet inclut aussi un extracteur OpenCV generique si vous avez deja des fichiers video locaux.

```powershell
python src/model_training/extract_frames.py --videos chemin/vers/videos --out outputs/extracted_frames --fps 1
```

Options utiles:

```powershell
python src/model_training/extract_frames.py --videos chemin/vers/videos --out data/frames/initial_sort_via_script --fps 1 --sample-every-n 2 --class-dir footage
```

## Details entrainement

Le bas de l'image est ignore par defaut a `20%` pour reduire l'effet des sous-titres.
Le dossier d'entrainement par defaut est `data/frames/corrected_sort`.

```powershell
python src/model_training/train.py --data data/frames/corrected_sort --model-out models/frame_filter.joblib --crop-bottom 0.20
```

Le modele sauvegarde contient:

- le classifier sklearn
- le scaler
- les noms des backbones
- les parametres de crop
- le seuil recommande
- les metadonnees d'entrainement

Le classifieur par defaut est `LogisticRegression` avec ponderation de classe pour favoriser le rappel de `graphic`. Un MLP PyTorch est aussi disponible:

```powershell
python src/model_training/train.py --data data/frames/corrected_sort --classifier mlp --epochs 30 --batch-size 32
```

### Predire sur une video

```powershell
python src/model_training/predict_video.py --video input.mp4 --model models/frame_filter.joblib --out outputs/predictions.json --fps 2 --crop-bottom 0.20
```

Options utiles:

```powershell
python src/model_training/predict_video.py --video input.mp4 --model models/frame_filter.joblib --out outputs/predictions.json --fps 2 --threshold 0.45 --smoothing-window 9 --min-segment-duration 1.0 --merge-gap 0.75 --device cuda
```

Le JSON contient les predictions par frame, les predictions lissees et les segments temporels a filtrer.

## Hard-negative mining / review

Pendant l'inference, vous pouvez exporter les frames incertaines dans `outputs/review/`:

```powershell
python src/model_training/predict_video.py --video input.mp4 --model models/frame_filter.joblib --out outputs/predictions.json --export-review --review-dir outputs/review --uncertain-margin 0.10
```

Ou apres coup depuis un fichier de predictions:

```powershell
python src/model_training/export_review_frames.py --predictions outputs/predictions.json --out outputs/review --top-k 100
```

Si un fichier de labels est disponible, le script peut aussi exporter les faux positifs et faux negatifs:

```powershell
python src/model_training/export_review_frames.py --predictions outputs/predictions.json --labels labels.json --out outputs/review
```

Formats de labels acceptes:

```json
{"video_name": {"12.5": 1, "18.0": 0}}
```

ou:

```json
[{"time": 12.5, "label": 1}, {"time": 18.0, "label": 0}]
```

## Notes pratiques

- Les backbones DINOv2 ViT-B/14 et CLIP ViT-B/32 sont charges en mode frozen.
- CUDA est utilise si disponible, sinon CPU.
- Les embeddings d'images annotees sont caches dans `data/frames/.embedding_cache/`.
- Les embeddings de frames temporaires de prediction sont caches dans `outputs/cache/`.
- Augmentez le rappel `graphic` en baissant `--threshold` ou en entrainant avec plus d'exemples difficiles.
- Pour eviter les faux positifs isoles, utilisez le lissage temporel, `--min-segment-duration` et `--merge-gap`.
