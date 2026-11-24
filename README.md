# Segmentation d'anomalies agricoles (Agriculture-Vision)

Lab 2 du cours *AI for Agriculture & Environment* (aivancity, PGE5). Segmentation sémantique d'anomalies sur des images aériennes de champs (RGB + proche infrarouge), en PyTorch, avec des **prédictions écrites avant les expériences** puis comparées aux mesures.

**Rapport : [rapport_evaluation.pdf](rapport_evaluation.pdf)** · Carnet : [lab2_agriculture_vision.ipynb](lab2_agriculture_vision.ipynb) · Chiffres bruts : [results_summary.json](results_summary.json)

## Question

Sur un jeu de données très déséquilibré (le fond fait 65 % des pixels, l'anomalie la plus rare est 65 fois moins fréquente que la plus courante), qu'est-ce qui améliore vraiment la segmentation : la bande infrarouge, un indice de végétation, une autre fonction de perte, une architecture plus forte ? Et quelle part du score vient d'une évaluation trop favorable ?

## Ce qui a été fait

- Pipeline de données multi-canaux (RGB, NIR, masques de validité), 8 401 tuiles indexées, découpage **par champ** (6 150 / 1 124 / 1 127 tuiles pour entraînement, validation et test, sans champ commun entre les parties).
- IoU, Dice et mIoU codés à la main, sans scikit-learn.
- Grille de 8 configurations (C1 à C8), 3 graines chacune, un seul facteur modifié à la fois.
- Deux références : seuil sur le NDVI et forêt aléatoire.
- Analyse d'erreurs (8 cas types lus d'un point de vue agronomique), robustesse aux perturbations, calibration, règle de décision à coûts asymétriques.

## Résultats (mIoU de test, classes d'anomalies, moyenne sur 3 graines)

| Config | Architecture | NIR | NDVI | Perte | mIoU |
|---|---|---|---|---|---|
| C1 | U-Net | non | non | CE pondérée | 0,0517 ± 0,0046 |
| C2 | U-Net | oui | non | CE pondérée | 0,0620 ± 0,0092 |
| C3 | U-Net | oui | oui | CE pondérée | 0,0525 ± 0,0037 |
| C4 | U-Net | oui | oui | Dice | 0,0594 ± 0,0040 |
| C5 | U-Net | oui | non | Dice | 0,0559 ± 0,0055 |
| C6 | U-Net | oui | non | Focal | 0,0748 ± 0,0114 |
| C7 | U-Net | oui | non | Tversky | 0,0563 ± 0,0091 |
| C8 | DeepLabv3+ | oui | non | CE pondérée | **0,0886 ± 0,0259** |

Les valeurs absolues sont faibles : chaque cellule n'a été entraînée que 8 époques sur 600 tuiles. Ce qui compte ici, ce sont les comparaisons relatives.

## Ce que ça montre

| Prédiction écrite à l'avance | Verdict |
|---|---|
| La bande NIR améliore le mIoU de plus de 0,05 | Partiellement : +0,0103, dans le bruit des graines |
| L'indice NDVI en entrée améliore encore | Réfutée : -0,0095 (il se déduit déjà de R et NIR) |
| Les pertes Dice et Tversky battent la CE pondérée | Réfutée : seule la perte Focal gagne (+0,0128) |
| DeepLabv3+ est le meilleur modèle | Confirmée : +0,0266, mais avec la plus grande variance |
| Un découpage par tuile gonfle le score | Confirmée : 0,1038 contre 0,0620 par champ, soit +67 % |

**Calibration.** L'ECE est de 0,1605 : le modèle est trop confiant. Une règle qui garde seulement les pixels « sûrs » fait *baisser* la précision (0,410 contre 0,567 sur tous les pixels). Résultat négatif, expliqué par la calibration : il faut recalibrer avant de filtrer sur la confiance. Seuil de décision optimal avec un coût de 20 $ par faux négatif et 5 $ par faux positif : 0,65.

## Limites

- Classes rares (tempête, semis manqué, eau, voie d'eau, bord de champ, plantes doubles) : IoU proche de zéro, modèle inutilisable pour ces décisions.
- Un seul jeu de données, une seule région et saison : pas de garantie ailleurs.
- Le modèle dépend du proche infrarouge : il ne doit pas tourner sur des images RGB seules.
- Pas d'action automatique : validation humaine obligatoire.

## Reproduire

Le carnet a été exécuté sur Google Colab avec GPU. Les données (Agriculture-Vision-Lite, sous-ensemble documenté) ne sont pas incluses : elles sont distribuées par leurs auteurs (agriculture-vision.com) et leur licence s'applique. Placer l'archive sur Google Drive, ajuster `zip_path` dans la cellule de configuration, puis exécuter de haut en bas. Graines : 13, 42, 1337.

Référence : Chiu et al., *Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis*, CVPR 2020.
