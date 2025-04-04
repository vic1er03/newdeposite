# Dashboard d'Analyse des Campagnes de Don de Sang

Ce dashboard interactif présente une analyse approfondie des données de donneurs de sang, permettant d'optimiser les campagnes de collecte et d'améliorer la gestion des dons.

## Fonctionnalités

Le dashboard répond aux objectifs suivants :

1. **Distribution Géographique des Donneurs** : Visualisation de la répartition géographique des donneurs de sang basée sur leur lieu de résidence.

2. **Conditions de Santé & Éligibilité** : Analyse de l'impact des conditions de santé sur l'éligibilité au don de sang.

3. **Profils des Donneurs Idéaux** : Utilisation de techniques de clustering pour identifier des profils de donneurs similaires et déterminer les caractéristiques des donneurs idéaux.

4. **Efficacité des Campagnes** : Analyse des tendances temporelles et des facteurs qui influencent le succès des campagnes de don de sang.

5. **Fidélisation des Donneurs** : Étude de la fidélisation des donneurs en examinant la fréquence des dons et les facteurs qui influencent le retour des donneurs.

6. **Analyse des Retours** : Analyse des commentaires et retours des donneurs pour identifier les points forts et les axes d'amélioration.

7. **Prédiction d'Éligibilité** : Modèle de prédiction de l'éligibilité au don de sang basé sur les caractéristiques des donneurs.

## Installation

1. Cloner ce dépôt :
```
git clone <URL_DU_REPO>
cd dashboard-don-sang
```

2. Installer les dépendances :
```
pip install -r requirements.txt
```

3. Lancer l'application :
```
streamlit run app.py
```

## Structure des données

Le dashboard utilise trois jeux de données :
- `data_2019_pretraite.csv` : Données des donneurs de 2019
- `data_2020_pretraite.csv` : Données des donneurs de 2020
- `data_Volontaire_pretraite.csv` : Données des donneurs volontaires

## Déploiement

Ce dashboard peut être déployé sur différentes plateformes :

### Déploiement sur Streamlit Cloud

1. Créer un compte sur [Streamlit Cloud](https://streamlit.io/cloud)
2. Connecter votre dépôt GitHub
3. Déployer l'application en sélectionnant le fichier `app.py`

### Déploiement sur Heroku

1. Créer un compte sur [Heroku](https://www.heroku.com/)
2. Installer Heroku CLI et se connecter
3. Créer une application Heroku :
```
heroku create nom-de-votre-app
```
4. Déployer l'application :
```
git push heroku main
```

## Utilisation

Le dashboard est organisé en sections accessibles via la barre latérale. Chaque section correspond à l'un des objectifs d'analyse mentionnés ci-dessus.

Pour utiliser le modèle de prédiction d'éligibilité, accédez à la section "7. Prédiction d'Éligibilité" et remplissez le formulaire avec les caractéristiques du donneur potentiel.

## Auteur

Ce dashboard a été créé dans le cadre du challenge d'analyse des données de campagnes de don de sang.

## Licence

Ce projet est sous licence MIT.
