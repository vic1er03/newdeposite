"""Tableau de bord interactif pour l'analyse des données de donneurs de sang.
Ce script crée un tableau de bord Streamlit avec des visualisations innovantes
pour répondre aux objectifs du concours de data visualisation.
"""

import streamlit as st
import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import folium_static
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import nltk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
from PIL import Image
import streamlit.components.v1 as components

#import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer

from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.metrics import roc_curve, auc, classification_report,confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import  f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
# Créer un pipeline pour le prétraitement et la modélisation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
import base64
import warnings
import io
import json
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.graphics.mosaicplot import mosaic
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import calendar

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
nltk.download('punkt_tab')
nltk.download('wordnet')

    


def paginate_dataframe(dataframe, page_size=10):
    """Ajoute une pagination à un DataFrame pour améliorer les performances."""
    # Obtenir le nombre total de pages
    n_pages = len(dataframe) // page_size + (1 if len(dataframe) % page_size > 0 else 0)
    
    # Ajouter un sélecteur de page
    page = st.selectbox('Page', range(1, n_pages + 1), 1)
    
    # Afficher la page sélectionnée
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(dataframe))
    
    return dataframe.iloc[start_idx:end_idx]


# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse des Donneurs de Sang",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Définir les chemins des fichiers
data_2019_path = "data_2019_preprocessed.csv"
data_volontaire_path = "data_volontaire_preprocessed.csv"
df_2020 = pd.read_csv("data_2020_pretraite.csv")
df_volontaire = pd.read_csv(data_volontaire_path)
df=df_volontaire.copy()
df_volontaires=df_volontaire.copy()
#analysis_results_dir = r"C:\Users\Ultra Tech\Desktop\analysis_results"
#model_path = r"C:\Users\Ultra Tech\Desktop\preprocessor_eligibility.pkl"

# Fonction pour charger les données
@st.cache_data
@st.cache_data(ttl=3600, max_entries=2)


def load_data():
    """
    #Charge les données prétraitées à partir des fichiers CSV.
    """
    data_2019_path = "data_2019_preprocessed.csv"
    data_2020_path = "data_2020_pretraite.csv"
    data_volontaire_path = "data_volontaire_preprocessed.csv"

    df_2019 = pd.read_csv(data_2019_path)
    df_2020 = pd.read_csv(data_2020_path)
    df_volontaire = pd.read_csv(data_volontaire_path)

    

    # Convertir les colonnes de dates au format datetime
    date_columns = [col for col in df_2019.columns if 'date' in col.lower()]
    for col in date_columns:
        if col in df_2019.columns:
            try:
                df_2019[col] = pd.to_datetime(df_2019[col], errors='coerce')
            except:
                pass
    
    date_columns = [col for col in df_volontaire.columns if 'date' in col.lower()]
    for col in date_columns:
        if col in df_volontaire.columns:
            try:
                df_volontaire[col] = pd.to_datetime(df_volontaire[col], errors='coerce')
            except:
                pass
    
    return df_2019, df_2020, df_volontaire

@st.cache_data

# Fonction pour charger le cache de géocodage
def load_geo_cache():
    cache_file = "geo_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}

# Fonction pour sauvegarder le cache de géocodage
def save_geo_cache(geo_cache):
    cache_file = "geo_cache.json"
    with open(cache_file, "w") as f:
        json.dump(geo_cache, f)

# Fonction pour obtenir les coordonnées avec cache
def get_coordinates_with_cache(row, geo_cache, geolocator, geocode):
    modalites_indesirables = [
        "Pas precise", "Pas précisé", "Pas precise", "Pas précise", "Non précisé", "non précisé",
        " RAS", "Ras", "R A S ", "ras", "r a s", " ", "nan",
        " Pas mentionné", "Pasmentionné", "Pas mentionne"
    ]
    
    # Vérifier si les colonnes nécessaires existent
    quartier = row.get('Quartier', '')
    arrondissement = row.get('Arrondissement', '')
    
    # Construire la requête
    query = f"{quartier}, {arrondissement}, Douala, Cameroun"
    
    # Vérifier si l'adresse est déjà dans le cache
    if query in geo_cache:
        return geo_cache[query]['latitude'], geo_cache[query]['longitude']
    
    # Vérifier si les données sont valides avant d'envoyer la requête
    if pd.isna(quartier) or pd.isna(arrondissement) or quartier in modalites_indesirables:
        geo_cache[query] = {'latitude': 4.0483, 'longitude': 9.7043}  # Coordonnées par défaut de Douala
        return 4.0483, 9.7043
    
    try:
        location = geocode(query)
        if location:
            geo_cache[query] = {'latitude': location.latitude, 'longitude': location.longitude}
            return location.latitude, location.longitude
        else:
            geo_cache[query] = {'latitude': 4.0483, 'longitude': 9.7043}  # Valeur par défaut
            return 4.0483, 9.7043
    except Exception as e:
        st.warning(f"⚠️ Erreur pour {query}: {e}")
        geo_cache[query] = {'latitude': 4.0483, 'longitude': 9.7043}  # Valeur par défaut
        return 4.0483, 9.7043

# Fonction pour créer la carte Folium
def create_map(df):
    # Créer une carte sans tuiles (fond blanc)
    m = folium.Map(location=[4.0483, 9.7043], zoom_start=12, tiles=None)
    
    # Ajouter un fond blanc personnalisé (carte vide)
    folium.raster_layers.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB',
        name='Blanc',
        control=False,
        overlay=False
    ).add_to(m)
    
    # Cluster de marqueurs
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, row in df.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue
        
        popup_content = f"""
        <b>{row.get('Genre', 'N/A')}, {row.get('Age', 'N/A')} ans</b><br>
        <i>{row.get('Profession', 'N/A')}</i><br>
        Éligible: {row.get('Eligibilite_Don', 'N/A')}<br>
        Quartier: {row.get('Quartier', 'N/A')}<br>
        Arrondissement: {row.get('Arrondissement', 'N/A')}
        """
        
        eligibilite = str(row.get('Eligibilite_Don', '')).lower()
        color = 'green' if eligibilite == 'oui' else 'red' if eligibilite == 'non' else 'blue'
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=250),
            icon=folium.Icon(color=color, icon='tint', prefix='fa')
        ).add_to(marker_cluster)
    
    return m


# Fonction pour créer une carte de distribution géographique
@st.cache_data
def create_geo_map(df, location_column, color_column=None, zoom_start=10):
    """
    Crée une carte interactive montrant la distribution géographique des donneurs.
    Utilise les coordonnées chargées depuis le fichier JSON.
    """
    # Charger les coordonnées géographiques
    geo_data = load_geo_cache()
    
    # Déterminer le type de localisation (arrondissement, quartier, ville)
    if "Arrondissement" in location_column.lower():
        coords_dict = geo_data["Arrondissement"]
        location_type = "Arrondissement"
    elif "Quartier" in location_column.lower():
        coords_dict = geo_data["Quartier"]
        location_type = "Quartier"
    else:
        # Coordonnées par défaut pour Douala, Cameroun
        coords_dict = {}
        location_type = "Localisation"
        
    # Coordonnées par défaut pour Douala, Cameroun
    default_coords = [4.0511, 9.7679]
    
    # Créer une carte centrée sur Douala
    m = folium.Map(location=default_coords, zoom_start=zoom_start, tiles="OpenStreetMap")
    folium.raster_layers.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB',
        name='Blanc',
        control=False,
        overlay=False
    ).add_to(m)
    # Créer un cluster de marqueurs pour améliorer les performances
    marker_cluster = MarkerCluster().add_to(m)
    
    # Ajouter des marqueurs pour chaque localisation
    locations_count = df[location_column].value_counts().to_dict()
    
    for location, count in locations_count.items():
        if location in coords_dict:
            coords = coords_dict[location]
            
            # Créer le texte de popup
            popup_text = f"<strong>{location}</strong><br>Nombre de donneurs: {count}"
            
            # Ajouter des informations supplémentaires si une colonne de couleur est spécifiée
            if color_column and color_column in df.columns:
                color_values = df[df[location_column] == location][color_column].value_counts()
                popup_text += "<br><br>Distribution par " + color_column + ":<br>"
                for val, val_count in color_values.items():
                    popup_text += f"- {val}: {val_count} ({val_count/count*100:.1f}%)<br>"
            
            # Ajouter le marqueur au cluster
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(icon="info-sign", prefix='fa', color='red' if count > 50 else 'blue')
            ).add_to(marker_cluster)
    
    # Ajouter une couche de chaleur pour visualiser la densité
    heat_data = []
    for location, count in locations_count.items():
        if location in coords_dict:
            coords = coords_dict[location]
            # Répéter les coordonnées en fonction du nombre de donneurs pour créer l'effet de chaleur
            for _ in range(min(count, 100)):  # Limiter à 100 pour les performances
                heat_data.append(coords)
    
    # Ajouter la couche de chaleur si des données sont disponibles
    if heat_data:
        HeatMap(heat_data, radius=15).add_to(m)
    
    return m



# Fonction pour créer un graphique de santé et éligibilité
@st.cache_data
def analyze_categorical_relationships(df, sheet_name):
    st.subheader(f"Analyse des relations entre variables catégorielles – {sheet_name}")
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    valid_columns = [col for col in categorical_columns if 1 < df[col].nunique() <= 10]

    if len(valid_columns) >= 2:
        target_column = None
        for potential_target in ['ÉLIGIBILITÉ_AU_DON.', 'Éligibilité_au_don', 'Eligibilité']:
            if potential_target in valid_columns:
                target_column = potential_target
                break

        if target_column:
            st.info(f"Variable cible détectée : **{target_column}**")
            for col in valid_columns:
                if col != target_column:
                    contingency = pd.crosstab(df[target_column], df[col])
                    chi2, p, dof, expected = stats.chi2_contingency(contingency)
                    association = "✅ significative" if p < 0.05 else "❌ non significative"
                    
                    st.markdown(f"**Relation entre `{target_column}` et `{col}`**")
                    st.write(f"Test du chi2 : χ² = {chi2:.2f}, p = {p:.4f} → {association}")
                    
                    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                    fig = px.bar(contingency_pct, 
                                 barmode='stack',
                                 title=f"{target_column} vs {col} (p={p:.4f})",
                                 labels={'value': 'Pourcentage (%)', 'index': target_column},
                                text=contingency_pct['value'])
                    
                    # Afficher automatiquement les textes sur les barres
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                    
                    fig.update_layout(template='plotly_white')
                    st.plotly_chart(fig)

                    with st.expander(f"Graphique en mosaïque pour {target_column} vs {col}"):
                        fig_mosaic, ax = plt.subplots(figsize=(8, 6))
                        mosaic_data = {(i, j): contingency.loc[i, j] for i in contingency.index for j in contingency.columns}
                        mosaic(mosaic_data, ax=ax, title=f'{target_column} vs {col}')
                        st.pyplot(fig_mosaic)

        else:
            st.warning("Aucune variable cible claire trouvée. Affichage de quelques relations aléatoires entre variables.")
            for col1, col2 in zip(valid_columns, valid_columns[1:3]):
                contingency = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                association = "✅ significative" if p < 0.05 else "❌ non significative"

                st.markdown(f"**Relation entre `{col1}` et `{col2}`**")
                st.write(f"Test du chi2 : χ² = {chi2:.2f}, p = {p:.4f} → {association}")

                fig = px.imshow(contingency,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="Viridis")
                fig.update_layout(title=f"{col1} vs {col2} (p={p:.4f})")
                st.plotly_chart(fig)
    else:
        st.warning("Pas assez de variables catégorielles valides pour analyser les relations.")


# Fonction pour créer un graphique de clustering des donneurs
@st.cache_data
def create_donor_clustering(df):
    """
    Crée une visualisation interactive des clusters de donneurs.
    """
    # Sélectionner les variables numériques pour le clustering
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Afficher les informations sur les colonnes numériques pour le débogage
    print(f"Colonnes numériques disponibles: {numeric_df.columns.tolist()}")
    print(f"Nombre de colonnes numériques: {numeric_df.shape[1]}")
    
    # Sélectionner uniquement les colonnes sans valeurs manquantes ou avec peu de valeurs manquantes
    # pour éviter les problèmes de dimensionnalité
    threshold = 0.5  # Colonnes avec moins de 50% de valeurs manquantes
    numeric_df = numeric_df.loc[:, numeric_df.isnull().mean() < threshold]
    print(f"Colonnes numériques après filtrage: {numeric_df.columns.tolist()}")
    
    if numeric_df.shape[1] >= 2:
        # Gérer les valeurs manquantes avant le clustering
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        
        # Imputer les valeurs manquantes
        imputed_values = imputer.fit_transform(numeric_df)
        
        # Créer un nouveau DataFrame avec les valeurs imputées
        numeric_df_imputed = pd.DataFrame(
            imputed_values,
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        
        # Standardiser les données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df_imputed)
        
        # Appliquer K-means avec 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Ajouter les clusters au DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = clusters
        
        # Appliquer PCA pour la visualisation
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Créer un DataFrame pour la visualisation
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': clusters.astype(str)
        })
        
        # Ajouter des informations supplémentaires si disponibles
        if 'Age' in df.columns:
            pca_df['Age'] = df['Age'].values
        if 'Genre_' in df.columns:
            pca_df['Genre'] = df['Genre_'].values
        if 'Niveau_d\'etude' in df.columns:
            pca_df['Niveau_d\'études'] = df['Niveau_d\'etude'].values
        
        # Créer un graphique interactif avec Plotly
        fig = px.scatter(pca_df,
            render_mode='auto', 
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data=pca_df.columns,
            title="Clustering des donneurs (PCA)",
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} de variance expliquée)',
                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} de variance expliquée)'},
            color_discrete_sequence=px.colors.qualitative.Bold)
        
        # Personnaliser le graphique
        fig.update_layout(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} de variance expliquée)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} de variance expliquée)',
            legend_title="Cluster",
            font=dict(size=12),
            height=600)
        
        # Analyser les caractéristiques de chaque cluster
        cluster_stats = df_with_clusters.groupby('Cluster').agg({
            col: ['mean', 'std'] for col in numeric_df.columns
        })
        
        return fig, cluster_stats, df_with_clusters
    else:
        return None, None, df

# Fonction pour créer un graphique d'analyse de campagne
@st.cache_data
def create_campaign_analysis(df):
    """
    Crée des visualisations pour analyser l'efficacité des campagnes de don.
    """
    # Identifier les colonnes de date
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    # Sélectionner une colonne de date appropriée
    selected_date_col = None
    for col in date_columns:
        if df[col].notna().sum() > 100:  # Vérifier qu'il y a suffisamment de données
            selected_date_col = col
            break
    
    if selected_date_col:
        # Créer une copie du DataFrame avec la colonne de date
        df_temp = df.copy()
        
        # Convertir en datetime si ce n'est pas déjà fait
        df_temp[selected_date_col] = pd.to_datetime(df_temp[selected_date_col], errors='coerce')
        
        # Extraire l'année et le mois
        df_temp['year'] = df_temp[selected_date_col].dt.year
        df_temp['month'] = df_temp[selected_date_col].dt.month
        df_temp['year_month'] = df_temp[selected_date_col].dt.to_period('M')
        
        # Compter le nombre de donneurs par mois
        monthly_counts = df_temp.groupby('year_month').size().reset_index(name='count')
        monthly_counts['year_month_str'] = monthly_counts['year_month'].astype(str)
        
        # Créer un graphique de tendance temporelle
        fig1 = px.line(
            monthly_counts,
            x='year_month_str',
            y='count',
            markers=True,
            title=f"Évolution du nombre de donneurs au fil du temps (basé sur {selected_date_col})",
            labels={'count': 'Nombre de donneurs', 'year_month_str': 'Année-Mois'}
        )
        
        # Personnaliser le graphique
        fig1.update_layout(
            xaxis_title="Période",
            yaxis_title="Nombre de donneurs",
            font=dict(size=12),
            height=500
        )
        
        # Analyser les tendances par caractéristiques démographiques
        demographic_figs = []
        
        # Analyser par genre si disponible
        if 'Genre_' in df_temp.columns:
            gender_monthly = df_temp.groupby(['year_month', 'Genre_']).size().reset_index(name='count')
            gender_monthly['year_month_str'] = gender_monthly['year_month'].astype(str)
            
            fig_gender = px.line(
                gender_monthly,
                x='year_month_str',
                y='count',
                color='Genre_',
                markers=True,
                title="Évolution du nombre de donneurs par genre",
                labels={'count': 'Nombre de donneurs', 'year_month_str': 'Année-Mois', 'Genre_': 'Genre'}
            )
            
            fig_gender.update_layout(
                xaxis_title="Période",
                yaxis_title="Nombre de donneurs",
                legend_title="Genre",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_gender)
        
        # Analyser par niveau d'études si disponible
        if 'Niveau_d\'etude' in df_temp.columns:
            # Simplifier les catégories pour une meilleure lisibilité
            df_temp['Niveau_simplifié'] = df_temp['Niveau_d\'etude'].apply(
                lambda x: 'Universitaire' if 'Universitaire' in str(x) 
                else ('Secondaire' if 'Secondaire' in str(x)
                     else ('Primaire' if 'Primaire' in str(x)
                          else ('Aucun' if 'Aucun' in str(x) else 'Non précisé')))
            )
            
            edu_monthly = df_temp.groupby(['year_month', 'Niveau_simplifié']).size().reset_index(name='count')
            edu_monthly['year_month_str'] = edu_monthly['year_month'].astype(str)
            
            fig_edu = px.line(
                edu_monthly,
                x='year_month_str',
                y='count',
                color='Niveau_simplifié',
                markers=True,
                title="Évolution du nombre de donneurs par niveau d'études",
                labels={'count': 'Nombre de donneurs', 'year_month_str': 'Année-Mois', 'Niveau_simplifié': "Niveau d'études"}
            )
            
            fig_edu.update_layout(
                xaxis_title="Période",
                yaxis_title="Nombre de donneurs",
                legend_title="Niveau d'études",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_edu)
        
        return fig1, demographic_figs
    else:
        return None, []

# Fonction pour créer une analyse de fidélisation des donneurs
@st.cache_data
def create_donor_retention_analysis(df):
    """
    Crée des visualisations pour analyser la fidélisation des donneurs.
    """
    # Vérifier si la colonne indiquant si le donneur a déjà donné est disponible
    if 'A-t-il_(elle)_déjà_donné_le_sang_' in df.columns:
        # Compter le nombre de donneurs qui ont déjà donné et ceux qui n'ont pas donné
        retention_counts = df['A-t-il_(elle)_déjà_donné_le_sang_'].value_counts().reset_index()
        retention_counts.columns = ['Statut', 'Nombre']
        
        # Créer un graphique circulaire
        fig1 = px.pie(
            retention_counts,
            values='Nombre',
            names='Statut',
            title="Proportion de donneurs fidélisés",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig1.update_layout(
            font=dict(size=12),
            height=400
        )
        
        # Analyser la fidélisation par caractéristiques démographiques
        demographic_figs = []
        
        # Analyser par genre si disponible
        if 'Genre_' in df.columns:
            gender_retention = pd.crosstab(df['Genre_'], df['A-t-il_(elle)_déjà_donné_le_sang_'])
            gender_retention_pct = gender_retention.div(gender_retention.sum(axis=1), axis=0) * 100
            
            # Convertir en format long pour Plotly
            gender_retention_long = gender_retention_pct.reset_index().melt(
                id_vars='Genre_',
                var_name='Statut',
                value_name='Pourcentage'
            )
            
            fig_gender = px.bar(
                gender_retention_long,
                x='Genre_',
                y='Pourcentage',
                color='Statut',
                barmode='group',
                title="Fidélisation des donneurs par genre",
                labels={'Pourcentage': 'Pourcentage (%)', 'Genre_': 'Genre', 'Statut': 'A déjà donné'}
            )
            
            fig_gender.update_layout(
                xaxis_title="Genre",
                yaxis_title="Pourcentage (%)",
                legend_title="A déjà donné",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_gender)
        
        # Analyser par niveau d'études si disponible
        if 'Niveau_d\'etude' in df.columns:
            # Simplifier les catégories pour une meilleure lisibilité
            df['Niveau_simplifié'] = df['Niveau_d\'etude'].apply(
                lambda x: 'Universitaire' if 'Universitaire' in str(x) 
                else ('Secondaire' if 'Secondaire' in str(x)
                     else ('Primaire' if 'Primaire' in str(x)
                          else ('Aucun' if 'Aucun' in str(x) else 'Non précisé')))
            )
            
            edu_retention = pd.crosstab(df['Niveau_simplifié'], df['A-t-il_(elle)_déjà_donné_le_sang_'])
            edu_retention_pct = edu_retention.div(edu_retention.sum(axis=1), axis=0) * 100
            
            # Convertir en format long pour Plotly
            edu_retention_long = edu_retention_pct.reset_index().melt(
                id_vars='Niveau_simplifié',
                var_name='Statut',
                value_name='Pourcentage'
            )
            
            fig_edu = px.bar(
                edu_retention_long,
                x='Niveau_simplifié',
                y='Pourcentage',
                color='Statut',
                barmode='group',
                title="Fidélisation des donneurs par niveau d'études",
                labels={'Pourcentage': 'Pourcentage (%)', 'Niveau_simplifié': "Niveau d'études", 'Statut': 'A déjà donné'}
            )
            
            fig_edu.update_layout(
                xaxis_title="Niveau d'études",
                yaxis_title="Pourcentage (%)",
                legend_title="A déjà donné",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_edu)
        
        # Analyser par âge si disponible
        if 'Age' in df.columns:
            # Créer des tranches d'âge
            df['Tranche_âge'] = pd.cut(
                df['Age'],
                bins=[0, 18, 25, 35, 45, 55, 100],
                labels=['<18', '18-25', '26-35', '36-45', '46-55', '>55']
            )
            
            age_retention = pd.crosstab(df['Tranche_âge'], df['A-t-il_(elle)_déjà_donné_le_sang_'])
            age_retention_pct = age_retention.div(age_retention.sum(axis=1), axis=0) * 100
            
            # Convertir en format long pour Plotly
            age_retention_long = age_retention_pct.reset_index().melt(
                id_vars='Tranche_âge',
                var_name='Statut',
                value_name='Pourcentage'
            )
            
            fig_age = px.bar(
                age_retention_long,
                x='Tranche_âge',
                y='Pourcentage',
                color='Statut',
                barmode='group',
                title="Fidélisation des donneurs par tranche d'âge",
                labels={'Pourcentage': 'Pourcentage (%)', 'Tranche_âge': "Tranche d'âge", 'Statut': 'A déjà donné'}
            )
            
            fig_age.update_layout(
                xaxis_title="Tranche d'âge",
                yaxis_title="Pourcentage (%)",
                legend_title="A déjà donné",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_age)
        
        return fig1, demographic_figs
    else:
        return None, []

# Fonction pour créer une analyse de sentiment
@st.cache_data
def create_sentiment_analysis(df):
    """
    Crée des visualisations pour l'analyse de sentiment des commentaires des donneurs (en français).
    """
    # Détecter les colonnes textuelles susceptibles d'être des commentaires
    comment_columns = [col for col in df.columns if any(term in col.lower() for term in 
                      ['préciser','autre', 'commentaire', 'feedback'])]

    if not comment_columns:
        st.warning("Aucune colonne de commentaires trouvée.")
        return None, None, None

    selected_col = comment_columns[0]
    comments_df = df[df[selected_col].notna() & (df[selected_col].astype(str).str.strip() != '')].copy()
    
    if comments_df.empty:
        st.warning("Aucun commentaire exploitable pour l'analyse.")
        return None, None, None

    st.subheader("💬 Analyse de sentiment des commentaires (en français)")

    # Initialisation de TextBlob-FR
    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

    # Analyse de sentiment
    def analyze_sentiment(text):
        try:
            blob = tb(str(text))
            polarite = blob.sentiment[0]
            # Catégorisation
            if polarite > 0.01:
                return "Positif"
            elif polarite < -0.01:
                return "Négatif"
            else:
                return "Neutre"
        except:
            return "Indéfini"

    comments_df['Sentiment'] = comments_df[selected_col].apply(analyze_sentiment)
    def analyze_sentiments(text):
        try:
            blob = tb(str(text))
            polarite = blob.sentiment[0]
            return polarite
        except:
            return 'pass'
            
    comments_df['Score'] = comments_df[selected_col].apply(analyze_sentiments)
    # 🔢 Compter les sentiments
    sentiment_counts = comments_df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Nombre']

    # 📊 Graphique circulaire
    fig1 = px.pie(sentiment_counts,
                  values='Nombre',
                  names='Sentiment',
                  title="Répartition des sentiments",
                  color_discrete_sequence=px.colors.qualitative.Set3)

    st.plotly_chart(fig1)

    # ☁️ Nuage de mots
    all_comments = ' '.join(comments_df[selected_col].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_comments)

    st.subheader("☁️ Nuage de mots des commentaires")
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig2)

    # 📋 Afficher le tableau avec commentaires et sentiment
    st.subheader("🧾 Détail des commentaires analysés")
    st.dataframe(comments_df[[selected_col, 'Sentiment','Score']])

    return fig1, wordcloud, comments_df

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    return st.markdown(bg_image, unsafe_allow_html=True)

def Apercue(df_2019):
            st.dataframe(df_2019.head())
            
            # Afficher les statistiques descriptives
            st.subheader("Statistiques descriptives")
            st.dataframe(df_2019.describe())
            
            # Visualiser les valeurs manquantes
            st.subheader("Valeurs manquantes")
            missing = df_2019.isnull().sum().sort_values(ascending=False)
            missing = missing[missing > 0]
            if len(missing) > 0:
                missing_percent = (missing / len(df_2019) * 100).round(2)
                missing_df = pd.DataFrame({'Nombre': missing, 'Pourcentage (%)': missing_percent})
                fig = px.bar(missing_df, x=missing_df.index, y='Pourcentage (%)', 
                            text='Nombre',
                            title='Valeurs manquantes - Données 2019',
                            color='Pourcentage (%)',
                            color_continuous_scale='Viridis')
                fig.update_layout(xaxis_title='Variables', yaxis_title='Pourcentage (%)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune valeur manquante dans les données 2019.")
        
# Visualisation des valeurs manquantes
def visualize_missing_values(df, title):
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing_percent = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Nombre de valeurs manquantes': missing, 'Pourcentage (%)': missing_percent})

    if len(missing_df) > 0:
        fig = px.bar(
            missing_df,
            x=missing_df.index,
            y='Pourcentage (%)',
            title=f'Valeurs manquantes - {title}',
            text='Nombre de valeurs manquantes',
            color='Pourcentage (%)',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title='Variables',
            yaxis_title='Pourcentage de valeurs manquantes (%)',
            xaxis={'categoryorder': 'total descending'},
            height=500
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        #st.plotly_chart(fig, use_container_width=True)
    else:
        st.success(f"Aucune valeur manquante dans le jeu de données : {title}")





def analyze_distributions(df, sheet_name):
    """
    Analyse les distributions des variables avec visualisation intégrable à un dashboard Streamlit.
    """
    st.header(f"Analyse des distributions - {sheet_name}")

    # Analyser les variables numériques
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numeric_columns) > 0:
        st.subheader("Variables numériques")
        selected_numeric = list(numeric_columns)[:min(5, len(numeric_columns))]

        for col in selected_numeric:
            col1, col2 = st.columns(2)

            # Histogramme avec KDE
            fig1, ax1 = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax1, color='steelblue', alpha=0.7)
            ax1.set_title(f'Distribution de {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Fréquence')

            # Test de normalité
            stat, p_value = stats.shapiro(df[col].dropna())
            normality = "normale" if p_value > 0.05 else "non normale"
            ax1.annotate(f'p = {p_value:.4f}\nDistribution {normality}',
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='top')

            with col1:
                st.pyplot(fig1)

            # Boxplot
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[col].dropna(), ax=ax2, color='lightseagreen')
            ax2.set_title(f'Boxplot de {col}')
            ax2.set_xlabel(col)

            # Statistiques descriptives
            stats_desc = df[col].describe()
            stats_text = (f"Moyenne: {stats_desc['mean']:.2f}\n"
                          f"Médiane: {stats_desc['50%']:.2f}\n"
                          f"Écart-type: {stats_desc['std']:.2f}\n"
                          f"Min: {stats_desc['min']:.2f}\n"
                          f"Max: {stats_desc['max']:.2f}")
            ax2.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='top')

            with col2:
                st.pyplot(fig2)

        # Graphique interactif Violin
        if len(selected_numeric) > 1:
            fig = go.Figure()
            for col in selected_numeric:
                fig.add_trace(go.Violin(y=df[col].dropna(), name=col, box_visible=True, meanline_visible=True))
            fig.update_layout(title=f'Comparaison des distributions - {sheet_name}',
                              xaxis_title='Variables', yaxis_title='Valeurs', height=600,
                              template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    # Analyser les variables catégorielles
    categorical_columns = df.select_dtypes(include=['object']).columns

    if len(categorical_columns) > 0:
        st.subheader("Variables catégorielles")
        selected_categorical = list(categorical_columns)[:min(5, len(categorical_columns))]

        for col in selected_categorical:
            value_counts = df[col].value_counts()
            value_counts_pct = df[col].value_counts(normalize=True) * 100

            if len(value_counts) > 10:
                top_n = value_counts.nlargest(9)
                others = pd.Series({'Autres': value_counts.iloc[9:].sum()})
                value_counts = pd.concat([top_n, others])

                top_n_pct = value_counts_pct.nlargest(9)
                others_pct = pd.Series({'Autres': value_counts_pct.iloc[9:].sum()})
                value_counts_pct = pd.concat([top_n_pct, others_pct])

            # Subplots Barres + Camembert
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                                subplot_titles=[f'Distribution de {col} (Barres)', f'Distribution de {col} (Camembert)'])

            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values,
                       text=[f"{x:.1f}%" for x in value_counts_pct.values.round(1)],
                       textposition='outside', marker_color='lightseagreen'),
                row=1, col=1
            )

            fig.add_trace(
                go.Pie(labels=value_counts.index, values=value_counts.values,
                       textinfo='label+percent', marker=dict(colors=px.colors.sequential.Viridis)),
                row=1, col=2
            )

            fig.update_layout(title=f'Distribution de {col} - {sheet_name}', height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"Détails sur {col}"):
                for val, count in value_counts.items():
                    st.write(f"- {val}: {count} ({value_counts_pct[val]:.1f}%)")



        


# Interface principale du tableau de bord
def main():
    # Charger les données
    df_2019, df_2020, df_volontaire =load_data() #"df,df_volontaires"
    
    # Barre latérale pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Sélectionnez une page",
        ["Accueil","Aperçu des données", "Distribution géographique", "Santé et éligibilité", 
         "Profils des donneurs", "Analyse des campagnes", "Fidélisation des donneurs",
         "Analyse de sentiment", "Prédiction d'éligibilité","Bonus"]
    )
    
    # Sélection du jeu de données
    st.sidebar.title("Jeu de données")
    dataset = st.sidebar.radio(
        "Sélectionnez un jeu de données",
        ["2019", "Volontaire","2020"]
    )
    
    
    if page == "Accueil":
        # Afficher l'image en haut, sur toute la largeur
        image_file="Image_sang.jpg"
        image = Image.open(image_file)
        st.image(image, use_container_width=True)  # ✅ remplace use_column_width     
        st.markdown('<p style="color:black;">Texte en noir</p>', unsafe_allow_html=True)
        st.title("📊 Tableau de Bord d'Analyse des Donneurs de Sang")
        st.markdown("""
        Ce tableau de bord interactif présente une analyse approfondie des données de donneurs de sang,
        permettant d'optimiser les campagnes de don et d'améliorer la gestion des donneurs.
        """)
        
        #set_background(image_file)
        """
        Fonction principale qui crée l'interface du tableau de bord Streamlit.
        """

    

    # Titre et introduction
    
    
    
    # Afficher la page sélectionnée
    # Page d'accueil
    
    if page == "Aperçu des données":
        data_2019_path ="data_2019_pretraite.csv"
        data_2020_path = "data_2020_pretraite.csv"
        data_volontaire_path = "data_Volontaire_pretraite.csv"
    
        df_2019 = pd.read_csv(data_2019_path)
        df_2020 = pd.read_csv(data_2020_path)
        df_volontaire = pd.read_csv(data_volontaire_path)
        st.header("Aperçu des Données")
        # Configuration matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 20
       
    
        # Affichage selon le choix
        if dataset == "2019":
            visualize_missing_values(df_2019, "Données 2019")
            analyze_distributions(df_2019, "Données 2019")
        elif dataset== "2020":
            visualize_missing_values(df_2020, "Données 2020")
            analyze_distributions(df_2020, "Données 2020")
        else:
            visualize_missing_values(df_volontaire, "Données Volontaire")
            analyze_distributions(df_volontaire, "Données Volontaire")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de donneurs (2019)", len(df_2019))
        with col2:
            st.metric("Nombre de donneurs (2020)", len(df_2020))
        with col3:
            st.metric("Nombre de donneurs (Volontaires)", len(df_volontaire))
        
        st.subheader("Aperçu des données")

        if dataset=="2019":
            Apercue(df_2019)
        elif dataset=="2020":
            Apercue(df_2020)
        else :
            Apercue(df_volontaire)
            
     # Charger les données
    df_2019, df_2020, df_volontaire =load_data() 
    # Sélectionner le DataFrame en fonction du choix
    df = df_2019 if dataset == "2019" else df_volontaire
    
    
    if page == "Distribution géographique":
        st.header("🗺️ Distribution géographique des donneurs")
         # 1️⃣ Chargement des données (Fichier Excel)
        #file_path = r"C:\Users\Ultra Tech\Desktop\Challenge dataset traité.xlsx"  # Mets le bon chemin
        df_carte = df_volontaire.copy()
        # 🔹 Dictionnaire des nouveaux noms de colonnes
        new_column_names = {
            "ID": "ID",
            "Age": "Age",
            "Horodateur": "Horodateur",
            "Niveau_d'etude": "Niveau_Etude",
            "Genre_": "Genre",
            "Taille_": "Taille",
            "Poids": "Poids",
            "Situation_Matrimoniale_(SM)": "Statut_Matrimonial",
            "Profession_": "Profession",
            "Arrondissement_de_résidence_": "Arrondissement",
            "Quartier_de_Résidence_": "Quartier",
            "Nationalité_": "Nationalite",
            "Religion_": "Religion",
            "A-t-il_(elle)_déjà_donné_le_sang_": "Deja_Donneur",
            "Si_oui_preciser_la_date_du_dernier_don._": "Date_Dernier_Don",
            "Taux_d’hémoglobine_": "Taux_Hemoglobine",
            "ÉLIGIBILITÉ_AU_DON.": "Eligibilite_Don",
            "Raison_indisponibilité__[Est_sous_anti-biothérapie__]": "Sous_Antibiotherapie",
            "Raison_indisponibilité__[Taux_d’hémoglobine_bas_]": "Hemoglobine_Bas",
            "Raison_indisponibilité__[date_de_dernier_Don_<_3_mois_]": "Dernier_Don_3Mois",
            "Raison_indisponibilité__[IST_récente_(Exclu_VIH,_Hbs,_Hcv)]": "IST_Recente",
            "Date_de_dernières_règles_(DDR)__": "DDR",
            "Raison_de_l’indisponibilité_de_la_femme_[La_DDR_est_mauvais_si_<14_jour_avant_le_don]": "DDR_Mauvaise",
            "Raison_de_l’indisponibilité_de_la_femme_[Allaitement_]": "Allaitement",
            "Raison_de_l’indisponibilité_de_la_femme_[A_accoucher_ces_6_derniers_mois__]": "Accouchement_6Mois",
            "Raison_de_l’indisponibilité_de_la_femme_[Interruption_de_grossesse__ces_06_derniers_mois]": "Interruption_Grossesse",
            "Raison_de_l’indisponibilité_de_la_femme_[est_enceinte_]": "Enceinte",
            "Autre_raisons,__preciser_": "Autres_Raisons",
            'Sélectionner_"ok"_pour_envoyer_': "Confirmation_OK",
            "Raison_de_non-eligibilité_totale__[Antécédent_de_transfusion]": "Transfusion_Antecedent",
            "Raison_de_non-eligibilité_totale__[Porteur(HIV,hbs,hcv)]": "Porteur_VIH_HBS_HCV",
            "Raison_de_non-eligibilité_totale__[Opéré]": "Opere",
            "Raison_de_non-eligibilité_totale__[Drepanocytaire]": "Drepanocytose",
            "Raison_de_non-eligibilité_totale__[Diabétique]": "Diabete",
            "Raison_de_non-eligibilité_totale__[Hypertendus]": "Hypertension",
            "Raison_de_non-eligibilité_totale__[Asthmatiques]": "Asthme",
            "Raison_de_non-eligibilité_totale__[Cardiaque]": "Probleme_Cardiaque",
            "Raison_de_non-eligibilité_totale__[Tatoué]": "Tatouage",
            "Raison_de_non-eligibilité_totale__[Scarifié]": "Scarification",
            "Si_autres_raison_préciser_": "Autres_Raisons_Precises"
        }
        
        # 🔹 Renommage des colonnes du DataFrame
        df_carte.rename(columns=new_column_names, inplace=True)

        
        
        # 2️⃣ Initialisation du Géocodeur avec timeout
        geolocator = Nominatim(user_agent="blood_donation_cameroon", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2, max_retries=3)
        
        # 3️⃣ Charger le cache si existant
        cache_file = "geo_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                geo_cache = json.load(f)
        else:
            geo_cache = {}
        
        
        modalites_indesirables = [
                "Pas precise", "Pas précisé", "Pas precise", "Pas précise","Non précisé","non précisé",
                " RAS", "Ras", "R A S ", "ras", "r a s"," ","nan",
                " Pas mentionné", "Pasmentionné", "Pas mentionne"
            ]
        # 4️⃣ Fonction pour obtenir les coordonnées avec cache
        def get_coordinates_with_cache(row):
            query = f"{row['Quartier']}, {row['Arrondissement']}, Douala, Cameroun"
        
            # Vérifier si l'adresse est déjà dans le cache
            if query in geo_cache:
                return pd.Series(geo_cache[query])
        
            # Vérifier si les données sont valides avant d'envoyer la requête
            if pd.isna(row["Quartier"]) or pd.isna(row["Arrondissement"]) or row["Quartier"] in modalites_indesirables :
                return pd.Series({'latitude': 4.0483, 'longitude': 9.7043})  # Coordonnées par défaut de Douala
        
            try:
                location = geocode(query)
                if location:
                    geo_cache[query] = {'latitude': location.latitude, 'longitude': location.longitude}
                else:
                    geo_cache[query] = {'latitude': 4.0483, 'longitude': 9.7043}  # Valeur par défaut
            except Exception as e:
                print(f"⚠️ Erreur pour {query}: {e}")
                geo_cache[query] = {'latitude': 4.0483, 'longitude': 9.7043}  # Valeur par défaut
        
            return pd.Series(geo_cache[query])
        
        # 5️⃣ Vérification et exécution du géocodage uniquement si nécessaire
        if "latitude" in df_carte.columns and "longitude" in df_carte.columns:
            print("📌 Coordonnées déjà présentes dans le fichier. Géocodage non nécessaire.")
        else:
            print("📌 Démarrage du géocodage avec cache...")
            tqdm.pandas()  # Activation de la barre de progression
            coordinates = df_carte.progress_apply(get_coordinates_with_cache, axis=1)
        
            # Fusionner les nouvelles coordonnées avec les données existantes
            df_carte = pd.concat([df_carte, coordinates], axis=1)
        
            # Sauvegarder les résultats pour éviter de refaire le géocodage
            with open(cache_file, "w") as f:
                json.dump(geo_cache, f)
            print("✅ Géocodage terminé et sauvegardé !")
        
        # 6️⃣ Création de la carte Folium
        m = folium.Map(location=[4.0483, 9.7043], zoom_start=12, tiles='cartodb dark_matter')
        
        # 7️⃣ Ajout des marqueurs pour chaque donneur
        for idx, row in df_carte.iterrows():
            popup_content = f"""
            <b>{row['Genre']}, {row['Age']} ans</b><br>
            <i>{row['Profession']}</i><br>
            Éligible: {row.get('Eligibilite_Don', 'N/A')}<br>
            Quartier: {row['Quartier']}
            """
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=250),
                icon=folium.Icon(
                    color='green' if row.get('Eligibilite_Don') == 'Oui' else 'red',
                    icon='tint',
                    prefix='fa'
                )
            ).add_to(m)

        # Ajouter une couche de chaleur pour visualiser la densité
        location=[row['latitude'], row['longitude']]
        # Avec pondération (par exemple, selon le nombre de dons)
        heat_data_weighted = []
        for idx, row in df_carte.iterrows():
            if not pd.isna(row['latitude']) and not pd.isna(row['longitude']):
                # Ajouter un poids (par exemple, nombre de dons)
                weight = row.get('Nombre_Dons', 1)  # Utiliser 1 comme valeur par défaut
                heat_data_weighted.append([row['latitude'], row['longitude'], weight])
        
        # Ajouter la couche de chaleur pondérée
        HeatMap(heat_data_weighted, radius=15, blur=10).add_to(m)

        from folium.plugins import MarkerCluster
        from folium import Choropleth, GeoJson
        
        # Supposons que vous avez un fichier GeoJSON avec les limites des quartiers
        # et un DataFrame avec le nombre de dons par quartier
        
        # 1. Charger le GeoJSON des quartiers
        def create_quartier_choropleth(df):
            # Créer une carte centrée sur Douala
            #m = folium.Map(location=[4.0483, 9.7043], zoom_start=12, tiles='CartoDB positron')
            m = folium.Map(location=[4.0483, 9.7043], zoom_start=12, tiles=None)
        
            # Ajouter une couche blanche personnalisée
            folium.raster_layers.TileLayer(
                tiles='',
                attr='',
                name='fond blanc',
                overlay=False,
                control=True,
                opacity=1.0,
                styles=[('background-color', '#ffffff')]
            ).add_to(m)
        
            
            # Compter le nombre de dons par quartier
            quartier_counts = df['Quartier'].value_counts().reset_index()
            quartier_counts.columns = ['Quartier', 'Nombre_Dons']
            
            # Créer un dictionnaire pour le style de chaque quartier
            quartier_data = quartier_counts.set_index('Quartier')['Nombre_Dons'].to_dict()
            
            # Charger le GeoJSON des quartiers (vous devez créer ou obtenir ce fichier)
            try:
                with open('data/quartiers_douala.geojson', 'r') as f:
                    quartiers_geo = json.load(f)
                    
                # Créer la couche Choropleth
                Choropleth(
                    geo_data=quartiers_geo,
                    name='Densité de dons par quartier',
                    data=quartier_counts,
                    columns=['Quartier', 'Nombre_Dons'],
                    key_on='feature.properties.name',  # Doit correspondre à la propriété dans le GeoJSON
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name='Nombre de dons',
                    highlight=True,
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['name', 'Nombre_Dons'],
                        aliases=['Quartier: ', 'Nombre de dons: '],
                        localize=True
                    )
                ).add_to(m)
                
                # Ajouter un contrôle de couches
                folium.LayerControl().add_to(m)
                
            except FileNotFoundError:
                # Si le fichier GeoJSON n'existe pas, créer une alternative
                st.warning("Fichier GeoJSON des quartiers non trouvé. Création d'une visualisation alternative.")
                
                # Alternative: utiliser des cercles proportionnels
                for quartier, count in quartier_data.items():
                    # Obtenir les coordonnées du quartier depuis le fichier JSON
                    geo_data = load_geo_coordinates()
                    if "Quartier" in geo_data and quartier in geo_data["Quartier"]:
                        coords = geo_data["Quartier"][quartier]
                        
                        # Créer un cercle dont la taille dépend du nombre de dons
                        folium.Circle(
                            location=coords,
                            radius=count * 20,  # Rayon proportionnel au nombre de dons
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=0.6,
                            tooltip=f"Quartier: {quartier}<br>Nombre de dons: {count}"
                        ).add_to(m)
            
        
               
                
        # 8️⃣ Affichage de la carte
      
        folium_static(m)

    
    elif page == "Santé et éligibilité":
        st.header("🩺 Conditions de santé et éligibilité au don")   
        data_2019_path ="data_2019_pretraite.csv"
        data_2020_path = "data_2020_pretraite.csv"
        data_volontaire_path = "data_Volontaire_pretraite.csv"
    
        df_2019 = pd.read_csv(data_2019_path)
        df_2020 = pd.read_csv(data_2020_path)
        df_volontaire = pd.read_csv(data_volontaire_path)
        if dataset == "2019":
            analyze_categorical_relationships(df_2019, "Données 2019")
        elif dataset == "2020":
            analyze_categorical_relationships(df_2020, "Données 2020")
        else :
            analyze_categorical_relationships(df_volontaire, "Données volontaire")

    # Charger les données
    df_2019, df_2020, df_volontaire =load_data() 
    
    if page == "Profils des donneurs":
        # Afficher l'image en haut, sur toute la largeur
        image_file="Profil_donneur.jpg"
        image = Image.open(image_file)
        st.image(image, use_container_width=True)  # ✅ remplace use_column_width     
        st.header("👥 Profils des donneurs")
        
        # Effectuer le clustering des donneurs
        cluster_fig, cluster_stats, df_with_clusters = create_donor_clustering(df)
        
        if cluster_fig is not None:
            # Afficher la visualisation des clusters
            st.subheader("Clustering des donneurs")
            st.plotly_chart(cluster_fig, use_container_width=True)
            
            # Afficher les caractéristiques des clusters
            st.subheader("Caractéristiques des clusters")
            st.dataframe(cluster_stats)
            
            # Créer des profils de donneurs idéaux
            st.subheader("Profils de donneurs idéaux")
            
            # Identifier le cluster avec le plus grand nombre de donneurs éligibles
            if 'ÉLIGIBILITÉ_AU_DON.' in df_with_clusters.columns and 'Cluster' in df_with_clusters.columns:
                # Compter le nombre de donneurs éligibles par cluster
                eligible_counts = df_with_clusters[df_with_clusters['ÉLIGIBILITÉ_AU_DON.'] == 'Eligible'].groupby('Cluster').size()
                
                if not eligible_counts.empty:
                    ideal_cluster = eligible_counts.idxmax()
                    
                    st.write(f"**Cluster idéal identifié: Cluster {ideal_cluster}**")
                    
                    # Extraire les caractéristiques du cluster idéal
                    ideal_profile = df_with_clusters[df_with_clusters['Cluster'] == ideal_cluster]
                    
                    # Afficher les caractéristiques démographiques du cluster idéal
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Caractéristiques démographiques:**")
                        
                        if 'Age' in ideal_profile.columns:
                            st.write(f"- Âge moyen: {ideal_profile['Age'].mean():.1f} ans")
                        
                        if 'Genre_' in ideal_profile.columns:
                            gender_pct = ideal_profile['Genre_'].value_counts(normalize=True) * 100
                            st.write(f"- Genre: {gender_pct.get('Homme', 0):.1f}% Hommes, {gender_pct.get('Femme', 0):.1f}% Femmes")
                        
                        if 'Niveau_d\'etude' in ideal_profile.columns:
                            top_edu = ideal_profile['Niveau_d\'etude'].value_counts(normalize=True).head(2)
                            st.write("- Niveau d'études principal:")
                            for edu, pct in top_edu.items():
                                st.write(f"  • {edu}: {pct*100:.1f}%")
                        
                        if 'Situation_Matrimoniale_(SM)' in ideal_profile.columns:
                            top_marital = ideal_profile['Situation_Matrimoniale_(SM)'].value_counts(normalize=True).head(2)
                            st.write("- Situation matrimoniale principale:")
                            for status, pct in top_marital.items():
                                st.write(f"  • {status}: {pct*100:.1f}%")
                    
                    with col2:
                        st.write("**Caractéristiques géographiques:**")
                        
                        geo_columns = [col for col in ideal_profile.columns if any(term in col for term in 
                                      ['Arrondissement', 'Quartier', 'Résidence'])]
                        
                        for geo_col in geo_columns:
                            top_geo = ideal_profile[geo_col].value_counts(normalize=True).head(3)
                            st.write(f"- {geo_col} principal:")
                            for zone, pct in top_geo.items():
                                st.write(f"  • {zone}: {pct*100:.1f}%")
                    
                    # Créer un radar chart pour visualiser le profil idéal
                    if 'Age' in ideal_profile.columns and 'Taille_' in ideal_profile.columns and 'Poids' in ideal_profile.columns:
                        # Calculer les moyennes normalisées
                        avg_age = ideal_profile['Age'].mean() / df['Age'].max()
                        avg_height = ideal_profile['Taille_'].mean() / df['Taille_'].max()
                        avg_weight = ideal_profile['Poids'].mean() / df['Poids'].max()
                        
                        # Créer un radar chart
                        categories = ['Âge', 'Taille', 'Poids']
                        values = [avg_age, avg_height, avg_weight]
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Profil idéal'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Caractéristiques physiques du profil idéal (normalisées)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Impossible de déterminer le profil idéal car la colonne d'éligibilité n'est pas disponible.")
        else:
            st.warning("Impossible d'effectuer le clustering car il n'y a pas assez de variables numériques dans les données.")
    
    elif page == "Analyse des campagnes":
        st.header("📈 Analyse des campagnes de don")
        
        # Créer des visualisations pour l'analyse des campagnes
        campaign_fig, demographic_figs = create_campaign_analysis(df)
        
        if campaign_fig is not None:
            # Afficher la tendance temporelle générale
            st.subheader("Évolution du nombre de donneurs au fil du temps")
            st.plotly_chart(campaign_fig, use_container_width=True)
            
            # Afficher les tendances par caractéristiques démographiques
            if demographic_figs:
                st.subheader("Tendances par caractéristiques démographiques")
                
                for fig in demographic_figs:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Ajouter des recommandations pour l'optimisation des campagnes
            st.subheader("Recommandations pour l'optimisation des campagnes")
            
            st.write("""
            Sur la base de l'analyse des données, voici quelques recommandations pour optimiser les futures campagnes de don de sang:
            
            1. **Ciblage démographique**: Concentrez les efforts sur les segments de population les plus susceptibles de donner, comme identifié dans l'analyse des profils.
            
            2. **Planification temporelle**: Organisez les campagnes pendant les périodes où le taux de participation est historiquement élevé.
            
            3. **Localisation géographique**: Privilégiez les zones géographiques avec un taux d'éligibilité élevé et une forte concentration de donneurs potentiels.
            
            4. **Sensibilisation ciblée**: Développez des messages spécifiques pour les groupes sous-représentés afin d'augmenter leur participation.
            
            5. **Fidélisation**: Mettez en place des stratégies pour encourager les donneurs à revenir régulièrement.
            """)
        else:
            st.warning("Impossible d'analyser les tendances temporelles car aucune colonne de date appropriée n'a été identifiée.")
    
    elif page == "Fidélisation des donneurs":
        # Afficher l'image en haut, sur toute la largeur
        image_file="Fidélisation.jpg"
        image = Image.open(image_file)
        st.image(image, use_container_width=True)  # ✅ remplace use_column_width     
        st.header("🔄 Fidélisation des donneurs")
        
        # Créer des visualisations pour l'analyse de fidélisation
        retention_fig, demographic_figs = create_donor_retention_analysis(df)
        
        if retention_fig is not None:
            # Afficher la proportion de donneurs fidélisés
            st.subheader("Proportion de donneurs fidélisés")
            st.plotly_chart(retention_fig, use_container_width=True)
            
            # Afficher la fidélisation par caractéristiques démographiques
            if demographic_figs:
                st.subheader("Fidélisation par caractéristiques démographiques")
                
                for fig in demographic_figs:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Ajouter des stratégies pour améliorer la fidélisation
            st.subheader("Stratégies pour améliorer la fidélisation des donneurs")
            
            st.write("""
            Voici quelques stratégies pour améliorer la fidélisation des donneurs de sang:
            
            1. **Programme de reconnaissance**: Mettre en place un système de reconnaissance pour les donneurs réguliers (badges, certificats, etc.).
            
            2. **Communication personnalisée**: Envoyer des rappels personnalisés aux donneurs en fonction de leur historique de don.
            
            3. **Expérience positive**: Améliorer l'expérience du donneur pendant le processus de don pour encourager le retour.
            
            4. **Éducation continue**: Informer les donneurs sur l'impact de leur don et l'importance de donner régulièrement.
            
            5. **Événements communautaires**: Organiser des événements spéciaux pour les donneurs réguliers afin de renforcer leur engagement.
            """)
        else:
            st.warning("Impossible d'analyser la fidélisation car les informations nécessaires ne sont pas disponibles dans les données.")
    
    elif page == "Analyse de sentiment":
        # Afficher l'image en haut, sur toute la largeur
        image_file="Analyse_sentiment.jpg"
        image = Image.open(image_file)
        st.image(image, use_container_width=True)  # ✅ remplace use_column_width     
        st.header("🩺 Conditions de santé et éligibilité au don")   
        data_2019_path ="data_2019_pretraite.csv"
        data_2020_path = "data_2020_pretraite.csv"
        data_volontaire_path = "data_Volontaire_pretraite.csv"
    
        df_2019 = pd.read_csv(data_2019_path)
        df_2020 = pd.read_csv(data_2020_path)
        df_volontaire = pd.read_csv(data_volontaire_path)
        
        st.header("💬 Analyse de sentiment des retours")
        nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
        
        if "Si_autres_raison_préciser_" in df_volontaires.columns:
            df_volontaires["Sentiment"] = df_volontaires["Si_autres_raison_préciser_"].dropna().apply(lambda x: sia.polarity_scores(str(x))["compound"])
            text = " ".join(str(f) for f in df_volontaires["Si_autres_raison_préciser_"].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
            st.image(wordcloud.to_array(), caption="Nuage de Mots des Feedbacks", use_column_width=True)
        if dataset == "2019":
            create_sentiment_analysis(df_2019)
        if dataset == "Volontaire":
            create_sentiment_analysis(df_volontaire)
        
    elif page == "Bonus" :
        st.markdown("### 🔍 Bonus - Exploration Interactive avec Pygwalker")
    
        # Section Pygwalker selon le dataset choisi
        if dataset == "2019":
            df_selected = df_2019
        elif dataset == "2020":
            df_selected = df_2020
        else:
            df_selected = df_volontaire
    
        with st.expander("🎯 Analyse interactive du dataset sélectionné"):
            pyg_html = pyg.walk(df_selected, output_type='html')
            components.html(pyg_html, height=500, scrolling=True)
    
        # Upload CSV ou Excel
        st.markdown("---")
        st.markdown("### 📂 Importer ton propre fichier")
    
        uploaded_file = st.file_uploader("Choisis un fichier CSV ou Excel", type=["csv", "xlsx"])
    
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                data = pd.read_excel(uploaded_file)
            else:
                st.warning("Format non supporté.")
                data = None
    
            if data is not None:
                st.success("✅ Fichier chargé avec succès !")
                with st.expander("📊 Analyse interactive de ton fichier importé"):
                    pyg_html = pyg.walk(data, output_type='html')
                    components.html(pyg_html, height=500, scrolling=True)

        
    elif page == "Prédiction d'éligibilité":
        st.header("🔮 Prédiction d'éligibilité au don")
        
        st.title("🔍 Prédiction d'Éligibilité au Don de Sang")
        # ==============================
        # 🎯 CHARGEMENT DU MODÈLE & ENCODEURS
        # ==============================
        @st.cache_resource
        def load_model():
            pathse="eligibility_model.pkl"
            data = joblib.load(pathse)
            return data["model"], data["X_test"], data["y_test"], data["target_encoder"], data["lpreprocessor"],data["resultat"]

        model, X_test, y_test, target_encoder, preprocessor,resultat = load_model()
        

        columns_binary = [
                "Drepanocytose", "Opere", "Transfusion_Antecedent", "Diabete",
                "Hypertension", "Porteur_VIH_HBS_HCV", "Asthme",
                "Probleme_Cardiaque", "Tatouage", "Scarification", "Deja_Donneur"
            ]



        # ==============================
        # 🗂 ORGANISATION EN ONGLETS
        # ==============================
        tab1, tab2,tab3= st.tabs([ "🔄 Prédiction Individuelle","📥 Télécharger/Charger un Fichier","Performance du modèle"])


        with tab1:
            col1,col2,col3=st.columns(3)
            st.subheader("🔄 Faire une prédiction individuelle")

            st.write("""Ce modèle prédit si un donneur est éligible ou non en fonction de ses caractéristiques médicales et personnelles.
            Remplissez les informations ci-dessous pour obtenir une prédiction.
        """)
            #df=pd.read_csv(patr)
            professions = ["Etudiant (e)", "Sans Emplois", "Tailleur", "Militaire", "Bijoutier", "Couturier","Jeune cadre", "Commerçant (e)", "Mototaximan", "Agent commercial", "Elève","Chauffeur", "COIFFEUSE", "Mécanicien", "Cultuvateur", "Fonctionnaires", "Marin", 
    "Infirmièr", "Footballeur", "Agent de securite", "Electronicien", "Électricien", 
    "Élève", "Sportifs", "Personnel de sante", "Formateur", "Trader indep", 
    "Chargé de clientèle", "CONTROLEUR DES DOUANES", "ASSISTANT DE  DIRECTION", 
    "STAGIAIRE", "Angent de securité", "Pasteur", "S1p", "Plombier", "Security officer", 
    "BUSINESMAN", "Footballeur ", "Technicien supérieur d’agriculture", "Vigil", 
    "Coordiste", "PEINTRE", "ADMINISTRATEUR DES HOPITAUX", "chauffeur", "Hôtelier ", 
    "Logisticien transport", "CHAUDRONIER", "Decorateur Baptiment", "Tâcheron", 
    "Cuisinier", "Imprimeur ", "missionnaire", "Patissier", "Docker", "Réalisateur ", 
    "assureur", "CHAUFFEUR ", "LAVEUR", "Coach kuine", "Conducteur", "Technicien ", 
    "Conseiller client", "Entrepreneur", "Magasinier ", "constructeur en bâtiment", 
    "Calier", "SOUDEUR", "AGENT D'APPUI PHARMICIE", "CHAUFFEUR", "Intendant ", 
    "conducteurs d'engins genie civil", "Chauffeur ", "Assistant transit", 
    "Agent des ressourses humaines", "Declarant en Douane", "Menuisier", "AIDE COMPTABLE", 
    "TECHNICIEN GENIE CIVIL", "Transitaire", "Coiffeur", "Ménagère", "VENDEUSE", 
    "médecin", "couturière", "Pompiste", "EDUCATEUR DES ENFANTS", "Hoteliere", 
    "OUVRIER", "Débrouillard", "MACHINISTE", "FORREUR", "CASINO", "TECHNICIEN TOPO", 
    "COUTURIERE", "RAS", "APPRENTI TOLERIE ", "SP", "Développeur en informatique ", 
    "Sérigraphie", "Estheticien", "Maintenancier industriel ", "Auditeur interne", 
    "Enseignant (e)", "Agent municipal ", "Tolier", "Agent de banque", "Prestataire de service et consultant sportif", 
    "Dolker ", "photographe", "Agent d'exploitation", "Cheminot", "ARGENT DE SÉCURITÉ ", 
    "Secrétaire comptable", "Contractuel d'administration", "Technicien de génie civile", 
    "Juriste", "Informaticien ", "Technicien en genie civil", "Agent administratif ", 
    "Comptable", "Laborantin", "Ingénieur génie civil", "Analyste -programmeur", 
    "Logisticien", "Agent de securité", "Maçon", "Menuisier ", "MENUSIER", "MENUISIER ", 
    "Plombier", "Bijoutier", "Soudeur", "Peintre", "Chaudronnier", "Électronicien ", 
    "Electricien", "Machiniste", "Pâtissier ", "Menuisier", "CHAUDRONNIER", 
    "Technicien génie civil", "Agent technique", "Technicien réseaux télécoms", 
    "Infographe", "Architecte", "Assistante", "Ménagère", "Commerçant (e)", 
    "Employé (e)  dans une entreprise", "Agent de sécurité", "Marin", "Débrouillard", 
    "Personnel de sante", "Comptable", "Enseignant", "Fonctionnaires", "Magasinier", 
    "Agent commercial", "Technicien", "Informaticien", "Electricien auto", 
    "Technicien de génie civile", "Technicien d'agriculture", "Technicien en bâtiment", 
    "Technicien en électricité", "Technicien en mécanique", "Technicien en plomberie", 
    "Technicien en soudure", "Technicien en informatique", "Technicien en génie civil", 
    "Technicien en électronique", "Technicien en climatisation", "Technicien en télécoms", 
    "Technicien en topographie", "Technicien en maintenance", "Technicien en chauffage", 
    "Technicien en froid", "Technicien en électricité bâtiment", "Technicien en électricité industrielle", 
    "Technicien en mécanique auto", "Technicien en mécanique industrielle", 
    "Technicien en plomberie sanitaire", "Technicien en soudure industrielle", 
    "Technicien en informatique bureautique", "Technicien en informatique réseau", 
    "Technicien en informatique développement", "Technicien en génie civil bâtiment", 
    "Technicien en génie civil travaux publics", "Technicien en électronique numérique", 
    "Technicien en électronique analogique", "Technicien en climatisation centrale", 
    "Technicien en télécoms réseau", "Technicien en topographie terrestre", 
    "Technicien en maintenance industrielle", "Technicien en chauffage central", 
    "Technicien en froid industriel", "Technicien en électricité bâtiment résidentiel", 
    "Technicien en électricité industrielle lourde", "Technicien en mécanique automobile", 
    "Technicien en mécanique industrielle lourde", "Technicien en plomberie sanitaire résidentielle", 
    "Technicien en soudure industrielle lourde", "Technicien en informatique bureautique avancée", 
    "Technicien en informatique réseau avancé", "Technicien en informatique développement avancé", 
    "Technicien en génie civil bâtiment avancé", "Technicien en génie civil travaux publics avancés", 
    "Technicien en électronique numérique avancée", "Technicien en électronique analogique avancée", 
    "Technicien en climatisation centrale avancée", "Technicien en télécoms réseau avancé", 
    "Technicien en topographie terrestre avancée", "Technicien en maintenance industrielle avancée", 
    "Technicien en chauffage central avancé", "Technicien en froid industriel avancé", 
    "Technicien en électricité bâtiment résidentiel avancé", "Technicien en électricité industrielle lourde avancée", 
    "Technicien en mécanique automobile avancée", "Technicien en mécanique industrielle lourde avancée", 
    "Technicien en plomberie sanitaire résidentielle avancée", "Technicien en soudure industrielle lourde avancée", 
    "Technicien en informatique bureautique expert", "Technicien en informatique réseau expert", 
    "Technicien en informatique développement expert", "Technicien en génie civil bâtiment expert", 
    "Technicien en génie civil travaux publics expert", "Technicien en électronique numérique expert", 
    "Technicien en électronique analogique expert", "Technicien en climatisation centrale expert", 
    "Technicien en télécoms réseau expert", "Technicien en topographie terrestre expert", 
    "Technicien en maintenance industrielle expert", "Technicien en chauffage central expert", 
    "Technicien en froid industriel expert", "Technicien en électricité bâtiment résidentiel expert", 
    "Technicien en électricité industrielle lourde expert", "Technicien en mécanique automobile expert", 
    "Technicien en mécanique industrielle lourde expert", "Technicien en plomberie sanitaire résidentielle expert", 
    "Technicien en soudure industrielle lourde expert"
]
            religions = [
    "Chretien (Catholique)", "Musulman", "Pas Précisé", "Chretien (Protestant )", 
    "Adventiste", "Non-croyant", "pentecôtiste", "Chretien (Ne de nouveau)", 
    "Traditionaliste", "BAPTISTE", "pentecôtiste", "Chrétien non précisé", 
    "pentecotiste", "Chretien (Protestant)", "Musulmane"
]
            niveaux_etude = [
    "Universitaire", "Aucun", "Secondaire", "Primaire", "Pas Précisé"
]
            statuts_matrimoniaux = [
    "Célibataire", "Marié (e)", "veuf (veuve)", "Divorcé(e)"
]
            # ==============================
            # 📌 FORMULAIRE DE SAISIE
            # ==============================
            with col1:
                age = st.number_input("Âge", min_value=18, max_value=100, value=30, step=1)

                profession = st.selectbox("Profession",list(professions) )
                religion = st.selectbox("Religion", list(religions))
                Niveau_Etude =st.selectbox("Niveau_Etude", list(niveaux_etude))
                Statut_Matrimonial= st.selectbox("Statut_Matrimonial", list(statuts_matrimoniaux))
                # État de santé (Binaire : Oui/Non)
            columns_binary = [
                "Drepanocytose", "Opere", "Transfusion_Antecedent", "Diabete",
                "Hypertension", "Porteur_VIH_HBS_HCV", "Asthme",
                "Probleme_Cardiaque", "Tatouage", "Scarification", "Deja_Donneur"
            ]
            columns_binarys=[
                "Drepanocytose", "Opere", "Transfusion_Antecedent", "Diabete",
                "Hypertension", "Porteur_VIH_HBS_HCV"]
            columns_binaryss=["Asthme",
                "Probleme_Cardiaque", "Tatouage", "Scarification", "Deja_Donneur"]
            
            with col2:
                binary_inputs = {}
                for col in columns_binarys:
                    binary_inputs[col] = st.radio(col, ["Non", "Oui"])
            with col3:
                for col in columns_binaryss:
                    binary_inputs[col] = st.radio(col, ["Non", "Oui"])

            # ==============================
            # 🔄 PRÉPARATION DES DONNÉES
            # ==============================
            # Convertir les entrées utilisateur en dataframe
            input_data = pd.DataFrame([[age, profession] + [binary_inputs[col] for col in columns_binary]+[religion,Niveau_Etude,Statut_Matrimonial]],
                                    columns=["Age", "Profession"] + columns_binary+["Religion","Niveau_Etude","Statut_Matrimonial"])

            # Encoder Profession et Religion
            input_data =preprocessor.transform(input_data)
            # ==============================
            # 🚀 PRÉDICTION
            # ==============================
            if st.button("Prédire l'éligibilité"):
                prediction = model.predict(input_data)[0]
                result = target_encoder.inverse_transform([prediction])[0]
                 # Afficher des recommandations en fonction de la prédiction
                st.subheader("Recommandations")
                


                # Affichage du résultat
                if prediction == 1:
                    st.success("✅ Le donneur est *ÉLIGIBLE* au don de sang !")
                    st.balloons()
                else:
                    st.error("❌ Le donneur *N'EST PAS ÉLIGIBLE* au don de sang.")

                # Affichage des valeurs d'entrée encodées
                st.subheader("🔎 Données encodées utilisées pour la prédiction :")
                #st.dataframe(input_data)
        
        
        # ==============================
        # 📥 ONGLET : TÉLÉCHARGEMENT / UPLOAD
        # ==============================
        with tab2:

            # ==============================
            # 📥 TÉLÉCHARGEMENT DU FICHIER MODÈLE
            # ==============================
            st.subheader("📥 Télécharger le modèle de fichier Excel")
            sample_data = pd.DataFrame({
                "Age": [30, 45],
                "Profession": ["Médecin", "Enseignant"],
                "Drepanocytose": ["Non", "Oui"],
                "Opere": ["Non", "Non"],
                "Transfusion_Antecedent": ["Oui", "Non"],
                "Diabete": ["Non", "Oui"],
                "Hypertension": ["Oui", "Non"],
                "Porteur_VIH_HBS_HCV": ["Non", "Non"],
                "Asthme": ["Non", "Oui"],
                "Probleme_Cardiaque": ["Non", "Non"],
                "Tatouage": ["Oui", "Non"],
                "Scarification": ["Non", "Oui"],
                "Deja_Donneur": ["Oui", "Non"],
                "Religion": ["Musulman", "Chrétien"],
                "Niveau_Etude":["Primaire","secondaire"],
                "Statut_Matrimonial":["Marier","Celibataire"]
            })

            # Génération du fichier Excel à télécharger
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                sample_data.to_excel(writer, index=False, sheet_name="Exemple")
                writer.close()

            st.download_button(
                label="📥 Télécharger le fichier modèle",
                data=output.getvalue(),
                file_name="Modele_Donneurs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # ==============================
            # 📤 UPLOADER UN FICHIER EXCEL AVEC LES DONNEURS
            # ==============================
            st.subheader("📤 Uploader un fichier Excel contenant les informations des donneurs")
            uploaded_file = st.file_uploader("Téléchargez un fichier Excel", type=["xlsx"])

            if uploaded_file:
                # Lecture du fichier
                df_uploaded = pd.read_excel(uploaded_file)

                # Vérification des colonnes attendues
                expected_columns = sample_data.columns.tolist()
                if not all(col in df_uploaded.columns for col in expected_columns):
                    st.error("⚠ Le fichier ne contient pas les bonnes colonnes ! Vérifiez le format et réessayez.")
                else:
                    st.success("✅ Fichier chargé avec succès !")

                    # ==============================
                    # 🔄 PRÉTRAITEMENT DES DONNÉES POUR TOUTES LES LIGNES
                    # ==============================
                    # Supprimer les lignes avec des valeurs non reconnues
                    df_uploaded.dropna(inplace=True)
                    colonne=["Age", "Profession", "Drepanocytose","Opere",
                "Transfusion_Antecedent","Diabete", "Hypertension", "Porteur_VIH_HBS_HCV", 
                "Asthme", "Probleme_Cardiaque","Tatouage","Scarification", "Deja_Donneur", "Religion","Niveau_Etude","Statut_Matrimonial"]
                    input_data =preprocessor.transform(df_uploaded)
                    df_uploadeds=input_data.copy()
                    

                    # Encoder les valeurs binaires (Oui = 1, Non = 0)
                    if df_uploadeds.shape[0] == 0:
                        st.error("Aucune donnée valide après prétraitement ! vérifiez les données d'entrée")

                    # ==============================
                    # 🚀 PRÉDICTION SUR TOUTES LES LIGNES
                    # ==============================
                    df_uploaded["Prédiction"] = model.predict(df_uploadeds)

                    # Décodage des résultats
                    df_uploaded["Prédiction"] = df_uploaded["Prédiction"].map(lambda x: target_encoder.inverse_transform([x])[0])

                    # Affichage du tableau avec les prédictions
                    st.subheader("📊 Résultats des prédictions")
                    st.dataframe(df_uploaded)

                    # ==============================
                    # 📤 TÉLÉCHARGER LE FICHIER AVEC PREDICTIONS
                    # ==============================
                    output_predictions = io.BytesIO()
                    with pd.ExcelWriter(output_predictions, engine='xlsxwriter') as writer:
                        df_uploaded.to_excel(writer, index=False, sheet_name="Prédictions")
                        writer.close()

                    st.download_button(
                        label="📥 Télécharger les résultats",
                        data=output_predictions.getvalue(),
                        file_name="Resultats_Predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    # ==============================
    # 📈 ONGLET : PERFORMANCE DU MODÈLE
    # ==============================
        with tab3:
            # ==============================
            # 🎯 CHARGEMENT DU MODÈLE & DONNÉES TEST
            # ==============================
            @st.cache_resource
            def load_model():
               pathse="eligibility_model.pkl"
               data = joblib.load(pathse)
               return data["model"], data["X_test"], data["y_test"], data["target_encoder"], data["lpreprocessor"],data["resultat"]

            model, X_test, y_test, target_encoder, preprocessor,resultat = load_model()
                # ==============================
            # 📊 PERFORMANCE DU MODÈLE SUR DONNÉES TEST
            # ==============================
            st.subheader("📈 Performance du Modèle sur Données de Test")

            # 🔮 Prédictions sur X_test
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # 📄 Rapport de Classification
            st.subheader("📄 Rapport de Classification")
            A=target_encoder.inverse_transform([0,1,2])
            report = classification_report(y_test, y_pred, target_names=A, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report)
            st.dataframe(resultat)
            

            # 🔄 Binarisation des étiquettes pour "One vs Rest" si multi-classe
            n_classes = len(np.unique(y_test))  # Nombre de classes
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))  # Transforme y_test en binaire
            y_pred_proba = model.predict_proba(X_test)  # Probabilités prédites pour chaque classe

            # 📈 Affichage de la courbe ROC pour chaque classe
            st.subheader("📈 Courbe ROC (One vs Rest)")

            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ['blue', 'red', 'green', 'purple', 'orange']  # Couleurs pour chaque classe

            for i in range(n_classes):
                if n_classes > 2:  # Cas multi-classe
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"Classe {A[i]} (AUC = {roc_auc:.2f})")
                else:  # Cas binaire classique
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")

            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax.set_xlabel("Taux de Faux Positifs (FPR)")
            ax.set_ylabel("Taux de Vrais Positifs (TPR)")
            ax.set_title("Courbe ROC par Classe")
            ax.legend(loc="lower right")
            st.pyplot(fig)

            
            # 📊 Matrice de Confusion
            st.subheader("📊 Matrice de Confusion")
            B=target_encoder.inverse_transform(y_test)
            C=target_encoder.inverse_transform(y_pred)
            conf_matrix = confusion_matrix(B, C)
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Matrice de Confusion")
            st.pyplot(fig)
            # Afficher les probabilités
            st.write("**Probabilités:**")
            
            # Créer un DataFrame pour les probabilités
            proba_df = pd.DataFrame({
                'Statut': model.classes_,
                'Probabilité': y_pred_proba[0]
            })
            
            # Créer un graphique à barres pour les probabilités
            fig = px.bar(
                proba_df,
                x='Statut',
                y='Probabilité',
                title="Probabilités pour chaque statut d'éligibilité",
                labels={'Probabilité': 'Probabilité', 'Statut': "Statut d'éligibilité"}
            )
            
            fig.update_layout(
                xaxis_title="Statut d'éligibilité",
                yaxis_title="Probabilité",
                font=dict(size=12),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

            #if hasattr(model, 'feature_importances_'):
            # Obtenir les noms des caractéristiques après one-hot encoding
            feature_names = []
            for name, transformer, features in preprocessor.transformers_:
                if name == 'cat':
                    # Pour les caractéristiques catégorielles, obtenir les noms après one-hot encoding
                    for i, feature in enumerate(features):
                        categories = transformer.named_steps['onehot'].categories_[i]
                        for category in categories:
                            feature_names.append(f"{feature}_{category}")
                else:
                    # Pour les caractéristiques numériques, conserver les noms d'origine
                    feature_names.extend(features)
            
            # Obtenir les importances des caractéristiques
            importances = model.feature_importances_
            
            # Créer un DataFrame pour les importances
            if len(feature_names) == len(importances):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                
                # Trier par importance décroissante
                importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
                
                # Créer un graphique des importances
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Importance des caractéristiques pour la prédiction d'éligibilité",
                    labels={'Importance': 'Importance relative', 'Feature': 'Caractéristique'}
                )
                
                fig_importance.update_layout(
                    xaxis_title="Importance relative",
                    yaxis_title="Caractéristique",
                    font=dict(size=12),
                    height=600
                )
            st.plotly_chart(fig_importance)
           
                    
                    
    
    # Pied de page
    st.markdown("---")
    st.markdown(""" <div style="text-align: center;">
        <p>Tableau de bord développé pour le concours de data visualisation sur les donneurs de sang</p>
        <p>© 2025 - Tous droits réservés</p>
    </div>
    """, unsafe_allow_html=True)

# Point d'entrée principal
if __name__ == "__main__":
    main()
