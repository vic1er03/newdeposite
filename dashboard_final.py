"""Tableau de bord interactif pour l'analyse des données de donneurs de sang.
Ce script crée un tableau de bord Streamlit avec des visualisations innovantes
pour répondre aux objectifs du concours de data visualisation.
"""

import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

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
from streamlit_option_menu import option_menu
from matplotlib.gridspec import GridSpec

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
from scipy import stats

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
    st.subheader(f"📊 Analyse des relations entre variables catégorielles – {sheet_name}")

    categorical_columns = df.select_dtypes(include=['object']).columns
    valid_columns = [col for col in categorical_columns if 1 < df[col].nunique() <= 10]

    if len(valid_columns) >= 2:
        # 🎯 Détection automatique de la variable cible
        target_column = None
        for potential_target in ['ÉLIGIBILITÉ_AU_DON.', 'Éligibilité_au_don', 'Eligibilité']:
            if potential_target in valid_columns:
                target_column = potential_target
                break

        if target_column:
            st.markdown(
                f"<div style='background-color: #e6f4ff; padding: 10px; border-radius: 5px; display: flex; align-items: center;'>"
                f"<strong>🎯 Variable cible détectée : <code>{target_column}</code></strong></div>",
                unsafe_allow_html=True
            )

            # === Résumé des tests ===
            total_tests = 0
            nb_significatives = 0
            p_values = []
            relations = []

            for col in valid_columns:
                if col != target_column:
                    contingency = pd.crosstab(df[target_column], df[col])
                    chi2, p, dof, expected = stats.chi2_contingency(contingency)
                    total_tests += 1
                    p_values.append(p)
                    if p < 0.05:
                        nb_significatives += 1
                    relations.append((p, col))

            relation_mini = min(relations, key=lambda x: x[0]) if relations else (None, None)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🔢 Total de tests", total_tests)
            col2.metric("✅ Relations significatives", nb_significatives)
            col3.metric("📉 p-value moyenne", f"{sum(p_values)/len(p_values):.4f}" if p_values else "N/A")
            col4.metric("📌 p-value min", f"{relation_mini[0]:.4f}" if relation_mini[0] else "N/A")
            col5.metric("📊 Plus forte relation", f"{target_column} vs {relation_mini[1]}" if relation_mini[1] else "N/A")

            st.markdown("---")

            # === Visualisation des relations ===
            for col in valid_columns:
                if col != target_column:
                    contingency = pd.crosstab(df[target_column], df[col])
                    chi2, p, dof, expected = stats.chi2_contingency(contingency)
                    association = "✅ significative" if p < 0.05 else "❌ non significative"

                    st.markdown(
                        f"<div style='background-color: #f0f8ff; padding: 10px; border-left: 5px solid #007ACC; margin-top: 20px;'>"
                        f"<strong>Relation entre <code>{target_column}</code> et <code>{col}</code></strong><br>"
                        f"Test du χ² : χ² = {chi2:.2f}, p = {p:.4f} → {association}</div>",
                        unsafe_allow_html=True
                    )

                    # Espacement des deux graphiques côte à côte
                    col_g1, col_g2 = st.columns([1, 1])

                    with col_g1:
                        st.markdown("#### 📊 Diagramme en barres")
                        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                        fig_bar = px.bar(
                            contingency_pct,
                            barmode='group',
                            title=f"{target_column} vs {col}",
                            labels={'value': 'Pourcentage (%)', 'index': target_column},
                            height=350
                        )
                        fig_bar.update_layout(template='plotly_white')
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with col_g2:
                        st.markdown("#### 🧁 Répartition circulaire")
                        pie_data = df[col].value_counts().reset_index()
                        pie_data.columns = [col, 'count']
                        fig_pie = px.pie(
                            pie_data,
                            names=col,
                            values='count',
                            hole=0.4,
                            color_discrete_sequence=px.colors.sequential.RdBu,
                            title=f"Distribution de {col}"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
        else:
            st.warning("⚠️ Aucune variable cible claire trouvée parmi les colonnes catégorielles.")
    else:
        st.warning("⚠️ Pas assez de variables catégorielles valides pour effectuer une analyse.")


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
    Crée des visualisations matplotlib (heatmap, barplots, line) pour analyser les dons de sang sur l'année 2019.
    """
    st.subheader("📊 Analyse des Dons de Sang – Année 2019")

    df = df.copy()

    # Nettoyage et formatage des dates de don
    df['Date_de_remplissage_de_la_fiche'] = pd.to_datetime(df['Date_de_remplissage_de_la_fiche'], errors='coerce')
    df = df.dropna(subset=['Date_de_remplissage_de_la_fiche'])

    # Extraire les informations temporelles utiles
    df["Année"] = df["Date_de_remplissage_de_la_fiche"].dt.year
    df["Mois"] = df["Date_de_remplissage_de_la_fiche"].dt.month
    df["Jour"] = df["Date_de_remplissage_de_la_fiche"].dt.day
    df["Jour de la semaine"] = df["Date_de_remplissage_de_la_fiche"].dt.dayofweek  # 0 = Lundi, 6 = Dimanche
    df["Mois_Nom"] = df["Mois"].apply(lambda m: calendar.month_name[m])

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)

    # 1. Carte thermique des dons par jour et mois
    ax1 = fig.add_subplot(gs[0, 0])
    heatmap_data = df.pivot_table(index="Jour de la semaine", columns="Mois", values="Date_de_remplissage_de_la_fiche", aggfunc="count")
    heatmap_data.index = [calendar.day_name[i] for i in heatmap_data.index]
    heatmap_data.columns = [calendar.month_name[int(col)] for col in heatmap_data.columns]
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.5, linecolor="black", ax=ax1)
    ax1.set_title("Carte thermique des dons par jour et mois", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Mois")
    ax1.set_ylabel("Jour de la semaine")
    ax1.tick_params(axis='x', rotation=20)
    ax1.tick_params(axis='y', rotation=0)

    # 2. Distribution des dons par mois
    ax2 = fig.add_subplot(gs[0, 1])
    monthly_counts = df.groupby("Mois_Nom")["Date_de_remplissage_de_la_fiche"].count()
    month_order = [calendar.month_name[i] for i in range(1, 13) if calendar.month_name[i] in monthly_counts.index]
    monthly_counts = monthly_counts.reindex(month_order)
    monthly_counts.plot(kind='bar', color='skyblue', ax=ax2)
    ax2.set_title("Distribution des dons par mois", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("Nombre de dons")
    ax2.tick_params(axis='x', rotation=45)

    # 3. Distribution des dons par jour de la semaine
    ax3 = fig.add_subplot(gs[1, 0])
    weekday_counts = df.groupby("Jour de la semaine")["Date_de_remplissage_de_la_fiche"].count()
    weekday_names = [calendar.day_name[i] for i in range(7)]
    weekday_counts.index = [weekday_names[i] for i in weekday_counts.index]
    weekday_counts.plot(kind='bar', color='lightgreen', ax=ax3)
    ax3.set_title("Distribution des dons par jour de la semaine", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Jour")
    ax3.set_ylabel("Nombre de dons")
    ax3.tick_params(axis='x', rotation=45)

    # 4. Tendance des dons au fil du temps
    ax4 = fig.add_subplot(gs[1, 1])
    time_series = df.groupby(pd.Grouper(key="Date_de_remplissage_de_la_fiche", freq='M')).size()
    time_series.plot(marker='o', linestyle='-', ax=ax4)
    ax4.set_title("Tendance des dons au fil du temps", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Nombre de dons")
    ax4.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle("Analyse des Dons de Sang – 2019", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    st.pyplot(fig)
    st.info("Cette analyse est basée sur la colonne 'Date_de_remplissage_de_la_fiche'.")

# Fonction pour créer une analyse de fidélisation des donneurs
@st.cache_data
def create_donor_retention_analysis(df):
    """
    Crée des visualisations pour analyser la fidélisation des donneurs.
    """
    #df=df_2019 if dataset=="2019" else (df=df_2020 if dataset=="2020" else df=df_Volontaire)
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
    Crée des visualisations stylisées pour l'analyse de sentiment des commentaires des donneurs (en français).
    """
    st.header("💬 Analyse de sentiment des commentaires")

    # Détection des colonnes textuelles
    comment_columns = [col for col in df.columns if any(term in col.lower() for term in ['préciser','autre', 'commentaire', 'feedback'])]

    if not comment_columns:
        st.warning("Aucune colonne de commentaires trouvée.")
        return None, None, None

    selected_col = comment_columns[0]
    comments_df = df[df[selected_col].notna() & (df[selected_col].astype(str).str.strip() != '')].copy()
    
    if comments_df.empty:
        st.warning("Aucun commentaire exploitable pour l'analyse.")
        return None, None, None

    # Initialisation analyseur de sentiment
    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

    # Fonction pour sentiment catégorique
    def analyze_sentiment(text):
        try:
            blob = tb(str(text))
            polarite = blob.sentiment[0]
            if polarite > 0.01:
                return "Positif"
            elif polarite < -0.01:
                return "Négatif"
            else:
                return "Neutre"
        except:
            return "Indéfini"

    # Fonction pour score brut
    def analyze_sentiments(text):
        try:
            blob = tb(str(text))
            return blob.sentiment[0]
        except:
            return None

    # Application des analyses
    comments_df['Sentiment'] = comments_df[selected_col].apply(analyze_sentiment)
    comments_df['Score'] = comments_df[selected_col].apply(analyze_sentiments)

    # 🔢 Résumés des résultats
    total_comments = len(comments_df)
    sentiment_counts = comments_df['Sentiment'].value_counts()
    avg_score = comments_df['Score'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='background-color:#E0F7FA;padding:15px;border-radius:10px;text-align:center'>
            <h3>{total_comments}</h3><p>Commentaires analysés</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        pos_pct = sentiment_counts.get("Positif", 0) / total_comments * 100
        st.markdown(f"""
        <div style='background-color:#E8F5E9;padding:15px;border-radius:10px;text-align:center'>
            <h3>{pos_pct:.1f}%</h3><p>Positifs</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style='background-color:#FFF3E0;padding:15px;border-radius:10px;text-align:center'>
            <h3>{avg_score:.2f}</h3><p>Score moyen</p>
        </div>""", unsafe_allow_html=True)

    # 📊 Graphique circulaire
    pie_df = sentiment_counts.reset_index()
    pie_df.columns = ['Sentiment', 'Nombre']
    fig1 = px.pie(pie_df, values='Nombre', names='Sentiment',
                  title="Répartition des sentiments",
                  color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig1, use_container_width=True)

    # ☁️ Nuage de mots
    st.subheader("☁️ Nuage de mots des commentaires")
    all_comments = ' '.join(comments_df[selected_col].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_comments)
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig2)

    # 🧾 Détails
    st.subheader("📝 Détails des commentaires analysés")
    st.dataframe(comments_df[[selected_col, 'Sentiment', 'Score']])

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
    st.header("🧾 Aperçu des données - Données 2019")

    # Bloc de résumé façon dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style='background-color:#DDEEFF;padding:15px;border-radius:10px;text-align:center'>
            <h3>{df_2019.shape[0]}</h3><p>Lignes</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color:#DDEEFF;padding:15px;border-radius:10px;text-align:center'>
            <h3>{df_2019.shape[1]}</h3><p>Colonnes</p>
        </div>""", unsafe_allow_html=True)

    missing_total = df_2019.isnull().sum().sum()
    with col3:
        st.markdown(f"""
        <div style='background-color:#FFDDDD;padding:15px;border-radius:10px;text-align:center'>
            <h3>{missing_total}</h3><p>Valeurs manquantes</p>
        </div>""", unsafe_allow_html=True)

    numeric_count = df_2019.select_dtypes(include=['int64', 'float64']).shape[1]
    with col4:
        st.markdown(f"""
        <div style='background-color:#E0FFE0;padding:15px;border-radius:10px;text-align:center'>
            <h3>{numeric_count}</h3><p>Colonnes numériques</p>
        </div>""", unsafe_allow_html=True)

    # Affichage preview
    st.subheader("🔍 Aperçu du DataFrame")
    st.dataframe(df_2019.head())

    # Statistiques descriptives
    st.subheader("📊 Statistiques descriptives")
    st.dataframe(df_2019.describe())

    # Valeurs manquantes
    st.subheader("❗ Valeurs manquantes")
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
        st.success("✅ Aucune valeur manquante dans les données 2019.")

        
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
    st.header(f"📊 Analyse des distributions - {sheet_name}")

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # ─────────────────────────────
    # 🔹 Bloc Résumé (statistiques clés)
    # ─────────────────────────────
    st.subheader("📌 Statistiques descriptives globales")

    if len(numeric_columns) > 0:
        stats_globales = df[numeric_columns].describe()

        total_obs = df.shape[0]
        moyennes = df[numeric_columns].mean()
        medianes = df[numeric_columns].median()
        ecarts_types = df[numeric_columns].std()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total d'observations", f"{total_obs}")
        with col2:
            st.metric("Moyenne globale", f"{moyennes.mean():,.2f}")
        with col3:
            st.metric("Écart-type global", f"{ecarts_types.mean():,.2f}")

    if len(categorical_columns) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Variables catégorielles", f"{len(categorical_columns)}")
        with col2:
            total_cat_values = sum(df[col].nunique() for col in categorical_columns)
            st.metric("Valeurs uniques totales", f"{total_cat_values}")

    st.markdown("---")

    # ─────────────────────────────
    # 🔹 Analyse des variables numériques
    # ─────────────────────────────
    if len(numeric_columns) > 0:
        st.subheader("📈 Variables numériques")
        selected_numeric = list(numeric_columns)[:min(5, len(numeric_columns))]

        colors = ['steelblue', 'lightseagreen', 'orangered', 'darkviolet', 'gold']
        for i, col in enumerate(selected_numeric):
            color = colors[i % len(colors)]
            col1, col2 = st.columns(2)

            # Histogramme avec KDE
            with col1:
                fig1, ax1 = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax1, color=color, alpha=0.7)
                ax1.set_title(f'Distribution de {col}')
                stat, p_value = stats.shapiro(df[col].dropna())
                normality = "normale" if p_value > 0.05 else "non normale"
                ax1.annotate(f'p = {p_value:.4f}\nDistribution {normality}',
                             xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", ec="gray", alpha=0.8),
                             ha='left', va='top')
                st.pyplot(fig1)

            # Boxplot avec stats descriptives
            with col2:
                fig2, ax2 = plt.subplots()
                sns.boxplot(x=df[col].dropna(), ax=ax2, color=color)
                ax2.set_title(f'Boxplot de {col}')
                stats_desc = df[col].describe()
                stats_text = (f"Moyenne: {stats_desc['mean']:.2f}\n"
                              f"Médiane: {stats_desc['50%']:.2f}\n"
                              f"Écart-type: {stats_desc['std']:.2f}\n"
                              f"Min: {stats_desc['min']:.2f}\n"
                              f"Max: {stats_desc['max']:.2f}")
                ax2.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", ec="gray", alpha=0.8),
                             ha='left', va='top')
                st.pyplot(fig2)

        # Comparaison globale avec Violin plot
        if len(selected_numeric) > 1:
            st.subheader("🎻 Comparaison globale (Violin plots)")
            fig = go.Figure()
            for col in selected_numeric:
                fig.add_trace(go.Violin(y=df[col].dropna(), name=col, box_visible=True, meanline_visible=True))
            fig.update_layout(title=f'Comparaison des distributions - {sheet_name}',
                              xaxis_title='Variables', yaxis_title='Valeurs', height=600,
                              template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        # ─────────────────────────────
        # 🔹 Analyse des variables catégorielles
        # ─────────────────────────────
        if len(categorical_columns) > 0:
            st.subheader("📊 Variables catégorielles")
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
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"<div style='background-color: #87CEFA; padding: 10px; border-radius: 5px; display: flex; align-items: center;'>"
                                        f"<strong>{val}</strong> <span style='margin-left: 10px;'>🔍</span></div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>"
                                        f"Count: {count} <br> Pourcentage: {value_counts_pct[val]:.1f}%</div>", unsafe_allow_html=True)



def analyze_blood_groups(df):
    """
    Analyse la distribution des groupes sanguins et phénotypes pour un dashboard Streamlit.
    """
    st.subheader("🔬 Analyse des groupes sanguins et phénotypes")

    if 'Groupe Sanguin ABO / Rhesus ' not in df.columns:
        st.warning("❗ Les informations sur les groupes sanguins ne sont pas disponibles dans ce jeu de données.")
        return

    # Distribution des groupes sanguins
    blood_group_counts = df['Groupe Sanguin ABO / Rhesus '].value_counts()
    blood_group_pct = df['Groupe Sanguin ABO / Rhesus '].value_counts(normalize=True) * 100

    # Palette de couleurs
    blood_colors = {
        'O+': '#e41a1c', 'O-': '#ff7f00', 'A+': '#4daf4a', 'A-': '#984ea3',
        'B+': '#377eb8', 'B-': '#ffff33', 'AB+': '#a65628', 'AB-': '#f781bf'
    }
    colors = [blood_colors.get(group, '#999999') for group in blood_group_counts.index]

    # Pourcentages arrondis
    blood_group_pct_numeric = pd.to_numeric(blood_group_pct, errors='coerce').round(1).fillna(0)
    text_labels = blood_group_pct_numeric.astype(str) + '%'

    # Graphiques combinés
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                        subplot_titles=['📊 Distribution (Barres)', '🧁 Distribution (Camembert)'])

    fig.add_trace(
        go.Bar(x=blood_group_counts.index, y=blood_group_counts.values,
               text=text_labels, textposition='outside', marker_color=colors),
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(labels=blood_group_counts.index, values=blood_group_counts.values,
               textinfo='label+percent', insidetextorientation='radial',
               marker=dict(colors=colors)),
        row=1, col=2
    )

    fig.update_layout(title='📌 Distribution des groupes sanguins',
                      showlegend=False, height=500)
    
    st.plotly_chart(fig, use_container_width=True)

    # Analyse des phénotypes
    if 'Phenotype ' in df.columns:
        st.markdown("### 🧬 Analyse des phénotypes")

        phenotype_data = {}
        for phenotype in df['Phenotype '].dropna():
            antigens = [a.strip() for a in phenotype.split(',')]
            for antigen in antigens:
                phenotype_data[antigen] = phenotype_data.get(antigen, 0) + 1

        phenotype_df = pd.DataFrame(list(phenotype_data.items()), columns=['Antigène', 'Fréquence'])
        phenotype_df = phenotype_df.sort_values('Fréquence', ascending=False)
        total_donors = len(df['Phenotype '].dropna())
        phenotype_df['Pourcentage'] = phenotype_df['Fréquence'] / total_donors * 100

        fig2 = px.bar(phenotype_df, x='Antigène', y='Pourcentage', text='Fréquence',
                      title='🔎 Fréquence des antigènes', color='Pourcentage',
                      color_continuous_scale='Viridis')

        fig2.update_layout(height=500, xaxis={'categoryorder': 'total descending'})
        fig2.update_traces(textposition='outside')

        st.plotly_chart(fig2, use_container_width=True)

        # Analyse croisée : groupe sanguin vs antigènes
        st.markdown("### 📈 Relation entre groupe sanguin et antigènes principaux")

        top_antigens = phenotype_df.head(5)['Antigène'].tolist()
        antigen_by_group = {}

        for blood_group in blood_group_counts.index:
            antigen_by_group[blood_group] = {}
            for antigen in top_antigens:
                subset = df[df['Groupe Sanguin ABO / Rhesus '] == blood_group]
                total = subset['Phenotype '].notna().sum()
                count = subset['Phenotype '].dropna().apply(lambda x: antigen in x).sum()
                antigen_by_group[blood_group][antigen] = (count / total * 100) if total > 0 else 0

        heatmap_df = pd.DataFrame(antigen_by_group).T

        fig3 = px.imshow(heatmap_df,
                         labels=dict(x="Antigène", y="Groupe sanguin", color="Pourcentage (%)"),
                         text_auto='.1f',
                         color_continuous_scale="Viridis",
                         aspect="auto",
                         title='🔥 Fréquence des antigènes par groupe sanguin (%)')

        st.plotly_chart(fig3, use_container_width=True)
        


# Interface principale du tableau de bord
def main():
    # === Chargement des données ===
    df_2019, df_2020, df_volontaire = load_data()
    
    # === Barre latérale avec menu de navigation ===
    st.sidebar.image('Image_sang.jpg')
    
    with st.sidebar:
        page = option_menu(
            "Navigation",
            ["Accueil", "Aperçu des données", "Distribution géographique", "Santé et éligibilité",
             "Profils des donneurs", "Analyse des campagnes", "Fidélisation des donneurs",
             "Analyse de sentiment", "Prédiction d'éligibilité", "Bonus"],
            icons=["house", "bar-chart", "geo-alt", "heart-pulse", "people", "megaphone",
                   "person-check", "chat-dots", "cpu", "gift"],
            menu_icon="cast",
            default_index=0,
        )
    
        st.markdown("---")  # séparation visuelle
    
        # === Sélection du jeu de données ===
        st.subheader("Jeu de données")
        dataset = st.radio(
            "Sélectionnez un jeu de données",
            ["2019", "Volontaire", "2020"]
        )

    # === Affichage de la page sélectionnée ===
    st.write(f"Page sélectionnée : {page}")
    
   
    # === Logique d'affichage par page ===
    # Page d'accueil
    if page == "Accueil":
        st.markdown("""
            <style>
            .block-container {
                padding: 0rem;
                margin: 0 auto;
                max-width: 100%;
            }
            .card {
                background-color: #0a0f3c;
                padding: 1rem;
                border-radius: 15px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
                color: white;
            }
            .centered {
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)
    
        st.markdown('''<h1 class="centered" style="color:#ffffff;">🩸 Tableau de Bord d'Analyse des Donneurs de Sang</h1>''', unsafe_allow_html=True)
        st.markdown('<p class="centered" style="font-size:18px; color:#b0b0b0;">Optimisation des campagnes de don et visualisation des données médicales</p>', unsafe_allow_html=True)
    
        # ========== Section 1 : Répartition géographique et types de donneurs ==========
        col1, col2 = st.columns([3, 2])
    
        with col1:
            st.subheader("📍 Répartition géographique des dons")
            cameroon_coords = {
                'Centre': (3.848, 11.502),
                'Littoral': (4.05, 9.7),
                'Nord-Ouest': (6.25, 10.26),
                'Ouest': (5.49, 10.42),
                'Adamaoua': (7.3, 13.58)
            }
            region_data = pd.DataFrame({
                'Region': list(cameroon_coords.keys()),
                'Lat': [v[0] for v in cameroon_coords.values()],
                'Lon': [v[1] for v in cameroon_coords.values()],
                'Dons': [150, 300, 90, 120, 180]
            })
            fig_map = px.scatter_mapbox(region_data, lat='Lat', lon='Lon', size='Dons', color='Region',
                                        hover_name='Region', zoom=5.5, height=400, mapbox_style="carto-positron")
            fig_map.update_traces(marker=dict(sizemode='area', opacity=0.6))
            st.plotly_chart(fig_map, use_container_width=True)
    
        with col2:
            st.subheader("🔘 Type de donneurs")
            pie_data = pd.DataFrame({
                "Type": ["Volontaire", "Familial", "Rémunéré"],
                "Pourcentage": [58, 22, 20]
            })
            fig_pie = px.pie(pie_data, values='Pourcentage', names='Type',
                             color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4)
            fig_pie.update_traces(pull=[0.05, 0, 0.05], textinfo='label+percent')
            st.plotly_chart(fig_pie, use_container_width=True)
    
        st.markdown("---")
    
        # ========== Section 2 : Tableau de contrôle et engagement ==========
        col3, col4, col5 = st.columns(3)
        with col3:
            freq = st.slider("🩺 Fréquence des dons", 0, 10, 6)
        with col4:
            retention = st.slider("📅 Fidélité", 0, 100, 45)
        with col5:
            satisfaction = st.slider("😊 Satisfaction", 0, 100, 80)
    
        score_mix = (freq * 10 + retention + satisfaction) / 3
        st.markdown(f"<h3 style='text-align:center; color:#29d8db;'>⭐ Score global d'engagement : {score_mix:.1f}%</h3>", unsafe_allow_html=True)
    
        # Radar chart dynamique
        radar_data = pd.DataFrame({
            'Critères': ['Fréquence', 'Fidélité', 'Satisfaction'],
            'Score': [freq * 10, retention, satisfaction]
        })
    
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data['Score'],
            theta=radar_data['Critères'],
            fill='toself',
            line_color='deepskyblue',
            name='Engagement'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                showlegend=False, height=400)
        st.plotly_chart(fig_radar, use_container_width=True)
    
        # ========== Section 3 : Graphiques animés supplémentaires ==========
        col6, col7 = st.columns(2)
    
        with col6:
            st.subheader("📊 Suivi des dons sur l’année")
            mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
            dons = np.random.randint(80, 300, size=12)
            fig_bar = px.bar(x=mois, y=dons, labels={'x': 'Mois', 'y': 'Nombre de dons'},
                             color=dons, color_continuous_scale='Bluered_r')
            fig_bar.update_traces(marker_line_color='white', marker_line_width=1.5)
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    
        with col7:
            st.subheader("🫀 Simulation du rythme cardiaque")
            cardiogramme = np.sin(np.linspace(0, 20, 200)) * np.exp(-0.05 * np.linspace(0, 20, 200))
            fig_cardio, ax = plt.subplots()
            ax.plot(cardiogramme, color="red", linewidth=2)
            ax.set_title("Signal cardiaque simulé", color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            fig_cardio.set_facecolor("#f9f9f9")
            st.pyplot(fig_cardio)
    
        st.markdown("---")
        st.markdown("<p style='text-align:center; color:#808080;'>© 2025 Plateforme d'analyse des dons de sang</p>", unsafe_allow_html=True)

# Tu peux ensuite continuer à ajouter les autres blocs : Aperçu des données, Distribution, etc.


    

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
            analyze_blood_groups(df_2020)
        else:
            visualize_missing_values(df_volontaire, "Données Volontaire")
            analyze_distributions(df_volontaire, "Données Volontaire")
            analyze_blood_groups(df_volontaire)
       # CSS personnalisé pour styling clair et propre
        # CSS personnalisé
        # CSS personnalisé
        st.markdown("""
            <style>
                .kpi-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    margin: 10px;
                }
        
                .kpi-top {
                    background-color: #d0e7ff;
                    padding: 10px;
                    border-radius: 12px;
                    text-align: center;
                    width: 100%;
                    font-size: 16px;
                    font-weight: bold;
                    color: #003366;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 8px;
                }
        
                .kpi-bottom {
                    border: 2px solid #d0d0d0;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    width: 100%;
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                    margin-top: 12px;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Affichage dans les colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-top">🩸 Donneurs 2019</div>
                    <div class="kpi-bottom">{len(df_2019)}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-top">🩸 Donneurs 2020</div>
                    <div class="kpi-bottom">{len(df_2020)}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-top">🩸 Volontaires</div>
                    <div class="kpi-bottom">{len(df_volontaire)}</div>
                </div>
            """, unsafe_allow_html=True)



        
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
                    
                    # Layout avec deux colonnes
                    col1, col2 = st.columns(2)
                    
                    # Colonne 1: Caractéristiques démographiques
                    with col1:
                        st.markdown("### **Caractéristiques démographiques**")
                        
                        # Âge moyen
                        if 'Age' in ideal_profile.columns:
                            # Bloc du titre
                            st.markdown(
                                f"<div style='background-color: lightblue; padding: 10px; border-radius: 5px; display: flex; align-items: center;'>"
                                f"<strong>Âge moyen</strong><i class='fa fa-calendar' style='margin-left: 10px;'></i></div>",
                                unsafe_allow_html=True
                            )
                            # Bloc de la valeur
                            st.markdown(
                                f"<div style='border: 2px solid black; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
                                f"<strong>{ideal_profile['Age'].mean():.1f} ans</strong></div>",
                                unsafe_allow_html=True
                            )
                        
                        # Genre
                        if 'Genre_' in ideal_profile.columns:
                            gender_pct = ideal_profile['Genre_'].value_counts(normalize=True) * 100
                            # Bloc du titre
                            st.markdown(
                                f"<div style='background-color: lightblue; padding: 10px; border-radius: 5px; display: flex; align-items: center;'>"
                                f"<strong>Genre</strong><i class='fa fa-user' style='margin-left: 10px;'></i></div>",
                                unsafe_allow_html=True
                            )
                            # Bloc de la valeur
                            st.markdown(
                                f"<div style='border: 2px solid black; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
                                f"<strong>{gender_pct.get('Homme', 0):.1f}% Hommes, {gender_pct.get('Femme', 0):.1f}% Femmes</strong></div>",
                                unsafe_allow_html=True
                            )
                        
                        # Niveau d'étude
                        if 'Niveau_d\'etude' in ideal_profile.columns:
                            top_edu = ideal_profile['Niveau_d\'etude'].value_counts(normalize=True).head(2)
                            for edu, pct in top_edu.items():
                                # Bloc du titre
                                st.markdown(
                                    f"<div style='background-color: lightblue; padding: 10px; border-radius: 5px; display: flex; align-items: center;'>"
                                    f"<strong>Niveau d'études: {edu}</strong><i class='fa fa-graduation-cap' style='margin-left: 10px;'></i></div>",
                                    unsafe_allow_html=True
                                )
                                # Bloc de la valeur
                                st.markdown(
                                    f"<div style='border: 2px solid black; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
                                    f"<strong>{pct*100:.1f}%</strong></div>",
                                    unsafe_allow_html=True
                                )
                        
                        # Situation matrimoniale
                        if 'Situation_Matrimoniale_(SM)' in ideal_profile.columns:
                            top_marital = ideal_profile['Situation_Matrimoniale_(SM)'].value_counts(normalize=True).head(2)
                            for status, pct in top_marital.items():
                                # Bloc du titre
                                st.markdown(
                                    f"<div style='background-color: lightblue; padding: 10px; border-radius: 5px; display: flex; align-items: center;'>"
                                    f"<strong>Situation matrimoniale: {status}</strong><i class='fa fa-heart' style='margin-left: 10px;'></i></div>",
                                    unsafe_allow_html=True
                                )
                                # Bloc de la valeur
                                st.markdown(
                                    f"<div style='border: 2px solid black; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
                                    f"<strong>{pct*100:.1f}%</strong></div>",
                                    unsafe_allow_html=True
                                )
                    
                    # Colonne 2: Caractéristiques géographiques
                    with col2:
                        st.markdown("### **Caractéristiques géographiques**")
                        
                        geo_columns = [col for col in ideal_profile.columns if any(term in col for term in ['Arrondissement', 'Quartier', 'Résidence'])]
                        for geo_col in geo_columns:
                            top_geo = ideal_profile[geo_col].value_counts(normalize=True).head(3)
                            for zone, pct in top_geo.items():
                                # Bloc du titre
                                st.markdown(
                                    f"<div style='background-color: lightblue; padding: 10px; border-radius: 5px; display: flex; align-items: center;'>"
                                    f"<strong>{geo_col} principal: {zone}</strong><i class='fa fa-map-marker' style='margin-left: 10px;'></i></div>",
                                    unsafe_allow_html=True
                                )
                                # Bloc de la valeur
                                st.markdown(
                                    f"<div style='border: 2px solid black; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
                                    f"<strong>{pct*100:.1f}%</strong></div>",
                                    unsafe_allow_html=True
                                )


                    
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
            if dataset=="2019":
                create_campaign_analysis(df_2019)
            elif dataset=="2020":
                create_campaign_analysis(df_2020)
            else:
                create_campaign_analysis(df_volontaire)
            
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
        else :
            st.warning("Impossible d'analyser les tendances temporelles car aucune colonne de date appropriée n'a été identifiée.")
    
    elif page == "Fidélisation des donneurs":
        # Afficher l'image en haut, sur toute la largeur
        image_file="Fidélisation.jpg"
        image = Image.open(image_file)
        st.image(image, use_container_width=True)  # ✅ remplace use_column_width     
        st.header("🔄 Fidélisation des donneurs")

        if dataset == "2019":
            df=df_2019
        elif dataset == "2020" :
            df=df_2020
        else :
            df=df_volontaire
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
            #pyg_html = pyg.walk(df_selected, output_type='html')
            #components.html(pyg_html, height=500, scrolling=True)
            pyg_app=StreamlitRenderer(df_selected)
            pyg_app.explorer()
    
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
                    #pyg_html = pyg.walk(data, output_type='html')
                    #components.html(pyg_html, height=500, scrolling=True)
                    pyg_app=StreamlitRenderer(data)
                    pyg_app.explorer()

        
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
            # 🎯 Chargement du modèle & données de test
            # ==============================
            @st.cache_resource
            def load_model():
                pathse = "eligibility_model.pkl"
                data = joblib.load(pathse)
                return data["model"], data["X_test"], data["y_test"], data["target_encoder"], data["lpreprocessor"], data["resultat"]
            
            model, X_test, y_test, target_encoder, preprocessor, resultat = load_model()
            
            # ==============================
            # 📈 Évaluation du modèle
            # ==============================
            st.header("🧠 Évaluation du modèle de prédiction d'éligibilité")
            
            # 🔮 Prédictions
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            
            # 📄 Rapport de classification
            A = target_encoder.inverse_transform([0, 1, 2])
            report = classification_report(y_test, y_pred, target_names=A, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            
            # Résumé rapide avec les principales métriques
            st.subheader("🔍 Résumé des performances")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Précision (macro)", f"{report['macro avg']['precision']:.2f}")
            with col2:
                st.metric("Recall (macro)", f"{report['macro avg']['recall']:.2f}")
            with col3:
                st.metric("F1-score (macro)", f"{report['macro avg']['f1-score']:.2f}")
            
            st.subheader("📋 Rapport complet")
            st.dataframe(df_report)
            
            st.subheader("📊 Résultats détaillés")
            st.dataframe(resultat)
            
            # ==============================
            # 📈 Courbe ROC (One vs Rest)
            # ==============================
            st.subheader("🧪 Courbe ROC par classe")
            
            n_classes = len(np.unique(y_test))
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
            y_pred_proba_all = model.predict_proba(X_test)
            
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ['blue', 'red', 'green', 'purple', 'orange']
            
            for i in range(n_classes):
                if n_classes > 2:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba_all[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                            label=f"Classe {A[i]} (AUC = {roc_auc:.2f})")
                else:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_all[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax.set_xlabel("Taux de Faux Positifs")
            ax.set_ylabel("Taux de Vrais Positifs")
            ax.set_title("Courbe ROC")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            
            # ==============================
            # 🧱 Matrice de confusion
            # ==============================
            st.subheader("🔀 Matrice de confusion")
            B = target_encoder.inverse_transform(y_test)
            C = target_encoder.inverse_transform(y_pred)
            conf_matrix = confusion_matrix(B, C)
            
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Matrice de Confusion")
            st.pyplot(fig)
            
            # ==============================
            # 📊 Probabilités d'un exemple
            # ==============================
            st.subheader("📌 Probabilités pour un exemple")
            proba_df = pd.DataFrame({
                'Statut': model.classes_,
                'Probabilité': y_pred_proba[0]
            })
            
            fig_proba = px.bar(
                proba_df,
                x='Statut',
                y='Probabilité',
                title="Distribution des probabilités pour chaque classe",
                labels={'Probabilité': 'Probabilité', 'Statut': 'Statut'}
            )
            fig_proba.update_layout(height=400)
            st.plotly_chart(fig_proba, use_container_width=True)
            
            # ==============================
            # 🧠 Importance des caractéristiques
            # ==============================
            if hasattr(model, 'feature_importances_'):
                st.subheader("📊 Importance des caractéristiques")
            
                feature_names = []
                for name, transformer, features in preprocessor.transformers_:
                    if name == 'cat':
                        for i, feature in enumerate(features):
                            categories = transformer.named_steps['onehot'].categories_[i]
                            for category in categories:
                                feature_names.append(f"{feature}_{category}")
                    else:
                        feature_names.extend(features)
            
                importances = model.feature_importances_
                if len(feature_names) == len(importances):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(15)
            
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 caractéristiques les plus importantes",
                        labels={'Importance': 'Importance relative', 'Feature': 'Caractéristique'}
                    )
                    fig_importance.update_layout(height=600)
                    st.plotly_chart(fig_importance, use_container_width=True)
       
                    
    
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
