import os
import re
import base64
import string
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from unidecode import unidecode
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# ============================================================
# RUTAS RELATIVAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TWITTER_CSV = os.path.join(BASE_DIR, "data", "tweets_data_twscrape.csv")
FACEBOOK_EXCEL = os.path.join(BASE_DIR, "data", "facebook_data.xlsx")

# ============================================================
# FUNCIONES Y STOPWORDS (sin cambios)
# ============================================================

SPANISH_STOPWORDS = set([
    "del","de","la","las","los","un","una","unos","unas","el","y","o","e",
    "a","al","en","por","para","con","sin","se","que","su","sus","ya","si",
    "esto","eso","este","ese","esa","esas","esos","como","es","pero",
    "todo","todos","todas","desde","han","ser","son","fue","fueron",
    "mas","lo","estado","ano","anos","www","http","https","toda","sobre",
    "va","vez","aqui","asi","cada","hoy","dia","muy","tambien","nos",
    "nuestro","nuestra","hace","entre","donde","favor","durante"
])

PALABRAS_INST = set([
    "world","vision","worldvision","visionmexico","mexico","ecuador","peru",
    "honduras","bolivia","republica","dominicana","colombia","salvador",
    "guatemala","nicaragua","venezuela","costa","rica","proyecto",
    "proyectos","programa","programas","ninas","ninos","ninez"
])

STOPWORDS_TOTAL = STOPWORDS.union(SPANISH_STOPWORDS).union(PALABRAS_INST)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text.strip()

# ============================================================
# CARGA DATOS
# ============================================================
try:
    fb_seguidores = pd.read_excel(FACEBOOK_EXCEL, sheet_name="Seguidores")
    fb_interacciones = pd.read_excel(FACEBOOK_EXCEL, sheet_name="Interacciones")
    fb_visualizaciones = pd.read_excel(FACEBOOK_EXCEL, sheet_name="Visualizaciones")
    fb_visitas = pd.read_excel(FACEBOOK_EXCEL, sheet_name="Visitas")
    fb_clics = pd.read_excel(FACEBOOK_EXCEL, sheet_name="Clics")

    for df in [fb_seguidores, fb_interacciones, fb_visualizaciones, fb_visitas, fb_clics]:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df["Primary"] = pd.to_numeric(df["Primary"], errors="coerce")
except FileNotFoundError:
    print("Archivo de Facebook no encontrado. Revisa la carpeta 'data/'")
    fb_seguidores = fb_interacciones = fb_visualizaciones = fb_visitas = fb_clics = pd.DataFrame()

try:
    twitter = pd.read_csv(TWITTER_CSV)
except FileNotFoundError:
    print("Archivo de Twitter no encontrado. Revisa la carpeta 'data/'")
    twitter = pd.DataFrame()

# ============================================================
# (Aquí seguiría todo tu código de limpieza, funciones, dashboard, callbacks)
# Solo cambiar TWITTER_CSV y FACEBOOK_EXCEL a rutas relativas
# ============================================================

# Al final:
if __name__ == "__main__":
    app.run(debug=True)
