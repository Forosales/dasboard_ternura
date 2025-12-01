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
# CONFIGURACIÓN RUTAS RELATIVAS (FUNCIONA EN HEROKU)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TWITTER_CSV = os.path.join(DATA_DIR, "tweets_data_twscrape.csv")
FACEBOOK_EXCEL = os.path.join(DATA_DIR, "facebook_data.xlsx")

# ============================================================
# STOPWORDS
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

# ============================================================
# LIMPIEZA TEXTO
# ============================================================
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
# CARGA FACEBOOK
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

except Exception as e:
    print("ERROR cargando Facebook:", e)
    fb_seguidores = fb_interacciones = fb_visualizaciones = fb_visitas = fb_clics = pd.DataFrame()

# ============================================================
# CARGA TWITTER
# ============================================================
try:
    twitter = pd.read_csv(TWITTER_CSV)
except Exception as e:
    print("ERROR cargando Twitter:", e)
    twitter = pd.DataFrame()

def clean_twitter(df):
    if df.empty:
        return df

    df["fecha"] = pd.to_datetime(df["fecha_iso"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["fecha"])
    df["texto_limpio"] = df["texto"].astype(str).apply(clean_text)

    for col in ["usuario", "nombre", "pais"]:
        df[col] = df[col].astype(str).str.strip()

    for col in ["retweets", "likes", "replies"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

# Limpieza
if not twitter.empty:
    twitter = clean_twitter(twitter)

    # Sentimiento
    analyzer = SentimentIntensityAnalyzer()
    twitter["sentimiento"] = twitter["texto_limpio"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    twitter["sent_cat"] = twitter["sentimiento"].apply(
        lambda x: "positivo" if x > 0.05 else "negativo" if x < -0.05 else "neutro"
    )

    # LDA
    def compute_lda(df):
        if df["texto_limpio"].str.len().sum() < 20:
            return pd.DataFrame({"cluster":[0],"words":["sin datos"]})
        vect = CountVectorizer(stop_words=list(STOPWORDS_TOTAL), max_features=1500)
        dtm = vect.fit_transform(df["texto_limpio"])
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)
        words = vect.get_feature_names_out()
        return pd.DataFrame([
            {"cluster": i, "words": ", ".join([words[j] for j in comp.argsort()[-10:]])}
            for i, comp in enumerate(lda.components_)
        ])

    twitter_topics = compute_lda(twitter)

    start = twitter["fecha"].min().strftime("%Y-%m-%d")
    end   = twitter["fecha"].max().strftime("%Y-%m-%d")
else:
    start = end = None


# ============================================================
# DASH – CREACIÓN APP
# ============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server   # << NECESARIO PARA HEROKU

# ============================================================
# LAYOUT COMPLETO
# ============================================================
app.layout = dbc.Container([
    html.H2("📊 Dashboard Integrado – Ternura"),

    dcc.Tabs(id="tabs", value="tab-twitter", children=[
        dcc.Tab(label="Twitter / X", value="tab-twitter", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Tweets totales"),
                    html.H3(len(twitter))
                ]), color="info", inverse=True)),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Sentimiento"),
                    html.H3(round(twitter["sentimiento"].mean(),3) if not twitter.empty else "0")
                ]), color="success", inverse=True)),
            ]),
            html.Br(),
            dcc.Dropdown(id="tema",
                options=[
                    {"label":"Todos","value":"todos"},
                    {"label":"Conexión Ternura","value":"ternura"},
                    {"label":"Bálsamo de Ternura","value":"balsamo"}
                ],
                value="todos",
                clearable=False
            ),
            dcc.Dropdown(
                id="pais",
                options=[{"label":p.capitalize(),"value":p} for p in sorted(twitter["pais"].unique())] if not twitter.empty else [],
                value="todos",
                clearable=False
            ),
            dcc.DatePickerRange(
                id="rango",
                start_date=start,
                end_date=end,
                min_date_allowed=start,
                max_date_allowed=end
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id="twt-ts")),
                dbc.Col(dcc.Graph(id="twt-sent")),
            ]),
        ])
    ])
], fluid=True)

# ============================================================
# CALLBACKS (puedes pegar el resto de tus callbacks aquí)
# ============================================================


# ============================================================
# RUN LOCAL
# ============================================================
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050)
