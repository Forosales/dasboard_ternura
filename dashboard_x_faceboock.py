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
DATA_DIR = os.path.join(BASE_DIR, "data")

TWITTER_CSV = os.path.join(DATA_DIR, "tweets_data_twscrape.csv")
FACEBOOK_EXCEL = os.path.join(DATA_DIR, "facebook_data.xlsx")
INSTAGRAM_EXCEL = os.path.join(DATA_DIR, "instagram.xlsx")

# ============================================================
# STOPWORDS
# ============================================================
SPANISH_STOPWORDS = {
    "del","de","la","las","los","un","una","unos","unas","el","y","o","e",
    "a","al","en","por","para","con","sin","se","que","su","sus","ya","si",
    "esto","eso","este","ese","esa","esas","esos","como","es","pero",
    "todo","todos","todas","desde","han","ser","son","fue","fueron",
    "mas","lo","estado","ano","anos","www","http","https","toda","sobre",
    "va","vez","aqui","asi","cada","hoy","dia","muy","tambien","nos",
    "nuestro","nuestra","hace","entre","donde","favor","durante", "esta", "te", "lleva",
    "nota", "tu", "chupa chups", "junto", "q estamos", "ante", "sido", "porque", "les",
    "hemos", "ha", "segun", "traves", "tiene", "san", "pm", "solo", "lleva", "er", "junto"
}

PALABRAS_INST = {
    "world","vision","worldvision","visionmexico","mexico","ecuador","peru",
    "honduras","bolivia","republica","dominicana","colombia","salvador",
    "guatemala","nicaragua","venezuela","costa","rica","proyecto",
    "proyectos","programa","programas","ninas","ninos","ninez"
}

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

except:
    fb_seguidores = fb_interacciones = fb_visualizaciones = fb_visitas = fb_clics = pd.DataFrame()

# ============================================================
# CARGA INSTAGRAM
# ============================================================
try:
    ig_seguidores = pd.read_excel(INSTAGRAM_EXCEL, sheet_name="Seguidores")
    ig_interacciones = pd.read_excel(INSTAGRAM_EXCEL, sheet_name="Interacciones")
    ig_visualizaciones = pd.read_excel(INSTAGRAM_EXCEL, sheet_name="Visualizaciones")
    ig_visitas = pd.read_excel(INSTAGRAM_EXCEL, sheet_name="Visitas")
    ig_clics = pd.read_excel(INSTAGRAM_EXCEL, sheet_name="Clics")
    ig_alcance = pd.read_excel(INSTAGRAM_EXCEL, sheet_name="Alcance")

    for df in [ig_seguidores, ig_interacciones, ig_visualizaciones, ig_visitas, ig_clics, ig_alcance]:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df["Primary"] = pd.to_numeric(df["Primary"], errors="coerce")

except:
    ig_seguidores = ig_interacciones = ig_visualizaciones = ig_visitas = ig_clics = ig_alcance = pd.DataFrame()

# ============================================================
# CARGA TWITTER
# ============================================================
try:
    twitter = pd.read_csv(TWITTER_CSV)
except:
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

if not twitter.empty:
    twitter = clean_twitter(twitter)

    analyzer = SentimentIntensityAnalyzer()
    twitter["sentimiento"] = twitter["texto_limpio"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    twitter["sent_cat"] = twitter["sentimiento"].apply(
        lambda x: "positivo" if x > 0.05 else "negativo" if x < -0.05 else "neutro"
    )

# ============================================================
# DASHBOARD
# ============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server

def card(title, value, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title),
            html.H3(value)
        ]),
        color=color,
        inverse=True
    )

# ============================================================
# LAYOUT
# ============================================================
app.layout = dbc.Container([

    html.H2("📊 Dashboard Integrado – Ternura"),

    dcc.Tabs(id="tabs", value="tab-twitter", children=[

        # -------------------------------------------------------
        # TWITTER
        # -------------------------------------------------------
        dcc.Tab(label="Twitter / X", value="tab-twitter", children=[
            html.Br(),

            dbc.Row([
                dbc.Col(card("Tweets totales", len(twitter), "info")),
                dbc.Col(card("Sentimiento",
                             round(twitter["sentimiento"].mean(),3) if not twitter.empty else 0, "success")),
                dbc.Col(card("RT totales",
                             twitter["retweets"].sum() if not twitter.empty else 0, "primary")),
                dbc.Col(card("Likes totales",
                             twitter["likes"].sum() if not twitter.empty else 0, "warning")),
            ]),

            html.Br(),

            dcc.Dropdown(
                id="tema",
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
                options=[{"label":p.capitalize(),"value":p}
                         for p in sorted(twitter["pais"].unique())] if not twitter.empty else [],
                value="todos",
                clearable=False
            ),

            dcc.Dropdown(
                id="periodo-twitter",
                options=[
                    {"label":"Día","value":"D"},
                    {"label":"Semana","value":"W"},
                    {"label":"Mes","value":"M"}
                ],
                value="D",
                clearable=False
            ),

            dcc.DatePickerRange(
                id="rango",
                start_date=twitter["fecha"].min() if not twitter.empty else None,
                end_date=twitter["fecha"].max() if not twitter.empty else None
            ),

            dbc.Row([
                dbc.Col(dcc.Graph(id="twt-ts")),
                dbc.Col(dcc.Graph(id="twt-sent"))
            ]),

            dbc.Row([
                dbc.Col(dcc.Graph(id="twt-map")),
                dbc.Col(dcc.Graph(id="twt-comp"))
            ]),

            dbc.Row([
                dbc.Col(dcc.Graph(id="twt-top-users")),
                dbc.Col(dcc.Graph(id="twt-top-rt"))
            ]),

            dbc.Row([
                dbc.Col(dcc.Graph(id="twt-top-likes")),
                dbc.Col(html.Img(id="twt-wc"))
            ]),

            html.H3("Temas (LDA)"),
            dcc.Graph(id="twt-lda"),
        ]),

        # -------------------------------------------------------
        # FACEBOOK
        # -------------------------------------------------------
        dcc.Tab(label="Facebook", value="tab-facebook", children=[
            html.Br(),

            dbc.Row([
                dbc.Col(card("Seguidores",
                    fb_seguidores["Primary"].sum() if not fb_seguidores.empty else 0, "primary")),
                dbc.Col(card("Interacciones",
                    fb_interacciones["Primary"].sum() if not fb_interacciones.empty else 0, "success")),
                dbc.Col(card("Visualizaciones",
                    fb_visualizaciones["Primary"].sum() if not fb_visualizaciones.empty else 0, "info")),
                dbc.Col(card("Clics",
                    fb_clics["Primary"].sum() if not fb_clics.empty else 0, "danger")),
                dbc.Col(card("Visitas",
                    fb_visitas["Primary"].sum() if not fb_visitas.empty else 0, "secondary")),
            ]),

            html.Br(),

            dcc.Dropdown(
                id="periodo-fb",
                options=[
                    {"label":"Día","value":"D"},
                    {"label":"Semana","value":"W"},
                    {"label":"Mes","value":"M"},
                ],
                value="D",
                clearable=False
            ),

            dcc.Graph(id="fb-seguid"),
            dcc.Graph(id="fb-inter"),
            dcc.Graph(id="fb-vis"),
            dcc.Graph(id="fb-clics"),
            dcc.Graph(id="fb-visitas"),
        ]),

        # -------------------------------------------------------
        # INSTAGRAM
        # -------------------------------------------------------
        dcc.Tab(label="Instagram", value="tab-instagram", children=[
            html.Br(),

            dbc.Row([
                dbc.Col(card("Seguidores",
                    ig_seguidores["Primary"].sum() if not ig_seguidores.empty else 0, "primary")),
                dbc.Col(card("Interacciones",
                    ig_interacciones["Primary"].sum() if not ig_interacciones.empty else 0, "success")),
                dbc.Col(card("Visualizaciones",
                    ig_visualizaciones["Primary"].sum() if not ig_visualizaciones.empty else 0, "info")),
                dbc.Col(card("Clics",
                    ig_clics["Primary"].sum() if not ig_clics.empty else 0, "danger")),
                dbc.Col(card("Visitas",
                    ig_visitas["Primary"].sum() if not ig_visitas.empty else 0, "secondary")),
                dbc.Col(card("Alcance",
                    ig_alcance["Primary"].sum() if not ig_alcance.empty else 0, "warning")),
            ]),

            html.Br(),

            dcc.Dropdown(
                id="periodo-ig",
                options=[
                    {"label":"Día","value":"D"},
                    {"label":"Semana","value":"W"},
                    {"label":"Mes","value":"M"},
                ],
                value="D",
                clearable=False
            ),

            dcc.Graph(id="ig-seguid"),
            dcc.Graph(id="ig-inter"),
            dcc.Graph(id="ig-vis"),
            dcc.Graph(id="ig-clics"),
            dcc.Graph(id="ig-visitas"),
            dcc.Graph(id="ig-alcance"),
        ]),

        # -------------------------------------------------------
        # COMPARACIÓN
        # -------------------------------------------------------
        dcc.Tab(label="Comparación", value="tab-compare", children=[
            html.Br(),

            dbc.Row([
                dbc.Col(card("Total Twitter",
                    len(twitter) if not twitter.empty else 0, "info")),
                dbc.Col(card("Total Facebook",
                    fb_interacciones["Primary"].sum() if not fb_interacciones.empty else 0, "primary")),
                dbc.Col(card("Total Instagram",
                    ig_interacciones["Primary"].sum() if not ig_interacciones.empty else 0, "danger")),
            ]),

            dcc.Graph(id="cmp-total"),
            dcc.Graph(id="cmp-tend"),
        ]),
    ])
], fluid=True)

# ============================================================
# CALLBACKS TWITTER
# ============================================================
@app.callback(
    [
        Output("twt-ts","figure"),
        Output("twt-sent","figure"),
        Output("twt-map","figure"),
        Output("twt-comp","figure"),
        Output("twt-top-users","figure"),
        Output("twt-top-rt","figure"),
        Output("twt-top-likes","figure"),
        Output("twt-wc","src"),
        Output("twt-lda","figure"),
    ],
    [
        Input("tema","value"),
        Input("pais","value"),
        Input("periodo-twitter","value"),
        Input("rango","start_date"),
        Input("rango","end_date")
    ]
)
def update_twitter(tema, pais, periodo, start, end):

    if twitter.empty:
        empty = go.Figure()
        return empty, empty, empty, empty, empty, empty, empty, None, empty

    dff = twitter.copy()

    # Filtrar por tema
    if tema == "ternura":
        dff = dff[dff["texto_limpio"].str.contains("ternur", na=False)]
    elif tema == "balsamo":
        dff = dff[dff["texto_limpio"].str.contains("balsam", na=False)]

    # Filtrar por país
    if pais != "todos":
        dff = dff[dff["pais"] == pais]

    # Filtrar por fechas
    try:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        dff = dff[(dff["fecha"] >= start) & (dff["fecha"] <= end)]
    except:
        pass

    if dff.empty:
        empty = go.Figure()
        return empty, empty, empty, empty, empty, empty, empty, None, empty

    # ----- Serie de tiempo
    ts = dff.set_index("fecha").resample(periodo).size().reset_index(name="tweets")
    fig_ts = px.line(ts, x="fecha", y="tweets", title="Tweets por periodo")

    # ----- Sentimiento
    sent = dff.set_index("fecha")["sentimiento"].resample(periodo).mean().reset_index()
    fig_sent = px.line(sent, x="fecha", y="sentimiento", title="Sentimiento promedio")

    # ----- Mapa
    mapa_df = dff.groupby("pais").size().reset_index(name="tweets")
    fig_map = px.choropleth(
        mapa_df,
        locations="pais",
        locationmode="country names",
        color="tweets",
        title="Tweets por país"
    )

    # ----- Comparación sentimiento
    comp = dff.groupby("pais")["sentimiento"].mean().reset_index()
    fig_comp = px.bar(comp, x="pais", y="sentimiento", title="Sentimiento por país")

    # ----- Top usuarios
    top_users = dff["usuario"].value_counts().nlargest(10)
    fig_users = px.bar(top_users, title="Top usuarios")

    # ----- Top retweets
    fig_rt = px.bar(dff.nlargest(10,"retweets"), x="usuario", y="retweets", title="Top retweets")

    # ----- Top likes
    fig_likes = px.bar(dff.nlargest(10,"likes"), x="usuario", y="likes", title="Top likes")

    # ----- Wordcloud
    text = " ".join(dff["texto_limpio"])
    wc_path = os.path.join("/tmp", "wc.png")

    if len(text.strip()) > 10:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="Dark2",
            stopwords=STOPWORDS_TOTAL
        ).generate(text)

        wc.to_file(wc_path)

        with open(wc_path, "rb") as f:
            wc_base64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    else:
        wc_base64 = None

    # ----- LDA
    if dff["texto_limpio"].str.len().sum() < 30:
        lda_df = pd.DataFrame({"cluster":[0], "words":["sin datos"]})
    else:
        vect = CountVectorizer(stop_words=list(STOPWORDS_TOTAL), max_features=1500)
        dtm = vect.fit_transform(dff["texto_limpio"])

        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)

        words = vect.get_feature_names_out()
        lda_df = pd.DataFrame([
            {"cluster": i, "words": ", ".join([words[j] for j in comp.argsort()[-10:]])}
            for i, comp in enumerate(lda.components_)
        ])

    fig_lda = px.bar(lda_df, x="cluster", y="words", title="Temas (LDA)")

    return (
        fig_ts,
        fig_sent,
        fig_map,
        fig_comp,
        fig_users,
        fig_rt,
        fig_likes,
        wc_base64,
        fig_lda
    )

# ============================================================
# CALLBACKS FACEBOOK
# ============================================================
@app.callback(
    [
        Output("fb-seguid","figure"),
        Output("fb-inter","figure"),
        Output("fb-vis","figure"),
        Output("fb-clics","figure"),
        Output("fb-visitas","figure"),
    ],
    Input("periodo-fb","value")
)
def update_facebook(periodo):

    figures = []

    for df_fb, title in [
        (fb_seguidores, "Seguidores"),
        (fb_interacciones, "Interacciones"),
        (fb_visualizaciones, "Visualizaciones"),
        (fb_clics, "Clics"),
        (fb_visitas, "Visitas")
    ]:

        if df_fb.empty:
            figures.append(go.Figure())
            continue

        col = "Primary"
        dff = df_fb.set_index("Fecha")[col].resample(periodo).sum().reset_index()
        fig = px.line(dff, x="Fecha", y=col, title=title)
        figures.append(fig)

    return figures

# ============================================================
# CALLBACKS INSTAGRAM
# ============================================================
@app.callback(
    [
        Output("ig-seguid","figure"),
        Output("ig-inter","figure"),
        Output("ig-vis","figure"),
        Output("ig-clics","figure"),
        Output("ig-visitas","figure"),
        Output("ig-alcance","figure"),
    ],
    Input("periodo-ig","value")
)
def update_instagram(periodo):

    figures = []

    for df_ig, title in [
        (ig_seguidores, "Seguidores"),
        (ig_interacciones, "Interacciones"),
        (ig_visualizaciones, "Visualizaciones"),
        (ig_clics, "Clics"),
        (ig_visitas, "Visitas"),
        (ig_alcance, "Alcance"),
    ]:

        if df_ig.empty:
            figures.append(go.Figure())
            continue

        col = "Primary"
        dff = df_ig.set_index("Fecha")[col].resample(periodo).sum().reset_index()
        fig = px.line(dff, x="Fecha", y=col, title=title)
        figures.append(fig)

    return figures

# ============================================================
# CALLBACKS COMPARACIÓN
# ============================================================
@app.callback(
    [
        Output("cmp-total","figure"),
        Output("cmp-tend","figure")
    ],
    Input("tabs","value")
)
def update_compare(tab):

    if tab != "tab-compare":
        return go.Figure(), go.Figure()

    if twitter.empty or fb_interacciones.empty or ig_interacciones.empty:
        return go.Figure(), go.Figure()

    tw_total = len(twitter)
    fb_total = fb_interacciones["Primary"].sum()
    ig_total = ig_interacciones["Primary"].sum()

    # --- Gráfico total comparativo ---
    fig_total = px.bar(
        x=["Twitter", "Facebook", "Instagram"],
        y=[tw_total, fb_total, ig_total],
        title="Comparación total de actividad"
    )

    # --- Tendencias semanales ---
    twitter_ts = twitter.set_index("fecha").resample("W").size().reset_index(name="Twitter")

    fb_ts = fb_interacciones.set_index("Fecha")["Primary"].resample("W").sum().reset_index()
    fb_ts.rename(columns={"Primary": "Facebook"}, inplace=True)

    ig_ts = ig_interacciones.set_index("Fecha")["Primary"].resample("W").sum().reset_index()
    ig_ts.rename(columns={"Primary": "Instagram"}, inplace=True)

    fig_tend = go.Figure()
    fig_tend.add_trace(go.Scatter(x=twitter_ts["fecha"], y=twitter_ts["Twitter"],
                                  mode="lines", name="Twitter"))
    fig_tend.add_trace(go.Scatter(x=fb_ts["Fecha"], y=fb_ts["Facebook"],
                                  mode="lines", name="Facebook"))
    fig_tend.add_trace(go.Scatter(x=ig_ts["Fecha"], y=ig_ts["Instagram"],
                                  mode="lines", name="Instagram"))

    fig_tend.update_layout(title="Tendencias semanales por plataforma")

    return fig_total, fig_tend

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)

