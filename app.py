# IMPORTS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html, Input, Output, State
from conseiller import simuler, repondre

# CHARGEMENT ET NETTOYAGE DES DONNÉES
df = pd.read_csv("insurance.csv")

# Renommer les colonnes en français
df = df.rename(columns={
    "age": "age",
    "sex": "genre",
    "bmi": "imc",
    "children": "enfants",
    "smoker": "fumeur",
    "region": "region",
    "charges": "charges"
})

# Supprimer le doublon
df = df.drop_duplicates()

# ENCODAGE DES COLONNES TEXTE
df['fumeur'] = df['fumeur'].map({'yes': 1, 'no': 0})
df['genre'] = df['genre'].map({'male': 1, 'female': 0})
df['region'] = df['region'].map({
    'northeast': 0,
    'northwest': 1,
    'southeast': 2,
    'southwest': 3
})

# ENTRAÎNEMENT DU MODÈLE RANDOM FOREST
X = df.drop(columns=['charges'])
Y = df['charges']

# Séparation 80% entraînement / 20% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
modele = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
modele.fit(X_train, Y_train)

# STYLES
POLICE = "Arial, Helvetica, sans-serif"
BLEU = "#118DFF"
STYLE_LABEL = {"fontSize": "15px", "fontWeight": "bold", "color": "#1A1A2E",
               "display": "block", "marginBottom": "6px"}
STYLE_CHAMP = {"display": "block", "padding": "12px", "width": "100%", "fontSize": "15px",
               "border": "1px solid #ccc", "borderRadius": "6px", "boxSizing": "border-box",
               "fontFamily": POLICE}
STYLE_LISTE = {"fontSize": "15px", "fontFamily": POLICE}


def champ(label, composant):
    return html.Div([html.Label(label, style=STYLE_LABEL), composant])


# APPLICATION DASH
app = dash.Dash(__name__, title="Prédiction de charges médicales")
server = app.server

app.layout = html.Div([html.Div([

    # Titre
    html.H1("Prédiction de charges médicales",
            style={"color": "#1A1A2E", "fontSize": "28px", "margin": "0 0 8px 0"}),

    # Sous-titre
    html.P("Entrez vos informations pour estimer votre charge annuelle d'assurance santé",
           style={"color": "#666677", "margin": "0 0 30px 0", "fontSize": "16px"}),

    # Champs sur 2 colonnes (1 colonne sur téléphone)
    html.Div([
        champ("Âge", dcc.Input(id="age", type="number", value=35, min=18, max=64, style=STYLE_CHAMP)),
        champ("Sexe", dcc.Dropdown(id="genre", options=[
            {"label": "Homme", "value": 1},
            {"label": "Femme", "value": 0}
        ], value=1, clearable=False, style=STYLE_LISTE)),
        champ("Indice de masse corporelle (IMC)",
              dcc.Input(id="imc", type="number", value=28, min=10, max=60, step=0.1, style=STYLE_CHAMP)),
        champ("Nombre d'enfants", dcc.Input(id="enfants", type="number", value=0, min=0, max=5, style=STYLE_CHAMP)),
        champ("Fumeur", dcc.Dropdown(id="fumeur", options=[
            {"label": "Non", "value": 0},
            {"label": "Oui", "value": 1}
        ], value=0, clearable=False, style=STYLE_LISTE)),
        champ("Région", dcc.Dropdown(id="region", options=[
            {"label": "Nord-Est", "value": 0},
            {"label": "Nord-Ouest", "value": 1},
            {"label": "Sud-Est", "value": 2},
            {"label": "Sud-Ouest", "value": 3}
        ], value=0, clearable=False, style=STYLE_LISTE)),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
              "gap": "20px 24px", "marginBottom": "30px"}),

    # Bouton
    html.Button("Estimer ma charge", id="bouton",
                style={"backgroundColor": BLEU, "color": "white", "padding": "14px 32px",
                       "border": "none", "borderRadius": "8px", "cursor": "pointer",
                       "fontSize": "16px", "fontWeight": "bold", "width": "100%", "fontFamily": POLICE}),

    # Résultat
    html.Div(id="resultat", style={"marginTop": "24px"}),

    # Conseiller (agent) : apparaît après l'estimation
    dcc.Store(id="profil"),
    dcc.Store(id="historique", data=[]),
    html.Div(id="zone_conseiller", hidden=True, children=[
        html.H2("Votre conseiller", style={"color": "#1A1A2E", "fontSize": "20px", "margin": "32px 0 4px 0"}),
        html.P("Des conseils pour payer moins cher et protéger votre santé, calculés à partir de votre profil.",
               style={"color": "#666677", "margin": "0 0 14px 0", "fontSize": "14px"}),
        dcc.Loading(html.Div(id="messages", style={"display": "flex", "flexDirection": "column", "gap": "10px"}),
                    type="dot", color=BLEU),
        html.Div([
            dcc.Input(id="question", type="text", placeholder="Posez votre question (ex. : comment arrêter de fumer ?)",
                      debounce=False, n_submit=0, style={**STYLE_CHAMP, "flex": "1"}),
            html.Button("Envoyer", id="envoyer", n_clicks=0,
                        style={"backgroundColor": BLEU, "color": "white", "padding": "12px 20px", "border": "none",
                               "borderRadius": "8px", "cursor": "pointer", "fontSize": "15px", "fontWeight": "bold",
                               "fontFamily": POLICE}),
        ], style={"display": "flex", "gap": "10px", "marginTop": "14px"}),
    ]),

], style={"maxWidth": "960px", "margin": "0 auto", "padding": "40px",
          "backgroundColor": "#FFFFFF", "borderRadius": "12px",
          "boxShadow": "0 2px 8px rgba(0,0,0,0.1)", "boxSizing": "border-box"})

], style={"backgroundColor": "#F3F2F1", "minHeight": "100vh", "padding": "40px 16px",
          "fontFamily": POLICE, "boxSizing": "border-box"})


# FONCTION DE PRÉDICTION
@app.callback(
    Output("resultat", "children"),
    Output("profil", "data"),
    Output("historique", "data", allow_duplicate=True),
    Output("zone_conseiller", "hidden"),
    Input("bouton", "n_clicks"),
    State("age", "value"),
    State("genre", "value"),
    State("imc", "value"),
    State("enfants", "value"),
    State("fumeur", "value"),
    State("region", "value"),
    prevent_initial_call=True
)
def predire(n_clicks, age, genre, imc, enfants, fumeur, region):
    valeurs = [age, genre, imc, enfants, fumeur, region]
    if any(v is None for v in valeurs):
        return (html.P("Merci de remplir tous les champs.",
                       style={"color": "#C0392B", "textAlign": "center", "fontSize": "16px"}),
                dash.no_update, dash.no_update, True)

    personne = pd.DataFrame([valeurs], columns=X.columns)
    charge = modele.predict(personne)[0]
    montant = f"{charge:,.0f}".replace(",", " ")

    # L'agent : simulations avec le modèle (son outil), puis premier conseil
    profil = {"age": age, "genre": genre, "imc": imc, "enfants": enfants, "fumeur": fumeur, "region": region}
    scenarios = simuler(modele, list(X.columns), profil)
    profil["scenarios"] = scenarios
    demande = {"role": "user", "content": "Quels conseils me donnez-vous pour payer moins cher et protéger ma santé ?", "cache": True}
    conseil = repondre([{"role": "user", "content": demande["content"]}], profil, scenarios)
    historique = [demande, {"role": "assistant", "content": conseil}]

    return html.Div([
        html.P("Charge estimée", style={"margin": "0", "color": "#666677", "fontSize": "15px"}),
        html.P(f"{montant} $ par an", style={"margin": "6px 0 0 0", "color": BLEU,
                                              "fontSize": "32px", "fontWeight": "bold"}),
    ], style={"backgroundColor": "#EAF4FF", "borderRadius": "10px", "padding": "20px",
              "textAlign": "center"}), profil, historique, False


# CONSEILLER : l'assuré pose une question, l'agent répond (la conversation est gardée en mémoire)
@app.callback(
    Output("historique", "data", allow_duplicate=True),
    Output("question", "value"),
    Input("envoyer", "n_clicks"),
    Input("question", "n_submit"),
    State("question", "value"),
    State("historique", "data"),
    State("profil", "data"),
    prevent_initial_call=True
)
def discuter(n_clicks, n_submit, question, historique, profil):
    if not question or not question.strip() or not profil:
        return dash.no_update, dash.no_update
    historique = (historique or []) + [{"role": "user", "content": question.strip()}]
    echange = [{"role": m["role"], "content": m["content"]} for m in historique]
    historique.append({"role": "assistant", "content": repondre(echange, profil, profil["scenarios"])})
    return historique, ""


# Affichage de la conversation (les messages de l'assuré à droite, ceux du conseiller à gauche)
@app.callback(Output("messages", "children"), Input("historique", "data"))
def afficher(historique):
    bulles = []
    for m in historique or []:
        if m.get("cache"):
            continue
        assure = m["role"] == "user"
        bulles.append(dcc.Markdown(m["content"], style={
            "alignSelf": "flex-end" if assure else "flex-start", "maxWidth": "85%",
            "backgroundColor": BLEU if assure else "#F3F2F1", "color": "white" if assure else "#1A1A2E",
            "borderRadius": "12px", "padding": "4px 14px", "fontSize": "15px", "lineHeight": "1.45"}))
    return bulles


# LANCER L'APPLICATION
if __name__ == "__main__":
    app.run(debug=True)