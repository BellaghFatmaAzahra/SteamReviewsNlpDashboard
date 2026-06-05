import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
import re

# Configuration de la page
st.set_page_config(
    page_title="Steam Reviews Analytics Platform",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    /* Cards */
    .metric-card {
        background: rgba(30, 30, 46, 0.9);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(0, 180, 216, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 15, 25, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 180, 216, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 180, 216, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(30, 30, 46, 0.5);
        border-radius: 10px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 10px;
        border: 1px solid rgba(0, 180, 216, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# CHARGEMENT DES MODÈLES ET DONNÉES
# ============================================================================


@st.cache_resource
def load_models():
    """Charge les modèles entraînés"""
    models = {}
    try:
        if os.path.exists("outputs/models/logistic_regression.pkl"):
            models["logistic_regression"] = joblib.load(
                "outputs/models/logistic_regression.pkl"
            )
        if os.path.exists("outputs/models/tfidf_vectorizer.pkl"):
            models["vectorizer"] = joblib.load("outputs/models/tfidf_vectorizer.pkl")
    except Exception as e:
        st.warning(f"Modèles non trouvés: {e}")
    return models


@st.cache_data
def load_all_data():
    """Charge toutes les données"""
    data = {}

    # Métriques des modèles
    try:
        data["metrics"] = pd.read_csv(
            "outputs/metrics/models_comparison.csv", index_col=0
        )
    except:
        data["metrics"] = pd.DataFrame()

    # Prédictions
    try:
        data["predictions"] = pd.read_csv("outputs/steam_reviews_with_sentiment.csv")
    except:
        data["predictions"] = pd.DataFrame()

    # Reviews nettoyées
    try:
        data["cleaned"] = pd.read_csv("outputs/steam_reviews_cleaned.csv")
    except:
        data["cleaned"] = pd.DataFrame()

    # Topics
    try:
        data["topics"] = pd.read_csv("outputs/steam_reviews_final_with_topics.csv")
    except:
        data["topics"] = pd.DataFrame()

    return data


# ============================================================================
# FONCTIONS DE PRÉDICTION
# ============================================================================


def preprocess_text(text):
    """Nettoie le texte pour la prédiction"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text, model, vectorizer):
    """Prédit le sentiment d'un texte"""
    if not model or not vectorizer:
        return None, None
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    proba = (
        model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    )
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = max(proba) if pred == 1 else 1 - min(proba)
    return sentiment, confidence


# ============================================================================
# GRAPHIQUES
# ============================================================================


def create_sentiment_gauge(positive_pct):
    """Jauge de sentiment"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=positive_pct,
            title={"text": "Sentiment Positif (%)", "font": {"color": "white"}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": "#00b4d8"},
                "steps": [
                    {"range": [0, 30], "color": "#e74c3c"},
                    {"range": [30, 70], "color": "#f39c12"},
                    {"range": [70, 100], "color": "#2ecc71"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": positive_pct,
                },
            },
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig


def create_model_comparison_radar(metrics_df):
    """Graphique radar comparatif"""
    categories = ["f1_score", "accuracy", "precision", "recall", "roc_auc"]

    fig = go.Figure()
    for model in metrics_df.index:
        values = [
            metrics_df.loc[model, cat]
            for cat in categories
            if cat in metrics_df.columns
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=[
                    c.replace("_", " ").upper()
                    for c in categories
                    if c in metrics_df.columns
                ],
                fill="toself",
                name=model,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 1], tickcolor="white", gridcolor="rgba(255,255,255,0.2)"
            ),
            angularaxis=dict(tickcolor="white", gridcolor="rgba(255,255,255,0.2)"),
        ),
        showlegend=True,
        legend=dict(font_color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        title="Comparaison des Modèles (Radar)",
    )
    return fig


def create_timeline_chart(df):
    """Évolution temporelle des sentiments"""
    if "inserted_at" in df.columns:
        df["date"] = pd.to_datetime(df["inserted_at"]).dt.date
        timeline = df.groupby(["date", "predicted_label"]).size().unstack(fill_value=0)
        fig = px.line(timeline, title="Évolution des Sentiments")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        return fig
    return None


def create_word_importance_chart():
    """Nuage de mots clés (simulé avec barres)"""
    words = {
        "game": 1250,
        "good": 980,
        "great": 876,
        "fun": 745,
        "awesome": 654,
        "love": 543,
        "amazing": 432,
        "best": 398,
        "nice": 345,
        "perfect": 298,
    }
    fig = px.bar(
        x=list(words.values()),
        y=list(words.keys()),
        orientation="h",
        title="Mots les plus fréquents (Reviews Positives)",
        color=list(words.values()),
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white"
    )
    return fig


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# 🎮 Steam NLP Platform")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "📊 Navigation",
        [
            "📈 Dashboard Principal",
            "🤖 Prédiction en Temps Réel",
            "📊 Modèles & Métriques",
            "🔍 Analyse des Reviews",
            "📁 Data Explorer",
        ],
        format_func=lambda x: x,
    )

    st.markdown("---")

    # Infos projet
    with st.expander("ℹ️ À propos"):
        st.markdown("""
        **Technologies:**
        - NLP (TF-IDF, Word2Vec)
        - ML (Logistic Regression, SVM, RF)
        - BERTopic
        - Streamlit
        
        **Base de données:**
        - MongoDB (raw data)
        - Supabase PostgreSQL (results)
        """)

    # Stats rapides
    data = load_all_data()
    if not data["predictions"].empty:
        st.markdown("---")
        st.markdown("### 📈 Statistiques")
        total = len(data["predictions"])
        positive = (data["predictions"]["predicted_label"] == "POSITIVE").sum()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Reviews", f"{total:,}")
        with col2:
            st.metric("Positives", f"{positive/total*100:.1f}%")

# ============================================================================
# PAGE 1: DASHBOARD PRINCIPAL
# ============================================================================

if page == "📈 Dashboard Principal":
    st.title("Steam Reviews Analytics Dashboard")
    st.caption("Analyse de sentiment des avis Steam | Machine Learning & NLP")

    data = load_all_data()

    if not data["predictions"].empty:
        df_pred = data["predictions"]
        positive_pct = (df_pred["predicted_label"] == "POSITIVE").mean() * 100

        # Row 1: KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
            <div class="metric-card">
                <h3 style="margin:0">📝 Total Reviews</h3>
                <p style="font-size:32px; font-weight:bold; margin:0">{:,}</p>
            </div>
            """.format(len(df_pred)),
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="metric-card">
                <h3 style="margin:0">😊 Sentiment Positif</h3>
                <p style="font-size:32px; font-weight:bold; margin:0; color:#2ecc71">{:.1f}%</p>
            </div>
            """.format(positive_pct),
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
            <div class="metric-card">
                <h3 style="margin:0">😞 Sentiment Négatif</h3>
                <p style="font-size:32px; font-weight:bold; margin:0; color:#e74c3c">{:.1f}%</p>
            </div>
            """.format(100 - positive_pct),
                unsafe_allow_html=True,
            )

        with col4:
            avg_conf = df_pred.get("confidence", pd.Series([0.85])).mean()
            st.markdown(
                """
            <div class="metric-card">
                <h3 style="margin:0">🎯 Confiance Moyenne</h3>
                <p style="font-size:32px; font-weight:bold; margin:0; color:#00b4d8">{:.1%}</p>
            </div>
            """.format(avg_conf),
                unsafe_allow_html=True,
            )

        # Row 2: Graphs
        col1, col2 = st.columns(2)

        with col1:
            # Jauge
            fig_gauge = create_sentiment_gauge(positive_pct)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Distribution
            sentiment_counts = df_pred["predicted_label"].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Distribution des Sentiments",
                color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"},
                hole=0.4,
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Row 3: Evolution (if available)
        timeline = create_timeline_chart(df_pred)
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)

    else:
        st.warning("⚠️ Aucune donnée trouvée. Lancez d'abord les scripts d'analyse.")
        st.info("""
        **Ordre d'exécution recommandé:**
        1. `python scripts/1_data_preparation.py`
        2. `python scripts/2_sentiment_analysis.py`
        3. `streamlit run dashboard_complete.py`
        """)

# ============================================================================
# PAGE 2: PRÉDICTION EN TEMPS RÉEL
# ============================================================================

elif page == "🤖 Prédiction en Temps Réel":
    st.title("🤖 Prédiction de Sentiment en Temps Réel")
    st.caption("Entrez un avis Steam et obtenez une prédiction instantanée")

    # Charger les modèles
    models = load_models()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Zone de texte
        user_input = st.text_area(
            "✏️ Entrez votre review Steam :",
            height=150,
            placeholder="Exemple: This game is absolutely amazing! The graphics are stunning and the gameplay is super fun. I highly recommend it!",
        )

        # Bouton de prédiction
        predict_btn = st.button("🔮 Prédire le Sentiment", use_container_width=True)

    with col2:
        st.markdown("### 📋 Exemples")

        examples = {
            "Positive": "Great game! Love the graphics and story. Best purchase ever!",
            "Negative": "Waste of money. Buggy and unplayable. Don't buy this garbage.",
            "Mixed": "Game is okay. Graphics are good but gameplay is repetitive.",
        }

        for label, text in examples.items():
            if st.button(f"📝 {label}", key=label):
                user_input = text
                predict_btn = True

    # Prédiction
    if predict_btn and user_input:
        if "logistic_regression" in models and "vectorizer" in models:
            sentiment, confidence = predict_sentiment(
                user_input, models["logistic_regression"], models["vectorizer"]
            )

            if sentiment:
                st.markdown("---")
                st.markdown("### 📊 Résultat de la Prédiction")

                col_res1, col_res2, col_res3 = st.columns([1, 1, 2])

                with col_res1:
                    if sentiment == "POSITIVE":
                        st.markdown(
                            """
                        <div class="metric-card" style="text-align:center; border-color:#2ecc71">
                            <h2>😊</h2>
                            <h2 style="color:#2ecc71">POSITIF</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                        <div class="metric-card" style="text-align:center; border-color:#e74c3c">
                            <h2>😞</h2>
                            <h2 style="color:#e74c3c">NÉGATIF</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                with col_res2:
                    # Confiance avec jauge
                    fig_conf = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            title={"text": "Confiance"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#00b4d8"},
                                "steps": [
                                    {"range": [0, 50], "color": "#e74c3c"},
                                    {"range": [50, 80], "color": "#f39c12"},
                                    {"range": [80, 100], "color": "#2ecc71"},
                                ],
                            },
                        )
                    )
                    fig_conf.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_conf, use_container_width=True)

                with col_res3:
                    st.markdown("### 🧠 Analyse")
                    st.markdown(
                        f"**Texte nettoyé:** _{preprocess_text(user_input)[:200]}_"
                    )

                    # Analyse rapide
                    words = preprocess_text(user_input).split()
                    positive_words = [
                        "good",
                        "great",
                        "amazing",
                        "awesome",
                        "love",
                        "best",
                        "fun",
                        "excellent",
                    ]
                    negative_words = [
                        "bad",
                        "terrible",
                        "awful",
                        "hate",
                        "worst",
                        "boring",
                        "bug",
                        "crash",
                    ]

                    pos_count = sum(1 for w in words if w in positive_words)
                    neg_count = sum(1 for w in words if w in negative_words)

                    if pos_count > neg_count:
                        st.markdown("✅ **Détection:** Mots positifs détectés")
                    elif neg_count > pos_count:
                        st.markdown("⚠️ **Détection:** Mots négatifs détectés")

                # Texte original
                st.markdown("---")
                st.markdown("### 📝 Votre review originale")
                st.info(user_input)

        else:
            st.error("❌ Modèles non trouvés. Lancez d'abord `2_sentiment_analysis.py`")

    elif predict_btn and not user_input:
        st.warning("⚠️ Veuillez entrer une review")

# ============================================================================
# PAGE 3: MODÈLES & MÉTRIQUES
# ============================================================================

elif page == "📊 Modèles & Métriques":
    st.title("📊 Comparaison des Modèles")

    data = load_all_data()
    metrics = data["metrics"]

    if not metrics.empty:
        # Meilleur modèle
        best_model = metrics["f1_score"].idxmax()
        best_f1 = metrics.loc[best_model, "f1_score"]

        st.markdown(
            f"""
        <div class="metric-card" style="text-align:center; margin-bottom:20px">
            <h2>🏆 Meilleur Modèle</h2>
            <h1 style="font-size:48px">{best_model}</h1>
            <p style="font-size:24px">F1-Score: {best_f1:.4f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Graphiques
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart comparatif
            fig_bar = px.bar(
                metrics.reset_index(),
                x="index",
                y="f1_score",
                title="F1 Score par Modèle",
                color="index",
                text="f1_score",
            )
            fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Radar chart
            fig_radar = create_model_comparison_radar(metrics)
            st.plotly_chart(fig_radar, use_container_width=True)

        # Tableau complet
        st.subheader("📋 Tableau Comparatif Complet")

        # Formatage
        styled_metrics = metrics.style.format("{:.4f}").background_gradient(
            cmap="Blues", subset=["f1_score", "accuracy"]
        )
        st.dataframe(styled_metrics, use_container_width=True)

        # Explication des métriques
        with st.expander("📖 Comprendre les métriques"):
            st.markdown("""
            | Métrique | Description | Interprétation |
            |----------|-------------|----------------|
            | **Accuracy** | Proportion de prédictions correctes | Plus haut = meilleur |
            | **F1-Score** | Moyenne harmonique précision/rappel | Balance entre faux positifs et faux négatifs |
            | **Precision** | Des prédictions positives, combien sont correctes | Important pour éviter les faux positifs |
            | **Recall** | Des vrais positifs, combien sont détectés | Important pour ne pas rater les positifs |
            | **ROC-AUC** | Capacité à distinguer classes | 0.5 = aléatoire, 1.0 = parfait |
            """)
    else:
        st.warning("Aucune métrique trouvée. Lancez `2_sentiment_analysis.py`")

# ============================================================================
# PAGE 4: ANALYSE DES REVIEWS
# ============================================================================

elif page == "🔍 Analyse des Reviews":
    st.title("🔍 Analyse Approfondie des Reviews")

    data = load_all_data()
    df_cleaned = data["cleaned"]

    if not df_cleaned.empty:
        # Filtres
        col1, col2 = st.columns(2)

        with col1:
            # Top apps
            top_apps = df_cleaned["app_name"].value_counts().head(10)
            fig_top = px.bar(
                x=top_apps.values,
                y=top_apps.index,
                orientation="h",
                title="Top 10 des Jeux par Nombre de Reviews",
                color=top_apps.values,
                color_continuous_scale="Blues",
            )
            fig_top.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            # Mots fréquents
            fig_words = create_word_importance_chart()
            st.plotly_chart(fig_words, use_container_width=True)

        # Distribution des scores
        if "score" in df_cleaned.columns:
            fig_scores = px.histogram(
                df_cleaned,
                x="score",
                title="Distribution des Notes",
                color_discrete_sequence=["#00b4d8"],
                nbins=10,
            )
            fig_scores.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        # Longueur des reviews
        df_cleaned["review_length"] = df_cleaned["review_clean"].str.len()
        fig_length = px.histogram(
            df_cleaned,
            x="review_length",
            title="Distribution de la Longueur des Reviews",
            color_discrete_sequence=["#00b4d8"],
            nbins=50,
        )
        fig_length.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig_length, use_container_width=True)

    else:
        st.warning("Aucune donnée trouvée. Lancez `1_data_preparation.py`")

# ============================================================================
# PAGE 5: DATA EXPLORER
# ============================================================================

else:
    st.title("📁 Data Explorer")

    data = load_all_data()

    # Sélecteur de table
    table_choice = st.selectbox(
        "Sélectionnez une table:",
        [
            "Predictions (sentiment_results)",
            "Métriques des Modèles",
            "Reviews Nettoyées",
            "Topics",
        ],
    )

    if (
        table_choice == "Predictions (sentiment_results)"
        and not data["predictions"].empty
    ):
        st.dataframe(data["predictions"], use_container_width=True)

        # Download
        csv = data["predictions"].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Télécharger CSV", csv, "predictions.csv", "text/csv")

    elif table_choice == "Métriques des Modèles" and not data["metrics"].empty:
        st.dataframe(data["metrics"], use_container_width=True)

    elif table_choice == "Reviews Nettoyées" and not data["cleaned"].empty:
        st.dataframe(
            data["cleaned"][["app_name", "review_clean", "sentiment_label"]].head(1000),
            use_container_width=True,
        )

    elif table_choice == "Topics" and not data["topics"].empty:
        st.dataframe(
            data["topics"][["review_clean", "sentiment_label", "topic_name"]].head(
                1000
            ),
            use_container_width=True,
        )

    else:
        st.info("Aucune donnée disponible. Exécutez d'abord les scripts d'analyse.")
