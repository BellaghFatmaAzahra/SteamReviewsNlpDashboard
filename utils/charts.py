import plotly.express as px
import plotly.graph_objects as go


def create_sentiment_bar_chart(df):
    sentiment_counts = df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"},
        title="Distribution des Sentiments",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font_color="white",
    )
    return fig


def create_model_comparison_chart(df):
    fig = px.bar(
        df,
        x="model_name",
        y="f1_score",
        color="model_name",
        title="Comparaison des Modeles (F1 Score)",
        text="f1_score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font_color="white",
        xaxis_title="Modele",
        yaxis_title="F1 Score",
    )
    return fig


def create_confidence_histogram(df):
    fig = px.histogram(
        df,
        x="confidence",
        nbins=30,
        title="Distribution des Confiances des Predictions",
        color_discrete_sequence=["#3498db"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font_color="white",
        xaxis_title="Niveau de Confiance",
        yaxis_title="Nombre de Predictions",
    )
    return fig
