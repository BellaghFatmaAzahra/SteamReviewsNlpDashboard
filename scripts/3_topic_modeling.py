import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import os

df = pd.read_csv("outputs/steam_reviews_with_sentiment.csv")
print(f"Chargement de {len(df)} reviews pour le topic modeling.")
print(f"Colonnes disponibles : {df.columns.tolist()}")

# Colonne sentiment — on prend ce qui existe
if "predicted_label" in df.columns:
    df["sentiment_roberta"] = df["predicted_label"]
elif "sentiment_label" in df.columns:
    df["sentiment_roberta"] = df["sentiment_label"]
else:
    df["sentiment_roberta"] = "POSITIVE"

if "confidence" not in df.columns:
    df["confidence"] = 1.0

print("Configuration de BERTopic...")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=5, metric="euclidean",
    cluster_selection_method="eom", prediction_data=True,
)

vectorizer_model = CountVectorizer(
    ngram_range=(1, 2), stop_words="english", max_features=1000
)

topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    verbose=True,
)

print("Entrainement du modele en cours...")
topics, probabilities = topic_model.fit_transform(df["review_clean"].tolist())

df["topic"] = topics
df["topic_probability"] = probabilities

topic_info = topic_model.get_topic_info()
print("\nTop 10 des sujets identifies :")
print(topic_info.head(10))

# Nommage des topics
topic_names = {}
for i in range(len(topic_info)):
    topic_id = topic_info.iloc[i]["Topic"]
    if topic_id != -1:
        words = topic_info.iloc[i]["Representation"][:3]
        topic_names[topic_id] = " | ".join(words)
    else:
        topic_names[topic_id] = "Outlier"

df["topic_name"] = df["topic"].apply(lambda x: topic_names.get(x, f"Topic_{x}"))

print("\nSentiment par topic :")
sentiment_by_topic = (
    df.groupby("topic_name")["sentiment_roberta"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0) * 100
)
print(sentiment_by_topic.head(10))

df.to_csv("outputs/steam_reviews_final_with_topics.csv", index=False)
print("\nFichier sauvegarde : outputs/steam_reviews_final_with_topics.csv")

topic_model.save("outputs/bertopic_model")
print("Modele BERTopic sauvegarde dans outputs/bertopic_model")
print("\nDone.")