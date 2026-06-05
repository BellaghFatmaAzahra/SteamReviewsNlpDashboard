import pandas as pd
import re
import os

os.makedirs("outputs", exist_ok=True)

print("Lecture d'un echantillon de 50000 reviews...")
df = pd.read_csv("data/dataset.csv", nrows=50000)
print(f"Nombre de reviews chargees : {len(df)}")

df = df.rename(columns={"review_text": "review", "review_score": "score"})

df["sentiment_label"] = df["score"].apply(
    lambda x: "POSITIVE" if x == 1 else "NEGATIVE"
)

print(f"\nDistribution des sentiments :")
print(df["sentiment_label"].value_counts())


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("\nNettoyage des reviews...")
df["review_clean"] = df["review"].apply(clean_text)

initial_count = len(df)
df = df[df["review_clean"].str.len() > 10]
print(f"Reviews supprimees (trop courtes) : {initial_count - len(df)}")

df = df.drop_duplicates(subset=["review_clean"])
print(f"Nombre final de reviews apres nettoyage : {len(df)}")

df.to_csv("outputs/steam_reviews_cleaned.csv", index=False)
print("\nFichier sauvegarde : outputs/steam_reviews_cleaned.csv")

print(f"\nStatistiques finales :")
print(f"- Jeux dans le dataset : {df['app_name'].nunique()}")
print(
    f"- Reviews positives : {df['sentiment_label'].value_counts().get('POSITIVE', 0)}"
)
print(
    f"- Reviews negatives : {df['sentiment_label'].value_counts().get('NEGATIVE', 0)}"
)
