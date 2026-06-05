import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONNEXION POSTGRESQL SUPABASE
# ─────────────────────────────────────────
conn = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    database=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
)
cursor = conn.cursor()
print("Connecté !")

# ─────────────────────────────────────────
# CRÉATION DES TABLES
# ─────────────────────────────────────────
print("\nCréation des tables...")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_results (
        id              SERIAL PRIMARY KEY,
        app_name        VARCHAR(255),
        review_clean    TEXT,
        sentiment_label VARCHAR(20),
        predicted_label VARCHAR(20),
        score           INTEGER,
        inserted_at     TIMESTAMP DEFAULT NOW()
    );
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        id          SERIAL PRIMARY KEY,
        model_name  VARCHAR(100),
        accuracy    FLOAT,
        f1_score    FLOAT,
        precision   FLOAT,
        recall      FLOAT,
        roc_auc     FLOAT,
        created_at  TIMESTAMP DEFAULT NOW()
    );
""")

conn.commit()
print("Tables créées : sentiment_results, model_metrics")

# ─────────────────────────────────────────
# INSERTION DES PRÉDICTIONS
# ─────────────────────────────────────────
print("\nChargement des prédictions...")
df = pd.read_csv("outputs/steam_reviews_with_sentiment.csv")
sample = df.sample(min(3000, len(df)), random_state=42).reset_index(drop=True)
print(f"Echantillon : {len(sample)} reviews")

# Vider la table
cursor.execute("TRUNCATE TABLE sentiment_results RESTART IDENTITY;")

rows = [
    (
        str(row.get("app_name", ""))[:255],
        str(row.get("review_clean", ""))[:2000],
        str(row.get("sentiment_label", "")),
        str(row.get("predicted_label", "")),
        int(row.get("score", 0)),
    )
    for _, row in sample.iterrows()
]

execute_values(cursor, """
    INSERT INTO sentiment_results (app_name, review_clean, sentiment_label, predicted_label, score)
    VALUES %s
""", rows)

conn.commit()
print(f"Lignes insérées dans sentiment_results : {len(rows)}")

# ─────────────────────────────────────────
# INSERTION DES MÉTRIQUES
# ─────────────────────────────────────────
print("\nInsertion des métriques des modèles...")
metrics_df = pd.read_csv("outputs/metrics/models_comparison.csv", index_col=0)

cursor.execute("TRUNCATE TABLE model_metrics RESTART IDENTITY;")

for model_name, row in metrics_df.iterrows():
    cursor.execute("""
        INSERT INTO model_metrics (model_name, accuracy, f1_score, precision, recall, roc_auc)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        model_name,
        float(row["accuracy"]),
        float(row["f1_score"]),
        float(row["precision"]),
        float(row["recall"]),
        float(row["roc_auc"]),
    ))

conn.commit()
print(f"Métriques insérées : {len(metrics_df)} modèles")

# ─────────────────────────────────────────
# VÉRIFICATION
# ─────────────────────────────────────────
print("\nVérification :")
cursor.execute("SELECT COUNT(*) FROM sentiment_results;")
print(f"  sentiment_results : {cursor.fetchone()[0]} lignes")

cursor.execute("SELECT COUNT(*) FROM model_metrics;")
print(f"  model_metrics     : {cursor.fetchone()[0]} lignes")

cursor.execute("SELECT model_name, f1_score, roc_auc FROM model_metrics ORDER BY f1_score DESC;")
print("\nMétriques dans PostgreSQL :")
for row in cursor.fetchall():
    print(f"  {row[0]:<30} F1={row[1]:.4f} ROC-AUC={row[2]:.4f}")

cursor.close()
conn.close()
print("\nDone — données disponibles sur Supabase !")