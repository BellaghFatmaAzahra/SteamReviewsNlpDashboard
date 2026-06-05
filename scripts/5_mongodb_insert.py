import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os

# ─────────────────────────────────────────
# CONNEXION MONGODB ATLAS
# ─────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://ftmaazahra_db_user:nseH2lZAbDP5eVRP@cluster0.iqhjijw.mongodb.net/?appName=Cluster0")

print("Connexion à MongoDB Atlas...")
client = MongoClient(MONGO_URI)
db = client["steam_nlp"]
collection = db["raw_reviews"]

print(f"Connecté à la base : {db.name}")

# ─────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
print("\nChargement des reviews brutes...")
df = pd.read_csv("outputs/steam_reviews_cleaned.csv")
print(f"Nombre de reviews : {len(df)}")

# On prend un échantillon de 5000 pour ne pas dépasser le free tier (512MB)
sample = df.sample(min(5000, len(df)), random_state=42).reset_index(drop=True)
print(f"Echantillon : {len(sample)} reviews")

# ─────────────────────────────────────────
# INSERTION DANS MONGODB
# ─────────────────────────────────────────
print("\nInsertion dans MongoDB...")

# Vider la collection si elle existe déjà
collection.drop()

# Préparer les documents
documents = []
for _, row in sample.iterrows():
    doc = {
        "app_id":          str(row.get("app_id", "")),
        "app_name":        str(row.get("app_name", "")),
        "review":          str(row.get("review", "")),
        "review_clean":    str(row.get("review_clean", "")),
        "sentiment_label": str(row.get("sentiment_label", "")),
        "score":           int(row.get("score", 0)),
        "inserted_at":     datetime.utcnow(),
    }
    documents.append(doc)

# Insertion en batch
result = collection.insert_many(documents)
print(f"Documents insérés : {len(result.inserted_ids)}")

# ─────────────────────────────────────────
# VÉRIFICATION
# ─────────────────────────────────────────
total = collection.count_documents({})
positifs = collection.count_documents({"sentiment_label": "POSITIVE"})
negatifs = collection.count_documents({"sentiment_label": "NEGATIVE"})

print(f"\nVérification dans MongoDB :")
print(f"  Total documents  : {total}")
print(f"  POSITIVE         : {positifs}")
print(f"  NEGATIVE         : {negatifs}")

# Exemple de document
print(f"\nExemple de document :")
exemple = collection.find_one({})
for k, v in exemple.items():
    if k != "_id":
        print(f"  {k:<20} : {str(v)[:60]}")

# Index sur app_name pour accélérer les recherches
collection.create_index("app_name")
collection.create_index("sentiment_label")
print(f"\nIndex créés sur app_name et sentiment_label")

print("\nDone — données disponibles sur MongoDB Atlas !")
client.close()