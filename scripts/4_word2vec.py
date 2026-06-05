import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from sklearn.manifold import TSNE

os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────
print("Chargement des reviews...")
df = pd.read_csv("outputs/steam_reviews_cleaned.csv")
print(f"Nombre de reviews : {len(df)}")
df["label"] = df["sentiment_label"].apply(lambda x: 1 if x == "POSITIVE" else 0)

# ─────────────────────────────────────────
# 2. TOKENISATION
# ─────────────────────────────────────────
print("\nTokenisation...")
df["tokens"] = df["review_clean"].apply(lambda x: str(x).lower().split())

# ─────────────────────────────────────────
# 3. WORD2VEC MAISON (Skip-Gram simplifié)
# ─────────────────────────────────────────
print("\nConstruction du vocabulaire...")

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 5

# Comptage des mots
word_counts = defaultdict(int)
for tokens in df["tokens"]:
    for word in tokens:
        word_counts[word] += 1

vocab = {word for word, count in word_counts.items() if count >= MIN_COUNT}
word2idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Taille du vocabulaire : {vocab_size}")

# Initialisation aléatoire des vecteurs (comme Word2Vec)
np.random.seed(42)
W = np.random.randn(vocab_size, VECTOR_SIZE) * 0.01  # vecteurs entrée
C = np.random.randn(vocab_size, VECTOR_SIZE) * 0.01  # vecteurs contexte

print(f"\nEntraînement Word2Vec Skip-Gram ({EPOCHS} epochs)...")
lr = 0.01

for epoch in range(EPOCHS):
    total_loss = 0
    count = 0
    for tokens in df["tokens"]:
        in_vocab = [t for t in tokens if t in word2idx]
        for i, center in enumerate(in_vocab):
            center_idx = word2idx[center]
            start = max(0, i - WINDOW)
            end = min(len(in_vocab), i + WINDOW + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context_idx = word2idx[in_vocab[j]]
                # Produit scalaire
                score = np.dot(W[center_idx], C[context_idx])
                prob = 1 / (1 + np.exp(-score))
                error = prob - 1
                # Mise à jour
                W[center_idx] -= lr * error * C[context_idx]
                C[context_idx] -= lr * error * W[center_idx]
                total_loss += -np.log(prob + 1e-9)
                count += 1

    print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/max(count,1):.4f}")

# Vecteurs finaux = moyenne W et C
embeddings = (W + C) / 2
print("\nEntraînement Word2Vec terminé.")

# Sauvegarde des embeddings
np.save("outputs/models/word2vec_embeddings.npy", embeddings)
with open("outputs/models/word2vec_vocab.txt", "w", encoding="utf-8") as f:
    for word in word2idx:
        f.write(f"{word}\n")
print("Embeddings sauvegardés.")

# Mots similaires
def most_similar(word, topn=5):
    if word not in word2idx:
        return []
    idx = word2idx[word]
    vec = embeddings[idx]
    norms = np.linalg.norm(embeddings, axis=1) + 1e-9
    sims = embeddings @ vec / (norms * np.linalg.norm(vec))
    top_idx = np.argsort(sims)[::-1][1:topn+1]
    idx2word = {v: k for k, v in word2idx.items()}
    return [(idx2word[i], round(float(sims[i]), 4)) for i in top_idx]

print("\nMots proches de 'good' :")
for word, score in most_similar("good"):
    print(f"  {word:<15} {score:.4f}")

print("\nMots proches de 'bad' :")
for word, score in most_similar("bad"):
    print(f"  {word:<15} {score:.4f}")

# ─────────────────────────────────────────
# 4. VECTORISATION DES PHRASES
# ─────────────────────────────────────────
print("\nVectorisation des phrases...")
def get_sentence_vector(tokens):
    vecs = [embeddings[word2idx[w]] for w in tokens if w in word2idx]
    return np.mean(vecs, axis=0) if vecs else np.zeros(VECTOR_SIZE)

X = np.array([get_sentence_vector(t) for t in df["tokens"]])
y = df["label"].values
print(f"Shape des features : {X.shape}")

# ─────────────────────────────────────────
# 5. SPLIT + MODÈLES
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    for k, v in metrics.items():
        print(f"  {k:<12} : {v}")
    print(classification_report(y_test, y_pred, target_names=["NEGATIVE", "POSITIVE"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NEGATIVE", "POSITIVE"],
                yticklabels=["NEGATIVE", "POSITIVE"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/metrics/confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.close()
    return metrics

mlflow.set_experiment("steam_sentiment_analysis")

models = {
    "W2V + Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "W2V + Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}
for model_name, model in models.items():
    print(f"\nEntraînement : {model_name}...")
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = metrics
        mlflow.log_params({"vectorizer": "Word2Vec_custom", "vector_size": VECTOR_SIZE, "window": WINDOW, "epochs": EPOCHS})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name.replace(" ", "_"))

# ─────────────────────────────────────────
# 6. COMPARAISON TFIDF vs WORD2VEC
# ─────────────────────────────────────────
print("\n\nCOMPARAISON TF-IDF vs Word2Vec")
print("="*55)
tfidf_results = pd.read_csv("outputs/metrics/models_comparison.csv", index_col=0)
w2v_df = pd.DataFrame(results).T
all_results = pd.concat([tfidf_results, w2v_df]).sort_values("f1_score", ascending=False)
print(all_results.to_string())
all_results.to_csv("outputs/metrics/all_models_comparison.csv")

fig, ax = plt.subplots(figsize=(12, 5))
all_results.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white")
ax.set_title("TF-IDF vs Word2Vec — Comparaison des modèles")
ax.set_ylim(0.5, 1.0)
ax.legend(loc="lower right")
ax.set_xticklabels(all_results.index, rotation=20, ha="right")
plt.tight_layout()
plt.savefig("outputs/metrics/all_models_comparison.png")
plt.close()
print("Graphique sauvegardé.")

# ─────────────────────────────────────────
# 7. VISUALISATION t-SNE
# ─────────────────────────────────────────
print("\nVisualisation t-SNE...")
sample_size = min(2000, len(X))
idx = np.random.choice(len(X), sample_size, replace=False)
X_2d = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500).fit_transform(X[idx])

fig, ax = plt.subplots(figsize=(10, 7))
for label, color, name in [(0, "#ff4444", "NEGATIVE"), (1, "#00cc66", "POSITIVE")]:
    mask = y[idx] == label
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name, alpha=0.5, s=10)
ax.set_title("t-SNE — Embeddings Word2Vec (2000 avis Steam)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/metrics/tsne_word2vec.png")
plt.close()
print("t-SNE sauvegardé.")

print("\nDone.")