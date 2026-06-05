import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# ─────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
print("Chargement des reviews nettoyées...")
df = pd.read_csv("outputs/steam_reviews_cleaned.csv")
print(f"Nombre de reviews : {len(df)}")

# Encodage des labels : POSITIVE = 1 / NEGATIVE = 0
df["label"] = df["sentiment_label"].apply(lambda x: 1 if x == "POSITIVE" else 0)

X = df["review_clean"]
y = df["label"]

# ─────────────────────────────────────────
# 2. SPLIT TRAIN / TEST
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {len(X_train)} | Test : {len(X_test)}")

# ─────────────────────────────────────────
# 3. VECTORISATION TF-IDF
# ─────────────────────────────────────────
print("\nVectorisation TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),  # unigrams + bigrams
    sublinear_tf=True,  # log normalization
    min_df=3,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Vocabulaire TF-IDF : {len(vectorizer.vocabulary_)} termes")


# ─────────────────────────────────────────
# 4. FONCTION D'ÉVALUATION
# ─────────────────────────────────────────
def evaluate_model(model, X_test_vec, y_test, model_name):
    y_pred = model.predict(X_test_vec)

    # ROC-AUC — nécessite des probabilités ou scores
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_vec)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test_vec)
    else:
        y_proba = y_pred

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    for k, v in metrics.items():
        print(f"  {k:<12} : {v}")
    print(
        f"\n{classification_report(y_test, y_pred, target_names=['NEGATIVE','POSITIVE'])}"
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NEGATIVE", "POSITIVE"],
        yticklabels=["NEGATIVE", "POSITIVE"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = f"outputs/metrics/confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix sauvegardée : {path}")

    return metrics, y_pred


# ─────────────────────────────────────────
# 5. MODÈLES + MLFLOW TRACKING
# ─────────────────────────────────────────
mlflow.set_experiment("steam_sentiment_analysis")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ),
    "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
}

results = {}

for model_name, model in models.items():
    print(f"\nEntraînement : {model_name}...")

    with mlflow.start_run(run_name=model_name):

        # Entraînement
        model.fit(X_train_vec, y_train)

        # Évaluation
        metrics, y_pred = evaluate_model(model, X_test_vec, y_test, model_name)
        results[model_name] = metrics

        # Log MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            f"outputs/metrics/confusion_matrix_{model_name.replace(' ', '_')}.png"
        )
        mlflow.sklearn.log_model(model, model_name.replace(" ", "_"))

# ─────────────────────────────────────────
# 6. COMPARAISON FINALE
# ─────────────────────────────────────────
print("\n\nCOMPARAISON DES MODÈLES")
print("=" * 55)
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values("f1_score", ascending=False)
print(results_df.to_string())
results_df.to_csv("outputs/metrics/models_comparison.csv")
print("\nComparaison sauvegardée : outputs/metrics/models_comparison.csv")

# Graphique comparaison
fig, ax = plt.subplots(figsize=(9, 5))
results_df.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white")
ax.set_title("Comparaison des modèles — Steam Sentiment Analysis")
ax.set_xlabel("Modèle")
ax.set_ylabel("Score")
ax.set_ylim(0.5, 1.0)
ax.legend(loc="lower right")
ax.set_xticklabels(results_df.index, rotation=15)
plt.tight_layout()
plt.savefig("outputs/metrics/models_comparison.png")
plt.close()
print("Graphique sauvegardé : outputs/metrics/models_comparison.png")

# Meilleur modèle
best_model_name = results_df["f1_score"].idxmax()
print(f"\nMeilleur modèle : {best_model_name}")
print(f"F1-Score        : {results_df.loc[best_model_name, 'f1_score']}")
print(f"ROC-AUC         : {results_df.loc[best_model_name, 'roc_auc']}")

# Sauvegarde des prédictions du meilleur modèle
best_model = models[best_model_name]
df_test = df.iloc[y_test.index].copy()
df_test["predicted_sentiment"] = best_model.predict(X_test_vec)
df_test["predicted_label"] = df_test["predicted_sentiment"].apply(
    lambda x: "POSITIVE" if x == 1 else "NEGATIVE"
)
df_test.to_csv("outputs/steam_reviews_with_sentiment.csv", index=False)
print(f"\nPrédictions sauvegardées : outputs/steam_reviews_with_sentiment.csv")


os.makedirs("outputs/models", exist_ok=True)

# Sauvegarde du vectorizer TF-IDF
joblib.dump(vectorizer, "outputs/models/tfidf_vectorizer.pkl")
print("Vectorizer sauvegarde : outputs/models/tfidf_vectorizer.pkl")

# Sauvegarde des modeles
for model_name, model in models.items():
    filename = model_name.lower().replace(" ", "_").replace("+", "").replace("  ", "_")
    path = f"outputs/models/{filename}.pkl"
    joblib.dump(model, path)
    print(f"Modele sauvegarde : {path}")
print("\nDone.")
