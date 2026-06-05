# 🎮 Steam Reviews Analytics

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28%2B-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/MLflow-2.5%2B-orange?style=for-the-badge&logo=mlflow" alt="MLflow">
  <img src="https://img.shields.io/badge/MongoDB-Atlas-green?style=for-the-badge&logo=mongodb" alt="MongoDB">
  <img src="https://img.shields.io/badge/PostgreSQL-Supabase-blue?style=for-the-badge&logo=postgresql" alt="PostgreSQL">
</p>

<p align="center">
  <a href="#">
    <img src="https://github.com/BellaghFatmaAzahra/SteamReviewsNlpDashboard/actions/workflows/ci.yml/badge.svg" alt="CI/CD">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Code%20Style-Black-black" alt="Code Style">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  </a>
</p>

> **Pipeline NLP complet d'analyse de 50 000 avis Steam — Sentiment Analysis + Topic Modeling + Dashboard interactif**

---

## 📋 Table des matières

- [Aperçu](#-aperçu)
- [Résultats clés](#-résultats-clés)
- [Architecture du pipeline](#-architecture-du-pipeline)
- [Modèles comparés](#-modèles-comparés)
- [Topic Modeling](#-topic-modeling-bertopic)
- [Base de données](#-base-de-données)
- [Structure du projet](#-structure-du-projet)
- [Quick Start](#-quick-start)
- [Dashboard](#-dashboard)
- [Stack technique](#-stack-technique)
- [Perspectives](#-perspectives-damélioration)
- [Auteur](#-auteur)

---

## 📊 Aperçu

Ce projet constitue un pipeline NLP complet d'analyse de **50 000 avis Steam**. L'objectif est d'extraire automatiquement des insights — sentiments, thèmes récurrents, tendances par jeu — en combinant :

- ✅ Approches classiques de machine learning (TF-IDF, Logistic Regression, SVM, Random Forest)
- ✅ Deep learning moderne (BERT fine-tuning)
- ✅ Topic modeling (BERTopic)
- ✅ Word2Vec implémenté from scratch en NumPy
- ✅ Double base de données (MongoDB + PostgreSQL)
- ✅ Dashboard interactif Streamlit

---

## 🎯 Résultats clés

<p align="center">
  <table>
    <tr>
      <td align="center"><strong>📝 Reviews analysées</strong><br><big><b>42 297</b></big></td>
      <td align="center"><strong>🏆 F1-Score (LR)</strong><br><big><b>0.9404</b></big></td>
      <td align="center"><strong>🤖 F1-Score (BERT)</strong><br><big><b>0.9429</b></big></td>
      <td align="center"><strong>📁 Docs MongoDB</strong><br><big><b>5 000</b></big></td>
      <td align="center"><strong>🗄️ Rows PostgreSQL</strong><br><big><b>3 000</b></big></td>
    </tr>
  </table>
</p>

| Métrique | Valeur |
|----------|--------|
| Reviews chargées | 50 000 |
| Après nettoyage | 42 297 |
| Sentiment positif | 35 169 (83%) |
| Sentiment négatif | 7 128 (17%) |
| Vocabulaire TF-IDF | 10 000 termes |
| Topics identifiés | 10 automatiquement |

---

## 🏗️ Architecture du pipeline
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Steam Reviews Analytics Pipeline │
├─────────────────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ 1. Data Prep │───▶│ 2. Sentiment │───▶│ 3. Topic │───▶│ 4. Word2Vec │ │
│ │ │ │ Analysis │ │ Modeling │ │ │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │ │ │ │ │
│ ▼ ▼ ▼ ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Nettoyage │ │ TF-IDF + LR │ │ BERTopic │ │ Skip-Gram │ │
│ │ Déduplication│ │ BERT fine- │ │ UMAP + │ │ NumPy from │ │
│ │ 42,297 │ │ tuned (0.94) │ │ HDBSCAN │ │ scratch │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │
│ ┌─────────────────────┐ │
│ │ 5. Double BDD │ │
│ │ MongoDB + Supabase │ │
│ └─────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────┐ │
│ │ 6. Dashboard │ │
│ │ Streamlit │ │
│ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

text

---

## 🤖 Modèles comparés

### TF-IDF + Modèles classiques

| Modèle | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|--------|----------|----------|-----------|--------|---------|
| **Logistic Regression** | 0.8978 | **0.9404** | 0.9128 | 0.9697 | 0.9376 |
| Linear SVM | 0.8969 | 0.9388 | 0.9265 | 0.9515 | 0.9281 |
| Random Forest | 0.8441 | 0.9141 | 0.8436 | 0.9974 | 0.8866 |

### BERT Fine-tuning (GPU T4 Colab)

| Époque | Loss | Train Accuracy |
|--------|------|----------------|
| 1/3 | 0.3450 | 84.00% |
| 2/3 | 0.1909 | 92.66% |
| 3/3 | 0.1203 | 95.99% |

| Métrique | Score |
|----------|-------|
| **F1-Score** | **0.9429** |
| ROC-AUC | 0.9321 |
| Accuracy | 0.9000 |
| Precision (NEG) | 0.72 |
| Recall (NEG) | 0.69 |

### Word2Vec (from scratch)

| Modèle | F1-Score | ROC-AUC |
|--------|----------|---------|
| W2V + Logistic Regression | 0.9080 | 0.6300 |
| W2V + Random Forest | 0.9076 | 0.6532 |

> **Conclusion** : TF-IDF surpasse Word2Vec sur ce dataset. Les avis Steam sont courts (2-5 phrases) et Word2Vec nécessite plus de contexte pour apprendre des représentations sémantiques pertinentes.

---

## 📊 Topic Modeling (BERTopic)

### Topics identifiés automatiquement

| Topic | Mots-clés | Count | Sentiment dominant |
|-------|-----------|-------|-------------------|
| 0 | game, play, shoot | 190 | 100% POSITIVE |
| 1 | duty, best duty, duty game | 154 | POSITIVE |
| **2** | **windows, work, crashes** | **149** | **NÉGATIVE (bugs)** |
| **3** | **hackers, hacked, lobby** | **120** | **NÉGATIVE** |
| 4 | counterstrike, terrorist, global | 120 | POSITIVE |
| 5 | best game, game best | 109 | 100% POSITIVE |
| 6 | tower, tower defense, defense | 108 | POSITIVE |
| 7 | zombie maps, custom zombie | 101 | 100% POSITIVE |
| 8 | cod, best cod, cod game | 101 | POSITIVE |
| -1 | Outlier (hors topics) | 3014 | Mixte |

### Insights clés

- 🔴 Les topics **2 (windows/crashes)** et **3 (hackers)** sont les seuls à dominante négative
- 💡 Les problèmes techniques et la triche sont les principales sources d'insatisfaction
- 🟢 Les topics Call of Duty, CS:GO et zombie maps sont quasi-unanimement positifs

---

## 🗄️ Base de données

### MongoDB Atlas — Données brutes

| Collection | Documents | POSITIVE | NEGATIVE | Index |
|------------|-----------|----------|----------|-------|
| `raw_reviews` | 5 000 | 4 169 (83.4%) | 831 (16.6%) | app_name, sentiment_label |

**Structure d'un document :**
```json
{
  "app_id": "104900",
  "app_name": "ORION: Prelude",
  "review": "this is the game of the year...",
  "sentiment_label": "POSITIVE",
  "score": 1,
  "inserted_at": "2026-06-04T10:10:34"
}
PostgreSQL Supabase — Résultats structurés
Table	Lignes	Colonnes principales
sentiment_results	3 000	app_name, review_clean, sentiment_label, predicted_label, score
model_metrics	3	model_name, accuracy, f1_score, precision, recall, roc_auc
Métriques stockées :

Modèle	F1-Score	ROC-AUC
Logistic Regression	0.9404	0.9376
Linear SVM	0.9388	0.9281
Random Forest	0.9141	0.8866
📂 Structure du projet
text
SteamReviewsNlpDashboard/
│
├── .github/workflows/          # CI/CD GitHub Actions
│   └── ci.yml                  # Tests automatiques à chaque push
│
├── scripts/
│   ├── 1_data_preparation.py   # Nettoyage et préparation des données
│   ├── 2_sentiment_analysis.py # TF-IDF + 3 modèles (MLflow tracking)
│   ├── 3_topic_modeling.py     # BERTopic (embedding + clustering)
│   ├── 4_word2vec.py           # Skip-Gram from scratch (NumPy)
│   ├── 5_mongodb_insert.py     # Insertion MongoDB Atlas
│   └── 6_postgresql_insert.py  # Insertion PostgreSQL Supabase
│
├── tests/                      # Tests unitaires pytest
│   ├── test_mongodb.py         # Connexion MongoDB
│   ├── test_postgresql.py      # Connexion PostgreSQL
│   └── test_sentiment.py       # Logique métier
│
├── utils/
│   ├── __init__.py
│   ├── db.py                   # Connexions BDD
│   └── charts.py               # Graphiques Plotly
│
├── dashboard.py                # Application Streamlit
├── requirements.txt            # Dépendances
├── .gitignore                  # Fichiers exclus
├── Dockerfile                  # Containerisation
└── README.md                   # Documentation
🚀 Quick Start
Prérequis
Python 3.10+

MongoDB Atlas (compte gratuit)

Supabase (compte gratuit)

Installation
bash
# 1. Cloner le projet
git clone https://github.com/BellaghFatmaAzahra/SteamReviewsNlpDashboard.git
cd SteamReviewsNlpDashboard

# 2. Créer l'environnement virtuel
python -m venv venv

# Activer (Windows)
venv\Scripts\activate
# Activer (Linux/Mac)
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
# Créer un fichier .env à la racine
Contenu de .env :

env
# MongoDB Atlas
MONGO_URI=mongodb+srv://username:password@cluster0.xxx.mongodb.net/

# PostgreSQL Supabase
PG_HOST=aws-1-eu-central-1.pooler.supabase.com
PG_PORT=5432
PG_DATABASE=postgres
PG_USER=postgres.xxx
PG_PASSWORD=votre_mot_de_passe
Exécution
bash
# Lancer l'analyse complète
python scripts/1_data_preparation.py
python scripts/2_sentiment_analysis.py
python scripts/3_topic_modeling.py
python scripts/4_word2vec.py
python scripts/5_mongodb_insert.py
python scripts/6_postgresql_insert.py

# Lancer le dashboard
streamlit run dashboard.py
Tests
bash
# Lancer les tests unitaires
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=. --cov-report=html
📊 Dashboard
Le dashboard expose deux onglets principaux :

Onglet	Fonctionnalités
Analytics	Distribution des sentiments, Top 10 topics, Heatmap sentiment×topic, Review Explorer, Métriques globales
Modèles & Métriques	Tableau comparatif, Graphiques bar/radar, 3 Confusion Matrices, Explication pédagogique
Accès
🌐 Local : http://localhost:8501

☁️ Déployé : steamreviewsnlpdashboard.onrender.com

🔧 Stack technique
Catégorie	Technologies
Langage	Python 3.14
ML Classique	Scikit-learn (TF-IDF, Logistic Regression, Random Forest, LinearSVC)
Deep Learning	PyTorch, Transformers (BERT fine-tuning)
Topic Modeling	BERTopic, SentenceTransformers, UMAP, HDBSCAN
Word Embeddings	Word2Vec Skip-Gram (NumPy from scratch)
NLP	NLTK, BERTopic, CountVectorizer
Visualisation	Streamlit, Plotly, Matplotlib, Seaborn
Base de données	MongoDB (pymongo), PostgreSQL (psycopg2)
Cloud BDD	MongoDB Atlas, Supabase
MLOps	MLflow 3.13 (experiment tracking, model logging)
DevOps	Git, GitHub Actions CI/CD, Docker, Render
Environnement	Google Colab (GPU T4)
🔮 Perspectives d'amélioration
Ajouter un pipeline Airflow pour automatiser le scraping quotidien

Implémenter Kafka pour le streaming de données en temps réel

Dockeriser l'application pour un déploiement reproductible

Ajouter SHAP/LIME pour l'explicabilité des modèles

Déployer BERT via FastAPI pour prédictions en temps réel

Intégrer PySpark pour le traitement de volumes plus importants

Ajouter plus de tests d'intégration

Mettre en place un monitoring des performances en production

👤 Auteur
Bellagh Fatma Azahra
M1 Intelligence Artificielle

🔗 GitHub

📧 [fatmaazahra.bellagh@gmail.com]

📄 License
MIT License - voir fichier LICENSE

<p align="center"> ⭐ N'hésitez pas à star le projet si ça vous a été utile ! ⭐ </p> ```

