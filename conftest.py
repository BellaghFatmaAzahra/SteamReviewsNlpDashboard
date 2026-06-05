import pytest
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import psycopg2

load_dotenv()

@pytest.fixture
def mongo_client():
    """Fixture pour la connexion MongoDB"""
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    yield client
    client.close()

@pytest.fixture
def postgres_conn():
    """Fixture pour la connexion PostgreSQL"""
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        database=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        sslmode='require'
    )
    yield conn
    conn.close()

@pytest.fixture
def sample_review():
    """Exemple de review pour les tests"""
    return {
        "review_text": "This game is amazing! I love the graphics and gameplay.",
        "app_name": "Test Game",
        "sentiment_label": "positive"
    }