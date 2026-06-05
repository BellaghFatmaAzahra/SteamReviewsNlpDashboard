import os
import pandas as pd
import psycopg2
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def get_postgres_connection():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        database=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        sslmode="require",
    )
    return conn


def load_sentiment_data():
    conn = get_postgres_connection()
    query = """
        SELECT review_text, sentiment_label, confidence, model_name
        FROM sentiment_results
        LIMIT 10000
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_model_metrics():
    conn = get_postgres_connection()
    query = "SELECT * FROM model_metrics ORDER BY f1_score DESC"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def check_mongodb():
    try:
        uri = os.getenv("MONGO_URI")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client["steam_nlp"]
        count = db["raw_reviews"].count_documents({})
        client.close()
        return {"status": "ok", "count": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_postgresql():
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sentiment_results")
        count = cursor.fetchone()[0]
        conn.close()
        return {"status": "ok", "count": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}
