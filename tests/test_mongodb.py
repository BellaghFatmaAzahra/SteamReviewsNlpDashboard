import pytest
import os
from dotenv import load_dotenv

load_dotenv()

def test_mongodb_connection():
    """Test 1: Vérifier que MONGO_URI est défini dans .env"""
    uri = os.getenv("MONGO_URI")
    assert uri is not None, "MONGO_URI non défini dans .env"
    assert uri.startswith("mongodb+srv://"), "MONGO_URI invalide"
    print("✓ MongoDB : URI trouvé")

def test_mongodb_uri_format():
    """Test 2: Vérifier le format de l'URI MongoDB"""
    uri = os.getenv("MONGO_URI")
    assert "cluster0" in uri if uri else False
    print("✓ MongoDB : format URI OK")