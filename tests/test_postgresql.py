import pytest
import os
from dotenv import load_dotenv

load_dotenv()

def test_postgresql_env_vars():
    """Test 1: Vérifier les variables PostgreSQL dans .env"""
    required_vars = ["PG_HOST", "PG_PORT", "PG_DATABASE", "PG_USER", "PG_PASSWORD"]
    
    for var in required_vars:
        value = os.getenv(var)
        assert value is not None, f"{var} non défini dans .env"
    
    assert "pooler.supabase.com" in os.getenv("PG_HOST", "")
    print("✓ PostgreSQL : variables d'environnement OK")

def test_postgresql_connection_params():
    """Test 2: Vérifier les paramètres de connexion"""
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")
    
    assert port == "5432", f"Port invalide: {port}"
    assert host is not None
    print("✓ PostgreSQL : paramètres OK")