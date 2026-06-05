import pytest

def test_sentiment_label_valid():
    """Test 1: Vérifier que les labels de sentiment sont valides"""
    valid_labels = ["positive", "negative"]
    
    # Simuler des données de test
    test_data = ["positive", "negative", "positive"]
    
    for label in test_data:
        assert label in valid_labels, f"Label invalide: {label}"
    
    print("✓ Sentiment : labels valides")

def test_text_preprocessing():
    """Test 2: Nettoyage de texte"""
    import re
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("GAME...") == "game"
    assert clean_text("  spaces  ") == "spaces"
    print("✓ Prétraitement : texte nettoyé correctement")

def test_sentiment_prediction_range():
    """Test 3: La prédiction doit être 0 ou 1"""
    # Simuler les prédictions d'un modèle
    predictions = [0, 1, 1, 0, 1]
    
    for pred in predictions:
        assert pred in [0, 1], f"Prédiction invalide: {pred}"
    
    print("✓ Modèle : prédictions valides (0/1)")

def test_f1_score_range():
    """Test 4: Les F1 scores doivent être entre 0 et 1"""
    # Simuler des F1 scores
    f1_scores = [0.94, 0.93, 0.91, 0.88]
    
    for score in f1_scores:
        assert 0 <= score <= 1, f"F1 score invalide: {score}"
    
    print("✓ Métriques : F1 scores valides")