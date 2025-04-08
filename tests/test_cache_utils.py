# SEO-GAP-ANALYSIS/tests/test_cache_utils.py
import sys
import os
import time
import pytest
import hashlib

# Pfad hinzufügen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Zu testende Funktionen importieren
import config # Brauchen wir für Cache-Pfade etc.
from cache_utils import get_cache_key, get_cache_path, is_cache_valid, save_to_cache, load_from_cache

# -- Hilfsfunktionen für Tests --
def create_dummy_cache_file(path: str, content: str = "test data"):
    """Erstellt eine temporäre Cache-Datei."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def remove_file_if_exists(path: str):
    """Löscht eine Datei, falls sie existiert."""
    if os.path.exists(path):
        os.remove(path)

# -- Tests --

def test_get_cache_key():
    """Testet die Generierung von Cache-Schlüsseln."""
    key1 = get_cache_key("type1", "query with space", 10, "de")
    key2 = get_cache_key("type1", "query with space", 10, "de")
    key3 = get_cache_key("type2", "query with space", 10, "de")
    key4 = get_cache_key("type1", "query with space", 20, "de")
    key5 = get_cache_key("type1", "QUERY with space", 10, "de") # Sollte gleich key1 sein wg. lower()

    assert key1 == key2
    assert key1 != key3
    assert key1 != key4
    assert key1 == key5 # Test auf Kleinschreibung
    # Prüfe ob es ein MD5 Hash ist (32 hex Zeichen)
    assert isinstance(key1, str)
    assert len(key1) == 32
    assert all(c in '0123456789abcdef' for c in key1)

def test_get_cache_path():
    """Testet die Erstellung von Cache-Pfaden."""
    key = "dummykey"
    path_json = get_cache_path("testtype", key, "json")
    path_txt = get_cache_path("testtype", key, "txt")

    expected_path_json = os.path.join(config.CACHE_DIR, f"testtype_{key}.json")
    expected_path_txt = os.path.join(config.CACHE_DIR, f"testtype_{key}.txt")

    assert path_json == expected_path_json
    assert path_txt == expected_path_txt
    # Stelle sicher, dass das Basisverzeichnis aus config verwendet wird
    assert path_json.startswith(config.CACHE_DIR)

# Tests für is_cache_valid, load und save (benötigen Dateisystem-Interaktion)
# pytest bietet `tmp_path` fixture für temporäre Verzeichnisse

def test_save_and_load_from_cache(tmp_path):
    """Testet das Speichern und Laden aus dem Cache (Text)."""
    # Wichtig: Überschreibe config.CACHE_DIR für diesen Test!
    original_cache_dir = config.CACHE_DIR
    config.CACHE_DIR = str(tmp_path) # Setze Cache Dir auf temporären Pfad

    cache_file = get_cache_path("test_sl", "mykey", "txt")
    test_data = "Das sind meine Testdaten."

    # 1. Test Laden (Datei existiert nicht)
    assert load_from_cache(cache_file) is None

    # 2. Test Speichern
    save_to_cache(test_data, cache_file)
    assert os.path.exists(cache_file)

    # 3. Test Laden (Datei existiert)
    loaded_data = load_from_cache(cache_file)
    assert loaded_data == test_data

    # Aufräumen nicht nötig, tmp_path macht das
    # Setze CACHE_DIR zurück
    config.CACHE_DIR = original_cache_dir


def test_is_cache_valid(tmp_path, mocker):
    """Testet die Gültigkeit des Caches basierend auf dem Alter."""
    original_cache_dir = config.CACHE_DIR
    config.CACHE_DIR = str(tmp_path)

    cache_file = get_cache_path("test_valid", "mykey", "txt")
    create_dummy_cache_file(cache_file)

    # Fall 1: Datei ist neu -> gültig
    config.MAX_CACHE_AGE_SECONDS = 3600 # 1 Stunde
    assert is_cache_valid(cache_file) is True

    # Fall 2: Datei ist älter als erlaubt -> ungültig
    # Mocke time.time(), um die Zeit "vorzuspulen"
    mock_time = mocker.patch('time.time')
    current_mtime = os.path.getmtime(cache_file)
    mock_time.return_value = current_mtime + config.MAX_CACHE_AGE_SECONDS + 60 # 1 Minute älter als erlaubt
    assert is_cache_valid(cache_file) is False

    # Fall 3: Datei existiert nicht -> ungültig
    remove_file_if_exists(cache_file)
    assert is_cache_valid(cache_file) is False

    config.CACHE_DIR = original_cache_dir