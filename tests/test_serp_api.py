# SEO-GAP-ANALYSIS/tests/test_serp_api.py
import sys
import os
import pytest
import requests
from unittest.mock import MagicMock, patch
import json # Importiere json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Importiere die zu testende Funktion und den TypedDict
from modules.serp_api import get_serp_results, SerpResults
import config
# Importiere Cache-Funktionen
from cache_utils import clear_cache_for_query, get_cache_key, get_cache_path, save_to_cache, load_from_cache

# Mock-Antworten
MOCK_SUCCESS_RESPONSE_JSON = {
    "search_information": {"query_displayed": "test query"}, # Beispiel Metadaten
    "organic_results": [
        {"position": 1, "title": "Result 1", "link": "https://e.com/1", "snippet": "..."},
        {"position": 2, "title": "Result 2", "link": "https://e.com/2", "snippet": "..."},
        {"position": 3, "title": "Result 4", "link": "https://e.com/4", "snippet": "..."} # Position übersprungen
    ],
    "related_questions": [ # Beispiel PAA
        {"question": "What is SERP?", "snippet": "...", "link": "..."},
        {"question": "How does SERP API work?", "snippet": "...", "link": "..."}
    ]
}
MOCK_EMPTY_RESPONSE_JSON = {"organic_results": [], "related_questions": []}

# Standard-Rückgabe bei Fehlern
DEFAULT_ERROR_RETURN: SerpResults = {"organic_results": [], "related_questions": [], "error": "Some error occurred"}

@pytest.fixture(autouse=True)
def clear_test_cache(tmp_path):
    """Setzt Cache auf tmp_path und leert ihn vor/nach dem Test."""
    original_cache_dir = config.CACHE_DIR
    test_cache_dir = tmp_path / "serp_cache" # Eindeutiges Verzeichnis
    config.CACHE_DIR = str(test_cache_dir)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Leeren vor dem Test
    queries_to_clear = ["test query", "empty query", "error query", "cache test query"]
    for q in queries_to_clear:
        cache_key = get_cache_key("serp_v2", q, 5, "en") # Verwende v2 Key
        cache_file = get_cache_path("serp_v2", cache_key, extension="json")
        if os.path.exists(cache_file):
            try: os.remove(cache_file)
            except OSError as e: print(f"Warnung: Konnte Cache nicht löschen: {e}")
    yield
    config.CACHE_DIR = original_cache_dir


# --- Tests ---
def test_get_serp_results_success(mocker):
    """Testet erfolgreichen Abruf mit organischen Ergebnissen und PAA."""
    mock_response = MagicMock(spec=requests.Response); mock_response.status_code = 200
    mock_response.json.return_value = MOCK_SUCCESS_RESPONSE_JSON; mock_response.raise_for_status = MagicMock()
    mock_get = mocker.patch('modules.serp_api.requests.get', return_value=mock_response)
    original_key = config.SERP_API_KEY; config.SERP_API_KEY = "dummy_test_key"
    query = "test query"; num = 2; lang = "en" # Frage nur 2 an, Mock hat 3

    results_dict = get_serp_results(query, num_results=num, use_cache=False, language=lang)

    assert isinstance(results_dict, dict)
    assert results_dict["error"] is None
    # Prüfe organische Ergebnisse (sollte auf num=2 begrenzt sein)
    assert "organic_results" in results_dict
    organic_results = results_dict["organic_results"]
    assert isinstance(organic_results, list)
    assert len(organic_results) == num # Sollte jetzt auf angeforderte Anzahl begrenzt sein
    assert organic_results[0]['title'] == "Result 1"
    assert organic_results[1]['title'] == "Result 2"
    # Prüfe PAA Fragen
    assert "related_questions" in results_dict
    paa_questions = results_dict["related_questions"]
    assert isinstance(paa_questions, list)
    assert len(paa_questions) == 2
    assert paa_questions[0] == "What is SERP?"

    mock_get.assert_called_once()
    config.SERP_API_KEY = original_key

def test_get_serp_results_empty(mocker):
    """Testet Fall mit leeren organischen Ergebnissen."""
    mock_response = MagicMock(spec=requests.Response); mock_response.status_code = 200
    mock_response.json.return_value = MOCK_EMPTY_RESPONSE_JSON; mock_response.raise_for_status = MagicMock()
    mock_get = mocker.patch('modules.serp_api.requests.get', return_value=mock_response)
    config.SERP_API_KEY = "dummy_test_key"

    results_dict = get_serp_results("empty query", 5, False, "en")

    assert isinstance(results_dict, dict)
    assert results_dict["error"] is None
    assert results_dict["organic_results"] == []
    assert results_dict["related_questions"] == [] # Keine PAA im Mock
    mock_get.assert_called_once()

def test_get_serp_results_http_error(mocker):
    """Testet HTTP-Fehler (z.B. 403)."""
    mock_response = MagicMock(spec=requests.Response); mock_response.status_code = 403; mock_response.reason = "Forbidden"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_get = mocker.patch('modules.serp_api.requests.get', return_value=mock_response)
    config.SERP_API_KEY = "dummy_test_key"

    # KORREKTUR: Erwarte ein Dictionary mit Fehlermeldung
    results_dict = get_serp_results("error query", 5, False, "en")

    assert isinstance(results_dict, dict)
    assert results_dict["error"] is not None
    # Prüfe, ob der Fehlertext die Exception enthält
    assert "Status: 403" in results_dict["error"] or "HTTPError" in results_dict["error"]
    assert results_dict["organic_results"] == []
    assert results_dict["related_questions"] == []
    mock_get.assert_called_once()

def test_get_serp_results_timeout_after_retries(mocker):
    """Testet Timeout-Fehler nach Retries."""
    mock_get = mocker.patch('modules.serp_api.requests.get', side_effect=requests.exceptions.Timeout("Test Timeout"))
    config.SERP_API_KEY = "dummy_test_key"

    # KORREKTUR: Erwarte ein Dictionary mit Fehlermeldung
    results_dict = get_serp_results("error query", num_results=5, use_cache=False, language="en")

    assert isinstance(results_dict, dict)
    assert results_dict["error"] is not None
    assert "Timeout" in results_dict["error"] or "RequestException" in results_dict["error"]
    assert results_dict["organic_results"] == []
    assert results_dict["related_questions"] == []
    assert mock_get.call_count == 3

def test_get_serp_results_caching(mocker, tmp_path):
    """Testet das Caching-Verhalten mit dem neuen Dictionary-Format."""
    # Mock für den ersten Aufruf (API)
    mock_response1 = MagicMock(spec=requests.Response); mock_response1.status_code = 200
    mock_response1.json.return_value = MOCK_SUCCESS_RESPONSE_JSON; mock_response1.raise_for_status = MagicMock()
    mock_get1 = mocker.patch('modules.serp_api.requests.get', return_value=mock_response1)

    config.SERP_API_KEY = "dummy_test_key"; query = "cache test query"; num = 2; lang = "en"

    # 1. Erster Aufruf -> API Call, speichert im Cache
    results1 = get_serp_results(query, num, True, lang)
    assert isinstance(results1, dict); assert len(results1["organic_results"]) == num; assert len(results1["related_questions"]) == 2; mock_get1.assert_called_once()

    # Prüfe Cache-Datei
    cache_key = get_cache_key("serp_v2", query, num, lang)
    cache_file = get_cache_path("serp_v2", cache_key, extension="json")
    assert os.path.exists(cache_file)
    # Lade und vergleiche den Inhalt (ignoriere 'error'-Key, falls er None ist)
    with open(cache_file, 'r') as f: cached_data = json.load(f)
    assert cached_data["organic_results"] == results1["organic_results"]
    assert cached_data["related_questions"] == results1["related_questions"]

    # 2. Zweiter Aufruf -> Sollte aus Cache laden
    mock_get2 = mocker.patch('modules.serp_api.requests.get') # Neuer Mock
    results2 = get_serp_results(query, num, True, lang)
    assert isinstance(results2, dict)
    assert results2["organic_results"] == results1["organic_results"]
    assert results2["related_questions"] == results1["related_questions"]
    assert results2["error"] is None
    mock_get2.assert_not_called()