# SEO-GAP-ANALYSIS/tests/test_core_analysis.py
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere zu testende Funktionen aus dem neuen Modul
from core_analysis import sanitize_filename, run_analysis, validate_openai_key
# Importiere auch config für Tests
import config

# --- Tests für sanitize_filename ---
@pytest.mark.parametrize("input_string, expected_output", [
    ("Normale Suchanfrage", "Normale_Suchanfrage"),
    ("Sonderzeichen /*?:<>|", "Sonderzeichen"),
    ("  Leerzeichen  am Anfang/Ende  ", "Leerzeichen_am_Anfang_Ende"),
    ("Langer Name über 100 Zeichen wird gekürzt und hoffentlich sinnvoll am Unterstrich getrennt abcdefghijklmnopqrstuvwxyz_abcdefghijklmnopqrstuvwxyz", "Langer_Name_über_100_Zeichen_wird_gekürzt_und_hoffentlich_sinnvoll_am_Unterstrich_getrennt"),
    ("NurSonderzeichen!!!", "NurSonderzeichen"),
    ("", "leerer_dateiname"),
    (".", "leerer_dateiname"),
    ("..", "leerer_dateiname"),
    ("test.datei.name_mit_Ümläuten", "test.datei.name_mit_Ümläuten"),
    ("test___datei", "test_datei"),
    ("_test_", "test"),
])
def test_sanitize_filename(input_string, expected_output):
    """Testet die sanitize_filename Funktion mit verschiedenen Eingaben."""
    assert sanitize_filename(input_string) == expected_output

# --- Tests für validate_openai_key ---
@patch('core_analysis.openai.OpenAI')
def test_validate_openai_key_success(mock_openai_class):
    """Testet gültigen OpenAI Key."""
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance
    mock_client_instance.models.list.return_value = None
    assert validate_openai_key("valid_key") is True
    mock_openai_class.assert_called_once_with(api_key="valid_key")
    mock_client_instance.models.list.assert_called_once()

@patch('core_analysis.openai.OpenAI')
def test_validate_openai_key_invalid(mock_openai_class):
    """Testet ungültigen OpenAI Key (AuthenticationError)."""
    from openai import AuthenticationError # Importiere Exception lokal
    mock_openai_class.side_effect = AuthenticationError(message="Invalid API key", response=MagicMock(), body=None)
    assert validate_openai_key("invalid_key") is False
    mock_openai_class.assert_called_once_with(api_key="invalid_key")

def test_validate_openai_key_no_key():
    """Testet Fall ohne Key."""
    assert validate_openai_key(None) is False
    assert validate_openai_key("") is False


# --- Tests für run_analysis ---
@patch('core_analysis._setup_analysis')
@patch('core_analysis._fetch_data')
@patch('core_analysis._load_reference_text')
@patch('core_analysis._perform_core_analysis')
@patch('core_analysis._generate_additional_outputs')
@patch('core_analysis._save_results')
def test_run_analysis_success_flow(mock_save, mock_generate, mock_perform, mock_load_ref, mock_fetch, mock_setup, tmp_path):
    """Testet den grundlegenden Erfolgs-Flow von run_analysis."""
    # Konfiguriere Mocks für einen erfolgreichen Durchlauf
    mock_nlp = MagicMock()
    mock_setup.return_value = mock_nlp
    # KORREKTUR: Mock gibt jetzt 4 Werte zurück (inkl. leerer related_questions)
    mock_fetch.return_value = (["text1", "text2"], ["url1", "url2"], [], [])
    mock_load_ref.return_value = None
    mock_tfidf_df = MagicMock(spec=pd.DataFrame)
    mock_tfidf_df.empty = False
    mock_tfidf_df.to_dict.return_value = [{'col1': 'val1'}]
    mock_analysis_summary = {"overall_top_terms_with_scores": [("term", 0.5)], "some_result": "value"}
    mock_perform.return_value = (mock_tfidf_df, mock_analysis_summary)
    mock_recommendations = "Mach dies und das."
    mock_wc_path = str(tmp_path / "wc.png")
    mock_generate.return_value = (mock_recommendations, mock_wc_path)
    mock_save.return_value = {"summary_json": "path/summary.json", "report_html": "path/report.html"}

    original_output_dir = config.OUTPUT_DIR
    config.OUTPUT_DIR = str(tmp_path)

    # Führe run_analysis aus
    result = run_analysis(query="test query", language="de", num_results=5, output_format="all")

    # Prüfe Ergebnis-Dictionary
    assert result["success"] is True
    assert result["query"] == "test query"
    assert result["language"] == "de"
    assert result["failed_urls"] == []
    assert result["analysis_summary"] == mock_analysis_summary
    assert result["recommendations"] == mock_recommendations
    assert result["wordcloud_file_path"] == mock_wc_path
    assert "tfidf_dataframe" in result
    assert "related_questions" in result # Prüfe ob Key da ist
    assert result["related_questions"] == [] # Prüfe den Wert
    assert result["output_files"] == mock_save.return_value

    # Prüfe, ob die gemockten Funktionen aufgerufen wurden
    mock_setup.assert_called_once_with("de")
    mock_fetch.assert_called_once()
    mock_load_ref.assert_called_once()
    mock_perform.assert_called_once()
    mock_generate.assert_called_once()
    # Prüfe Argumente von mock_generate (insbesondere related_questions)
    args_gen, kwargs_gen = mock_generate.call_args
    assert args_gen[0] == mock_analysis_summary # analysis_summary
    assert args_gen[1] == "test query"          # query
    assert args_gen[2] is None                  # reference_text
    assert args_gen[3] == []                    # related_questions
    # Prüfe Argumente von mock_save (insbesondere related_questions)
    mock_save.assert_called_once()
    args_save, kwargs_save = mock_save.call_args
    assert kwargs_save.get("related_questions") == []

    config.OUTPUT_DIR = original_output_dir


@patch('core_analysis._setup_analysis', return_value=None)
def test_run_analysis_spacy_fail(mock_setup):
    """Testet Fehler beim Laden des Spacy-Modells."""
    result = run_analysis(query="test query")
    assert result["success"] is False
    assert "Spacy-Modell" in result["error"]
    mock_setup.assert_called_once()

# Mock für _fetch_data angepasst, um 4 Werte zurückzugeben
@patch('core_analysis._setup_analysis', return_value=MagicMock())
@patch('core_analysis._fetch_data', return_value=([], [], [("url1", "Fetch Error")], []))
def test_run_analysis_fetch_fail(mock_fetch, mock_setup):
    """Testet Fehler beim Datenabruf."""
    result = run_analysis(query="test query")
    assert result["success"] is False
    assert "Keine Texte zur Analyse verfügbar" in result["error"]
    mock_setup.assert_called_once()
    mock_fetch.assert_called_once()

# TODO: Weitere Tests für run_analysis (Fehler in _perform, _generate, _save etc.) hinzufügen.