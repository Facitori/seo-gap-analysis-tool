# SEO-GAP-ANALYSIS/tests/test_visualization.py
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Füge das Projektverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere die zu testende Funktion
from modules.visualization import generate_wordcloud

# Testdaten
DUMMY_TERMS = ["seo", "analyse", "tool", "keywords", "optimierung", "ranking"]

# Testfunktion mit Mocking von matplotlib und WordCloud
@patch('modules.visualization.WordCloud') # Mocke die WordCloud-Klasse
@patch('modules.visualization.plt')      # Mocke matplotlib.pyplot
def test_generate_wordcloud_success(mock_plt, mock_wordcloud_class, tmp_path):
    """
    Testet, ob generate_wordcloud erfolgreich durchläuft und die
    matplotlib/WordCloud-Funktionen aufruft.
    """
    # Konfiguriere den Mock für die WordCloud-Instanz und deren Methoden
    mock_wc_instance = MagicMock()
    # KORREKTUR: Mocke die generate-Methode, sie soll die Instanz selbst zurückgeben
    mock_wc_instance.generate.return_value = mock_wc_instance
    # Sorge dafür, dass der Aufruf von WordCloud(...) die mock_wc_instanz zurückgibt
    mock_wordcloud_class.return_value = mock_wc_instance

    # Definiere den Ausgabepfad im temporären Verzeichnis
    output_file = tmp_path / "test_wordcloud.png"

    # Rufe die Funktion auf
    generate_wordcloud(DUMMY_TERMS, str(output_file))

    # Prüfe, ob WordCloud mit den richtigen Parametern instanziiert wurde
    mock_wordcloud_class.assert_called_once()
    # Prüfe die Argumente (Breite, Höhe, Hintergrundfarbe sind in generate_wordcloud hartcodiert)
    args, kwargs = mock_wordcloud_class.call_args
    assert kwargs.get('width') == 800
    assert kwargs.get('height') == 400
    assert kwargs.get('background_color') == "white"

    # Prüfe, ob die generate-Methode der WordCloud-Instanz aufgerufen wurde
    mock_wc_instance.generate.assert_called_once_with(" ".join(DUMMY_TERMS))

    # Prüfe, ob die matplotlib-Funktionen aufgerufen wurden
    mock_plt.figure.assert_called_once()
    mock_plt.imshow.assert_called_once()
    # Prüfe das erste Argument von imshow (sollte jetzt die gemockte WordCloud-Instanz sein)
    imshow_args, imshow_kwargs = mock_plt.imshow.call_args
    assert imshow_args[0] == mock_wc_instance # Sollte jetzt stimmen
    assert imshow_kwargs.get('interpolation') == "bilinear"
    mock_plt.axis.assert_called_once_with("off")
    mock_plt.savefig.assert_called_once_with(str(output_file))
    # Prüfe, ob close mit dem Figure-Objekt aufgerufen wurde (das von plt.figure zurückgegeben wird)
    # Wenn plt.figure gemockt ist, gibt es standardmäßig ein MagicMock-Objekt zurück.
    mock_plt.close.assert_called_once_with(mock_plt.figure.return_value)


@patch('modules.visualization.WordCloud')
@patch('modules.visualization.plt')
def test_generate_wordcloud_no_terms(mock_plt, mock_wordcloud_class):
    """
    Testet das Verhalten, wenn eine leere Liste von Begriffen übergeben wird.
    """
    output_file = "dummy.png"
    generate_wordcloud([], output_file)

    # Stelle sicher, dass WordCloud und matplotlib *nicht* aufgerufen werden
    mock_wordcloud_class.assert_not_called()
    mock_plt.figure.assert_not_called()
    mock_plt.savefig.assert_not_called()
    mock_plt.close.assert_not_called() # Auch close wird nicht aufgerufen

@patch('modules.visualization.WordCloud')
@patch('modules.visualization.plt')
def test_generate_wordcloud_savefig_error(mock_plt, mock_wordcloud_class, tmp_path, caplog):
    """
    Testet das Verhalten, wenn plt.savefig einen Fehler wirft.
    """
    mock_wc_instance = MagicMock()
    mock_wc_instance.generate.return_value = mock_wc_instance
    mock_wordcloud_class.return_value = mock_wc_instance

    # Simuliere einen Fehler beim Speichern
    mock_plt.savefig.side_effect = OSError("Kann Datei nicht schreiben")

    output_file = tmp_path / "error_cloud.png"

    # Rufe die Funktion auf (erwarte keinen Fehler, da er intern gefangen wird)
    generate_wordcloud(DUMMY_TERMS, str(output_file))

    # Prüfe, ob der Fehler geloggt wurde
    assert "Fehler beim Generieren der Wortwolke" in caplog.text
    assert "Kann Datei nicht schreiben" in caplog.text

    # Prüfe, dass close trotzdem aufgerufen wurde (falls möglich), mit dem Figure-Objekt
    mock_plt.close.assert_called_once_with(mock_plt.figure.return_value) # Sollte jetzt im finally passieren