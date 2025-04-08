Absolut verständlich! Es ist gut, den Fortschritt festzuhalten.

Hier ist die aktualisierte `README.md`, die die Verbesserungen aus den letzten Iterationen (Parallelisierung, Trafilatura, Logging, Retries, Refactoring, Testing-Grundlagen) widerspiegelt:

```markdown
# SEO Gap Analysis Tool

Dieses Tool führt eine SEO-Gap-Analyse durch, indem es die Top-Suchergebnisse für ein bestimmtes Keyword analysiert, deren Inhalte extrahiert und verschiedene textanalytische Verfahren anwendet. Es hilft dabei, relevante Begriffe, Themen und Entitäten zu identifizieren, die in den Inhalten der Wettbewerber vorkommen, und generiert Optimierungsempfehlungen.

## Features

*   **SERP-Analyse:** Abrufen der Top-Suchergebnisse von Google für ein Keyword (via SerpApi).
*   **Robuste Inhaltsextraktion:** Extrahiert den Haupttextinhalt von Webseiten mithilfe von `trafilatura`.
*   **TF-IDF-Analyse:** Identifiziert die wichtigsten Begriffe (Unigramme und Bigramme) in den Wettbewerbertexten insgesamt und pro URL.
*   **Vergleich mit Referenztext:** (Optional) Vergleicht die gefundenen Top-Begriffe mit einem eigenen Text, um fehlende Begriffe zu identifizieren.
*   **Named Entity Recognition (NER):** (Optional) Erkennt Personen, Organisationen und Orte/Regionen in den Texten (via Spacy).
*   **Keyword-Clustering:** (Optional) Gruppiert relevante Keywords thematisch (via Scikit-learn KMeans).
*   **Sentiment-Analyse:** (Optional) Bestimmt die durchschnittliche Tonalität der Wettbewerbertexte (via TextBlob).
*   **KI-Empfehlungen:** Generiert konkrete SEO-Optimierungsvorschläge basierend auf der Analyse (via OpenAI API).
*   **Visualisierung:** Erstellt eine Wortwolke der wichtigsten Begriffe.
*   **Caching:** Zwischenspeichert SERP-Ergebnisse und extrahierte Texte, um wiederholte Abrufe und API-Kosten zu reduzieren (zentralisiert in `cache_utils.py`).
*   **Parallele Verarbeitung:** Extrahiert Texte von mehreren URLs gleichzeitig (`ThreadPoolExecutor`), um die Ausführungszeit zu verkürzen.
*   **Netzwerk-Robustheit:** Implementiert automatische Wiederholungsversuche (Retries mit `tenacity`) für API-Aufrufe und Webseiten-Downloads.
*   **Verbessertes Logging:** Verwendet das `logging`-Modul für detaillierte Status- und Fehlermeldungen statt einfacher `print`-Anweisungen.
*   **Code Refactoring:** Die Kernanalyse-Logik in `cli.py` wurde in kleinere, wartbare Funktionen aufgeteilt.
*   **Ausgabeformate:** Generiert Ergebnisse als CSV (TF-IDF), JSON (Zusammenfassung), TXT (Empfehlungen) und einen umfassenden HTML-Report.
*   **Schnittstellen:** Bietet sowohl ein Command-Line Interface (CLI) als auch eine Web User Interface (Web UI via Flask mit Ladeindikator).
*   **Konfiguration:** Einstellungen und API-Keys werden primär über `.env` verwaltet, mit optionalen Überschreibungen durch `config.json`.
*   **Testing:** Erste Unit-Tests mit `pytest` und `pytest-mock` für Kernfunktionen und gemockte API-Calls sind implementiert.

## Technologie-Stack

*   Python 3.10+
*   Flask (für Web UI)
*   Requests (für HTTP-Anfragen)
*   Trafilatura (für Inhaltsextraktion)
*   Pandas (für Datenmanipulation, TF-IDF)
*   Scikit-learn (für TF-IDF, Clustering)
*   Spacy (für NLP-Preprocessing, NER)
    *   Benötigt Sprachmodelle (z.B. `de_core_news_sm`)
*   TextBlob (für Sentiment-Analyse)
*   OpenAI Python Client (für GPT-Empfehlungen)
*   WordCloud & Matplotlib (für Visualisierung)
*   Jinja2 (für HTML-Templating)
*   python-dotenv (zum Laden von `.env`-Dateien)
*   tenacity (für Retries)
*   tqdm (für Fortschrittsbalken im CLI)
*   pytest & pytest-mock (für Tests)
*   *(html2text, beautifulsoup4 könnten evtl. entfernt werden, falls nicht mehr genutzt)*

## Voraussetzungen

*   Python 3.10 oder höher
*   `pip` (Python Package Installer)
*   Git (optional, zum Klonen)
*   API-Schlüssel für:
    *   SerpApi ([https://serpapi.com/](https://serpapi.com/))
    *   OpenAI API ([https://platform.openai.com/](https://platform.openai.com/))

## Installation & Setup

1.  **Repository klonen (falls nötig):**
    ```bash
    git clone <repository_url>
    cd SEO-GAP-ANALYSIS
    ```
2.  **Virtuelle Umgebung erstellen & aktivieren:**
    ```bash
    python3 -m venv cleanenv
    source cleanenv/bin/activate
    # Unter Windows: .\cleanenv\Scripts\activate
    ```
3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Spacy Sprachmodelle herunterladen:**
    ```bash
    python -m spacy download de_core_news_sm
    # Optional: python -m spacy download en_core_web_sm
    ```
5.  **API-Schlüssel konfigurieren:**
    *   Erstelle/Bearbeite die Datei `.env` im Hauptverzeichnis.
    *   Füge deine Schlüssel hinzu:
        ```dotenv
        OPENAI_API_KEY="sk-DEIN_OPENAI_SCHLUESSEL"
        SERP_API_KEY="DEIN_SERPAPI_SCHLUESSEL"
        # Optional: FLASK_SECRET_KEY=ein_zufaelliger_geheimer_schluessel
        ```
    *   **WICHTIG:** Stelle sicher, dass `.env` in deiner `.gitignore` steht!

## Konfiguration

*   **`.env`:** Enthält API-Schlüssel und andere Umgebungsvariablen. Wird nicht versioniert.
*   **`config.json`:** (Optional) Überschreibt nicht-sensitive Standardwerte aus `config.py`.
*   **`config.py`:** Definiert Standardwerte, lädt `.env` und `config.json`.

## Verwendung

### Kommandozeilen-Interface (CLI)

(Virtuelle Umgebung muss aktiviert sein)

**Basisaufruf:**
```bash
python cli.py "Dein Keyword"
```

**Wichtige Optionen:**
*   `-l SPRACHE`: Sprache (`de`, `en`, ...). Standard: `de`.
*   `-n ANZAHL`: Anzahl Ergebnisse. Standard: 10.
*   `-r DATEI`: Pfad zur Referenzdatei (.txt).
*   `-o PREFIX`: Präfix für Ausgabedateien.
*   `-f FORMAT`: Ausgabeformat (`csv`, `json`, `html`, `all`). Standard: `all`.
*   `--ner`, `--cluster`, `--sentiment`: Analyse-Optionen aktivieren.
*   `--workers ANZAHL`: Parallele Worker für Extraktion. Standard: 5.
*   `--no-cache`, `--invalidate-cache`, `--clear-cache`: Cache-Optionen.
*   `-c DATEI`: Pfad zu `config.json`.

**Beispiel:**
```bash
python cli.py "nachhaltige mode" -l de -n 12 --ner --workers 8
```

### Web User Interface (Web UI)

1.  **Starte die Flask-App:**
    ```bash
    python app.py
    ```
2.  **Öffne den Browser:** Gehe zu `http://127.0.0.1:5000` (oder die angezeigte Adresse).
3.  **Gib Daten ein** und klicke "Analysieren". Ein Ladeindikator erscheint.
4.  **Ergebnisse** werden auf derselben Seite angezeigt.

Generierte Dateien landen im `output/`-Verzeichnis.

## Projektstruktur

```
SEO-GAP-ANALYSIS/
├── .env
├── .gitignore
├── app.py
├── cache_utils.py
├── cli.py
├── config.json
├── config.py
├── modules/
│   ├── __init__.py
│   ├── extractor.py     # Neu mit trafilatura
│   ├── openai_helper.py # Mit Retries
│   ├── serp_api.py      # Mit Retries
│   ├── tf_idf.py
│   └── visualization.py
├── output/
│   └── cache/
├── templates/
│   ├── cli_report.html
│   ├── index.html       # Mit Ladeindikator JS
│   └── results.html
├── tests/
│   ├── __init__.py
│   ├── test_cache_utils.py
│   ├── test_cli_utils.py
│   ├── test_extractor.py # Neu
│   ├── test_openai.py    # Überarbeitet (Mocking)
│   ├── test_serp_api.py  # Neu
│   └── test_tf_idf.py    # Neu (Basis)
├── requirements.txt      # Aktualisiert
└── README.md             # Diese Datei
```

## Testing

Die Testsuite verwendet `pytest`. Führe die Tests aus dem Hauptverzeichnis aus:

```bash
pytest
```
Die Tests nutzen `pytest-mock`, um externe API-Aufrufe zu simulieren.

## Letzte Verbesserungen

*   **Performance:** Parallele Text-Extraktion.
*   **Robustheit:** `trafilatura` für Extraktion, Netzwerk-Retries (`tenacity`).
*   **Wartbarkeit:** Code-Refactoring (`cli.py`), verbessertes Logging.
*   **Testing:** `pytest` eingeführt, erste Unit-Tests und Mocking implementiert.
*   **UX:** Ladeindikator im Web UI.

## Zukünftige Verbesserungen

*   Fehlgeschlagene Tests korrigieren/anpassen.
*   Testabdeckung erhöhen (TF-IDF-Logik, Flask-Routen).
*   Analysefunktionen erweitern (Fragen, Lesbarkeit, ...).
*   Prompt optimieren
*   Konfigurierbarkeit verbessern.
*   Deployment (Docker, Gunicorn, ...).

```