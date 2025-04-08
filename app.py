# SEO-GAP-ANALYSIS/app.py
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for, jsonify, abort
import tempfile
import os
import sys
import traceback
import logging
import json # Importiere json für das detaillierte Logging

# Logging Konfiguration (Basis, falls noch nicht geschehen)
if not logging.getLogger().hasHandlers():
     # Temporär auf DEBUG setzen für die Fehlersuche
     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     logging.getLogger("matplotlib").setLevel(logging.WARNING) # Verhindert zu viele Matplotlib-Logs
     logging.getLogger("wordcloud").setLevel(logging.INFO)     # Wordcloud auf INFO lassen
     logging.getLogger("PIL").setLevel(logging.INFO)          # PIL/Pillow Logs reduzieren
     logging.getLogger("urllib3").setLevel(logging.INFO)      # Urllib3 Logs reduzieren
     logging.getLogger("openai").setLevel(logging.INFO)       # OpenAI Logs reduzieren
     logging.getLogger("spacy").setLevel(logging.INFO)        # Spacy Logs reduzieren
# Logger für dieses Modul holen
logger = logging.getLogger(__name__)

# Versuche, Konfiguration und Kernanalyse zu importieren
try:
    import config
    from core_analysis import run_analysis # Import aus core_analysis
except ImportError as e:
     logger.critical(f"FEHLER beim Importieren der Kernkomponenten in app.py: {e}", exc_info=True)
     sys.exit(1)

# --- Flask App Initialisierung ---
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key_change_me_in_production")
if app.secret_key == "default_secret_key_change_me_in_production":
    logger.warning("Unsicherer Flask Secret Key! Bitte FLASK_SECRET_KEY in .env setzen.")

# Lade JSON Config
config.load_config_from_json()

# --- Routen ---

@app.route("/", methods=["GET"])
def index():
    """Zeigt das Hauptformular an."""
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Verarbeitet die Analyseanfrage.
    Gibt bei Erfolg das Ergebnis-HTML zurück (Status 200),
    bei Validierungsfehlern JSON mit Fehlermeldung (Status 400),
    bei Serverfehlern JSON mit Fehlermeldung (Status 500).
    """
    temp_file_handle = None
    reference_path = None
    logger.info("Neue Analyseanfrage über /analyze erhalten.")

    try:
        # --- Formulardaten & Validierung ---
        query = request.form.get("query", "").strip()
        language = request.form.get("language", config.LANGUAGE)
        num_results_str = request.form.get("num_results", str(config.RESULTS_COUNT))
        include_ner = request.form.get("ner") == "true"
        include_clustering = request.form.get("cluster") == "true"
        include_sentiment = request.form.get("sentiment") == "true"
        reference_file = request.files.get("reference_file")

        logger.debug(f"Empfangene Formular-Optionen: NER={include_ner}, Cluster={include_clustering}, Sentiment={include_sentiment}") # Logge empfangene Optionen

        if not query:
            logger.warning("Analyseanfrage ohne Keyword erhalten.")
            return jsonify({"success": False, "error": "Bitte geben Sie ein Keyword ein."}), 400
        try:
            num_results = int(num_results_str)
            if not 1 <= num_results <= 100: raise ValueError("Anzahl außerhalb des erlaubten Bereichs.")
        except ValueError:
            logger.warning(f"Ungültige Anzahl Ergebnisse erhalten: '{num_results_str}'")
            return jsonify({"success": False, "error": "Bitte geben Sie eine gültige Anzahl (1-100) ein."}), 400

        if not config.SERP_API_KEY:
             logger.error("SerpApi API Schlüssel nicht konfiguriert.")
             return jsonify({"success": False, "error": "Server-Konfigurationsproblem (SerpApi Key fehlt)."}), 500

        # --- Referenzdatei verarbeiten ---
        if reference_file and reference_file.filename:
            if not reference_file.filename.lower().endswith('.txt'):
                logger.warning(f"Ungültiger Dateityp für Referenz hochgeladen: {reference_file.filename}")
                return jsonify({"success": False, "error": "Nur .txt-Dateien als Referenz erlaubt."}), 400
            try:
                temp_file_handle = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8')
                reference_file.stream.seek(0)
                file_content = reference_file.stream.read().decode('utf-8')
                temp_file_handle.write(file_content)
                reference_path = temp_file_handle.name
                temp_file_handle.close()
                logger.info(f"Referenzdatei '{reference_file.filename}' temporär gespeichert unter {reference_path}")
            except Exception as e:
                logger.exception(f"Fehler beim Verarbeiten der Referenzdatei '{reference_file.filename}'")
                return jsonify({"success": False, "error": "Fehler beim Verarbeiten der hochgeladenen Referenzdatei."}), 500

        # --- Analyse starten (Aufruf der Kernfunktion) ---
        logger.info(f"Starte run_analysis über /analyze für: '{query}' (Sprache: {language})")
        # Stelle sicher, dass die Optionen korrekt übergeben werden
        results = run_analysis(
            query=query,
            language=language,
            num_results=num_results,
            reference_file=reference_path,
            use_cache=True, # Cache für Web-UI standardmäßig an
            include_ner=include_ner, # Wird korrekt übergeben
            include_clustering=include_clustering, # Wird korrekt übergeben
            include_sentiment=include_sentiment, # Wird korrekt übergeben
            max_workers=5,
            output_format="all"
        )
        logger.info("run_analysis über /analyze abgeschlossen.")

        # --- Ergebnisse verarbeiten ---
        if results and results.get("success"):
             logger.info(f"Analyse für '{query}' erfolgreich. Rendere Ergebnis-HTML.")

             # --- NEU: Detailliertes Logging des results-Dictionary ---
             try:
                 # Versuche, eine übersichtliche JSON-Darstellung zu loggen
                 # Wandle DataFrame ggf. vorher um, falls es Probleme macht
                 results_copy = results.copy() # Kopie erstellen, um Original nicht zu ändern
                 if 'tfidf_dataframe' in results_copy:
                     # Konvertiere DataFrame zu Liste von Dicts für einfacheres Logging
                     results_copy['tfidf_dataframe'] = "DataFrame vorhanden (nicht im Detail geloggt)" # Platzhalter
                     # Alternative: results_copy['tfidf_dataframe'] = results_copy['tfidf_dataframe'].head().to_dict('records') # Nur Kopf loggen

                 results_str = json.dumps(results_copy, indent=2, ensure_ascii=False, default=str) # default=str für nicht-serialisierbare Objekte
                 logger.debug(f"Vollständiges 'results'-Dictionary vor render_template:\n{results_str}")
             except Exception as log_err:
                 logger.error(f"Fehler beim Loggen des results-Dictionary: {log_err}")
                 # Fallback: Logge zumindest die Keys
                 logger.debug(f"Keys im 'results'-Dictionary vor render_template: {results.keys() if isinstance(results, dict) else 'Kein Dict'}")
             # ---------------------------------------------------------


             analysis_summary = results.get("analysis_summary", {})
             if not isinstance(analysis_summary, dict):
                 logger.warning(f"analysis_summary ist kein Dictionary: {type(analysis_summary)}. Setze auf leeres Dict.")
                 analysis_summary = {}

             rendered_html = render_template(
                 "results.html",
                 query=results.get("query"),
                 analysis_summary=analysis_summary,
                 recommendations=results.get("recommendations"),
                 failed_urls=results.get("failed_urls", []),
                 related_questions=results.get("related_questions", []),
                 wordcloud_file=os.path.basename(results["wordcloud_file_path"]) if results.get("wordcloud_file_path") else None
             )
             return rendered_html, 200
        else:
             error_msg = results.get("error", "Unbekannter Analysefehler.") if results else "Analyse fehlgeschlagen."
             logger.error(f"Analyse für '{query}' fehlgeschlagen: {error_msg}")
             return jsonify({"success": False, "error": f"Analyse fehlgeschlagen: {error_msg}"}), 500

    except Exception as e:
        logger.exception(" Kritischer Fehler in der Flask-Route '/analyze'")
        return jsonify({"success": False, "error": "Ein unerwarteter Serverfehler ist aufgetreten."}), 500

    finally:
        # --- Temporäre Referenzdatei sicher löschen ---
        if reference_path and os.path.exists(reference_path):
            try: os.remove(reference_path); logger.debug(f"Temp Ref-Datei gelöscht: {reference_path}")
            except Exception as e: logger.error(f"Fehler Löschen Temp Ref-Datei {reference_path}: {e}")
        elif temp_file_handle and not temp_file_handle.closed:
             try:
                 temp_file_handle.close()
                 if os.path.exists(temp_file_handle.name): os.remove(temp_file_handle.name); logger.debug(f"Temp Ref-Datei (Fallback) gelöscht: {temp_file_handle.name}")
             except Exception as e: logger.error(f"Fehler Schließen/Löschen Temp Handle: {e}")


@app.route("/output/<path:filename>")
def serve_output(filename: str):
    """Liefert Dateien aus dem Output-Verzeichnis (z.B. Wortwolke)."""
    try:
        output_dir_abs = os.path.abspath(config.OUTPUT_DIR)
        logger.debug(f"Versuche Datei aus Output zu liefern: {os.path.join(output_dir_abs, filename)}")
        return send_from_directory(output_dir_abs, filename, as_attachment=False)
    except FileNotFoundError:
        logger.warning(f"Datei nicht im Output gefunden: {filename}")
        abort(404)
    except Exception as e:
         logger.error(f"Fehler Liefern Datei {filename}: {e}", exc_info=True)
         abort(500)

# --- App Start ---
if __name__ == "__main__":
    # Logging wird bereits am Anfang konfiguriert (inkl. DEBUG Level)
    flask_debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = '0.0.0.0' if not flask_debug or os.getenv('FLASK_RUN_HOST') == '0.0.0.0' else '127.0.0.1'
    logger.info(f"Starte Flask App (Debug={flask_debug}, Host={host})...")
    app.run(debug=flask_debug, host=host, port=5000)