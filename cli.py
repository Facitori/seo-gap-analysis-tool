#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time

# Logging Konfiguration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Eigene Module importieren
try:
    import config
    from cache_utils import clear_all_cache, clear_cache_for_query
    # Importiere aus core_analysis
    from core_analysis import run_analysis, validate_openai_key, HTML_TEMPLATE # HTML_TEMPLATE hier importieren
except ImportError as e:
    logger.critical(f"Import-Fehler in cli.py: {e}", exc_info=True)
    sys.exit(1)

# --- Main Execution Block ---
def main():
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(
        description="SEO Gap Analysis Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("query", help="Die Suchanfrage/Keyword für die Analyse.")
    parser.add_argument("-r", "--reference", metavar="FILE",
                        help="Pfad zu einer optionalen Referenztextdatei (.txt).")
    parser.add_argument("-o", "--output", metavar="PREFIX",
                        help="Optionales Präfix für die Namen der Ausgabedateien.")
    parser.add_argument("-l", "--language", default=None, choices=config.SPACY_MODEL_MAP.keys(),
                        help=f"Sprache für SERP/Analyse (Standard: '{config.LANGUAGE}'). Verfügbar: {', '.join(config.SPACY_MODEL_MAP.keys())}")
    parser.add_argument("-n", "--num-results", type=int, default=None, metavar="N",
                        help=f"Anzahl Suchergebnisse (1-100, Standard: {config.RESULTS_COUNT}).")
    parser.add_argument("-f", "--format", choices=["csv", "json", "html", "all"], default="all",
                        help="Ausgabeformat (Standard: all).")
    parser.add_argument("--ner", action="store_true", help="Named Entity Recognition (NER) aktivieren.")
    parser.add_argument("--cluster", action="store_true", help="Keyword-Clustering aktivieren.")
    parser.add_argument("--sentiment", action="store_true", help="Sentiment-Analyse aktivieren.")
    parser.add_argument("--workers", type=int, default=5, metavar="W",
                        help="Anzahl paralleler Worker (Standard: 5).")

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--no-cache", dest="use_cache", action="store_false",
                             help="Cache deaktivieren.")
    cache_group.add_argument("--invalidate-cache", action="store_true",
                             help="Cache für diese Query löschen.")
    cache_group.add_argument("--clear-cache", action="store_true",
                             help="Gesamten Cache löschen und beenden.")

    parser.add_argument("-c", "--config", metavar="JSON_FILE",
                        help="Pfad zu einer optionalen JSON-Konfigurationsdatei.")
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    # --- Vorabaktionen ---
    if args.clear_cache:
        logger.info("Lösche gesamten Cache...")
        clear_all_cache(); logger.info("Cache geleert."); sys.exit(0)

    if args.config:
        logger.info(f"Lade Konfiguration aus Datei: {args.config}")
        config.load_config_from_json(args.config)
    else:
        config.load_config_from_json() # Versuche Standard config.json

    # KORREKTUR der Zuweisung:
    effective_language = args.language if args.language is not None else config.LANGUAGE
    effective_num_results = args.num_results if args.num_results is not None else config.RESULTS_COUNT

    # Cache für spezifische Query invalidieren
    if args.invalidate_cache:
        logger.info(f"Invalidiere Cache für Query='{args.query}', Num={effective_num_results}, Lang={effective_language}...")
        clear_cache_for_query(args.query, effective_num_results, effective_language)

    # --- Validierungen ---
    openai_available = validate_openai_key(config.OPENAI_API_KEY)
    if not openai_available and args.format in ['html', 'all']:
        logger.warning("Kein gültiger OpenAI API-Schlüssel gefunden. Es werden keine Empfehlungen generiert.")

    if not config.SERP_API_KEY:
        logger.critical("FEHLER: Kein SerpApi API-Schlüssel gefunden. Abbruch."); sys.exit(1)

    if not (1 <= effective_num_results <= 100):
         logger.critical(f"FEHLER: Ungültige Anzahl Ergebnisse ({effective_num_results}). Muss zwischen 1 und 100 liegen. Abbruch."); sys.exit(1)

    # --- Analyse starten ---
    logger.info(f"Starte Analyse für '{args.query}' mit Parametern: Lang={effective_language}, Num={effective_num_results}")
    try:
        analysis_result = run_analysis(
            query=args.query, language=effective_language, num_results=effective_num_results,
            output_prefix=args.output, reference_file=args.reference, use_cache=args.use_cache,
            include_ner=args.ner, include_clustering=args.cluster, include_sentiment=args.sentiment,
            max_workers=args.workers, output_format=args.format
        )

        # --- Ergebnisverarbeitung ---
        if analysis_result.get("success"):
            print("\n" + "=" * 50); print(f"Analyse erfolgreich abgeschlossen! 🎉"); print(f"Dauer: {analysis_result.get('duration_seconds', 0):.2f} Sekunden.")
            if analysis_result.get("output_files"):
                print("Ergebnisse gespeichert in:"); show_files_for_format = { "csv": ["tfidf_csv", "summary_json"], "json": ["summary_json"], "html": ["report_html", "summary_json", "wordcloud_file"], "all": ["tfidf_csv", "summary_json", "recommendations_txt", "report_html", "wordcloud_file"]}; relevant_types = show_files_for_format.get(args.format, [])
                for file_type, file_path in analysis_result["output_files"].items():
                    base_name = os.path.basename(file_path); type_key = file_type; wc_path = analysis_result.get("wordcloud_file_path"); recos = analysis_result.get("recommendations")
                    if file_type == "wordcloud_file_path" and wc_path: type_key = "wordcloud_file"; base_name = os.path.basename(wc_path)
                    if type_key in relevant_types or type_key == "summary_json":
                         if type_key == "recommendations_txt":
                             if recos and "Fehler" not in recos and "Übersprungen" not in recos: print(f"- Empfehlungen (TXT): {base_name}")
                         elif type_key == "wordcloud_file":
                              if wc_path: print(f"- Wortwolke (PNG):     {base_name}")
                         elif type_key == "report_html":
                              if HTML_TEMPLATE: print(f"- HTML Report:         {base_name}") # HTML_TEMPLATE muss hier bekannt sein
                         elif type_key == "tfidf_csv": print(f"- TF-IDF Daten (CSV):  {base_name}")
                         elif type_key == "summary_json": print(f"- Zusammenfassung (JSON): {base_name}")
            else: print("Keine Ausgabedateien wurden explizit gespeichert.");
            if analysis_result.get("failed_urls"):
                print("-" * 30); print(f"Warnung: {len(analysis_result['failed_urls'])} URL(s) konnten nicht verarbeitet werden:");
                for url, reason in analysis_result["failed_urls"][:5]: print(f"  - {url} ({reason})")
                if len(analysis_result['failed_urls']) > 5: print("  ..."); print("(Details siehe JSON-Zusammenfassung)")
            print("=" * 50 + "\n")
        else: logger.error(f"Analyse fehlgeschlagen: {analysis_result.get('error', 'Unbekannter Fehler')}"); print(f"\nFEHLER bei der Analyse: {analysis_result.get('error', 'Details siehe Log.')}"); sys.exit(1)
    except Exception as e: logger.critical("Unerwarteter Fehler im CLI-Hauptablauf.", exc_info=True); print(f"\nEin unerwarteter Programmfehler ist aufgetreten: {e}"); print("Details wurden in die Log-Datei geschrieben."); sys.exit(1)

if __name__ == "__main__":
    main()