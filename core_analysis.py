# SEO-GAP-ANALYSIS/core_analysis.py

# --- Standard Library Imports ---
import os
import sys
import json
import logging
import re
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

# --- Third Party Imports ---
import openai
import pandas as pd
import spacy
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from tqdm import tqdm

# --- Eigene Modul-Imports ---
try:
    import config
    from cache_utils import (
        clear_all_cache, clear_cache_for_query, load_from_cache,
        save_to_cache, get_cache_key, get_cache_path
    )
    from modules.serp_api import get_serp_results, SerpResults
    from modules.extractor import extract_text_from_url
    import modules.tf_idf as tfidf_module
    from modules.tf_idf import load_spacy_model
    from modules.openai_helper import generate_recommendations
    from modules.visualization import generate_wordcloud
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"FEHLER: Notwendige Module konnten nicht importiert werden in core_analysis.py: {e}", exc_info=True)
    logging.critical("Stelle sicher, dass du das Skript aus dem Hauptverzeichnis (SEO-GAP-ANALYSIS) ausführst oder die PYTHONPATH-Variable korrekt gesetzt ist.")
    sys.exit(1)

# --- Logger Initialisierung ---
logger = logging.getLogger(__name__)
# Setze Level hier testweise auf DEBUG, um alle Meldungen zu sehen
# In Produktion sollte dies über eine Konfiguration gesteuert werden
# logging.getLogger().setLevel(logging.DEBUG) # Optional: Für detaillierte Logs hier aktivieren

# --- Konstanten & Globale Variablen ---
HTML_TEMPLATE = None
try:
    possible_template_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'templates'),
        os.path.join(os.path.dirname(__file__), 'templates'),
        'templates'
    ]
    template_dir = None
    for path_option in possible_template_paths:
        if os.path.isdir(path_option):
            template_dir = path_option
            break
    if template_dir:
        logger.info(f"Versuche HTML-Template aus '{template_dir}' zu laden.")
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
        HTML_TEMPLATE = env.get_template('cli_report.html')
        logger.info("HTML-Template für Reports erfolgreich geladen.")
    else: logger.warning(f"Template-Verzeichnis nicht gefunden. Gesucht in: {possible_template_paths}"); HTML_TEMPLATE = None
except TemplateNotFound:
     logger.warning(f"Template 'cli_report.html' nicht im Verzeichnis '{template_dir}' gefunden."); HTML_TEMPLATE = None
except Exception as e: logger.warning(f"Fehler beim Laden des HTML-Templates: {e}", exc_info=True); HTML_TEMPLATE = None

# --- Hilfsfunktionen ---
# (validate_openai_key und sanitize_filename bleiben unverändert)
def validate_openai_key(api_key: Optional[str]) -> bool:
    if not api_key: logger.error("Kein OpenAI API-Schlüssel übergeben."); return False
    try:
        logger.info("Prüfe OpenAI API-Schlüssel..."); client = openai.OpenAI(api_key=api_key)
        client.models.list(); logger.info("OpenAI API-Schlüssel ist gültig."); return True
    except openai.AuthenticationError: logger.error("Ungültiger OpenAI API-Schlüssel."); return False
    except Exception as e: logger.error(f"Fehler bei der Verbindung zu OpenAI: {e}"); return False

def sanitize_filename(filename: str) -> str:
    if not filename: return "leerer_dateiname"
    sanitized = re.sub(r'[\s\\/:*?"<>|]+', '_', filename)
    sanitized = re.sub(r'[^\w.\-äöüÄÖÜß_]', '', sanitized)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('._')
    max_len = 100
    if len(sanitized) > max_len:
        cut_point = -1;
        for char in ['_', '.']:
            try: cut_point = max(cut_point, sanitized[:max_len].rindex(char))
            except ValueError: pass
        if cut_point > max_len * 0.6: sanitized = sanitized[:cut_point]
        else: sanitized = sanitized[:max_len]
        sanitized = sanitized.strip('._')
    return sanitized if sanitized else "leerer_dateiname"

# --- Kernanalyse-Hilfsfunktionen ---
def _setup_analysis(language: str) -> Optional[spacy.language.Language]:
    spacy_model_name = config.get_spacy_model_for_language(language)
    logger.info(f"Lade Spacy-Modell: {spacy_model_name}...")
    nlp = load_spacy_model(spacy_model_name)
    if nlp is None: logger.error(f"Spacy-Modell '{spacy_model_name}' konnte nicht geladen werden.")
    return nlp

def _fetch_data(
    query: str, num_results: int, language: str, use_cache: bool, max_workers: int
) -> Tuple[List[str], List[str], List[Tuple[str, str]], List[str]]:
    logger.info(f"Rufe SERP-Daten für '{query}' ab (Sprache: {language}, Anzahl: {num_results}, Cache: {use_cache})...")
    serp_data: SerpResults = get_serp_results(query, num_results=num_results, use_cache=use_cache, language=language)
    organic_results = serp_data.get("organic_results", []); related_questions = serp_data.get("related_questions", []); serp_error = serp_data.get("error")
    if serp_error: logger.error(f"Fehler von get_serp_results: {serp_error}"); return [], [], [("SERP API", serp_error)], []
    if not organic_results: logger.error("Keine organischen SERP-Ergebnisse erhalten."); return [], [], [("SERP API", "Keine organischen Ergebnisse")], related_questions
    urls = [result["url"] for result in organic_results if "url" in result]; logger.info(f"-> {len(urls)} URLs extrahiert.")
    if not urls: return [], [], [("SERP API", "Keine URLs in Ergebnissen")], related_questions
    logger.info(f"Extrahiere Texte von {len(urls)} URLs mit {max_workers} Worker(n)...")
    results_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}; failed_urls_with_reason: List[Tuple[str, str]] = []
    valid_texts: List[str] = []; valid_urls: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_text_from_url, url, use_cache): url for url in urls}
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Extrahiere Texte", unit="url"):
            url = future_to_url[future]
            try:
                text, error_msg = future.result(); results_map[url] = (text, error_msg)
                if error_msg: logger.warning(f"Fehler Extraktion {url}: {error_msg}"); failed_urls_with_reason.append((url, error_msg))
                elif text: valid_texts.append(text); valid_urls.append(url)
                else: err = "Kein Text/Fehler."; logger.warning(f"Problem {url}: {err}"); failed_urls_with_reason.append((url, err))
            except Exception as exc: logger.error(f"Executor-Fehler {url}: {exc}", exc_info=True); results_map[url] = (None, f"Exec-Fehler: {exc}"); failed_urls_with_reason.append((url, f"Exec-Fehler: {exc}"))
    logger.info(f"-> Text von {len(valid_texts)} URLs extrahiert.");
    if failed_urls_with_reason: logger.warning(f"-> Fehler bei {len(failed_urls_with_reason)} URLs.")
    return valid_texts, valid_urls, failed_urls_with_reason, related_questions

def _load_reference_text(reference_file_path: Optional[str]) -> Optional[str]:
    if not reference_file_path: return None
    logger.info(f"Lade Referenztext von: {reference_file_path}...")
    try:
        if os.path.exists(reference_file_path):
            with open(reference_file_path, 'r', encoding='utf-8') as f: text = f.read()
            if text: logger.info(f"-> Ref-Text ({len(text)} Zeichen) geladen."); return text
            else: logger.warning("Ref-Datei ist leer."); return None
        else: logger.warning(f"Ref-Datei nicht gefunden: {reference_file_path}"); return None
    except Exception as e: logger.error(f"Fehler Laden Ref-Text: {e}", exc_info=True); return None

def _perform_core_analysis(
    texts: List[str], urls: List[str], nlp: spacy.language.Language, reference_text: Optional[str],
    include_ner: bool, include_clustering: bool, include_sentiment: bool
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    logger.info("Führe Kernanalyse durch (TF-IDF, NER, Clustering, Sentiment)...")
    try:
        tfidf_df, analysis_summary = tfidf_module.perform_tf_idf_analysis(
            texts=texts, urls=urls, nlp=nlp, reference_text=reference_text,
            include_ner=include_ner, include_clustering=include_clustering, include_sentiment=include_sentiment
        )
        if tfidf_df is None and isinstance(analysis_summary, dict) and "error" in analysis_summary:
            logger.error(f"Fehler in perform_tf_idf_analysis: {analysis_summary['error']}")
            return None, analysis_summary
        elif tfidf_df is None:
             logger.error("TF-IDF gab None DataFrame ohne Fehlermeldung zurück."); return None, {"error": "Unbek. Fehler TF-IDF Analyse."}
        logger.info("-> Kernanalyse abgeschlossen.")
        return tfidf_df, analysis_summary
    except Exception as e: logger.exception("Unerwarteter Fehler während Kernanalyse"); return None, {"error": f"Unerw. Fehler Kernanalyse: {e}"}

def _generate_additional_outputs(
    analysis_summary: Dict[str, Any], query: str, reference_text: Optional[str],
    related_questions: List[str], output_base_path: str
) -> Tuple[Optional[str], Optional[str]]:
    recommendations: Optional[str] = None; wordcloud_file_path: Optional[str] = None
    logger.info("Generiere Wortwolke...")
    try:
        terms_for_wc = [term for term, score in analysis_summary.get("overall_top_terms_with_scores", [])[:50]]
        if terms_for_wc:
            wc_file = f"{output_base_path}_wordcloud.png"; generate_wordcloud(terms_for_wc, wc_file); wordcloud_file_path = wc_file
        else: logger.info("-> Keine Begriffe für Wortwolke.")
    except Exception as e: logger.error(f"Fehler Wortwolke: {e}", exc_info=True)
    if config.OPENAI_API_KEY:
        logger.info("Generiere OpenAI Empfehlungen...")
        try:
            if isinstance(analysis_summary, dict):
                recos = generate_recommendations(analysis_summary=analysis_summary, query=query, reference_text=reference_text, related_questions=related_questions)
                if recos and "Fehler" not in recos and "Übersprungen" not in recos: logger.info("-> OpenAI Empfehlungen generiert."); recommendations = recos
                else: err_msg = recos if recos else "Keine Empf."; logger.warning(f"-> Problem OpenAI: {err_msg}"); recommendations = f"Hinweis/Fehler: {err_msg}"
            else: err_msg = "Ungültige Analysedaten."; logger.error(err_msg); recommendations = f"Fehler: {err_msg}"
        except Exception as e: err_msg = f"Unerw. Fehler OpenAI: {e}"; logger.exception(err_msg); recommendations = f"Fehler: {err_msg}"
    else: logger.info("Überspringe OpenAI (kein Key)."); recommendations = "Übersprungen (kein API-Schlüssel)."
    return recommendations, wordcloud_file_path

def _save_results(
    output_format: str, output_base_path: str, query: str, language: str, num_results_requested: int,
    num_valid_urls: int, analysis_options: Dict, tfidf_df: Optional[pd.DataFrame], analysis_summary: Dict,
    related_questions: List[str], failed_urls: List, recommendations: Optional[str],
    wordcloud_file_path: Optional[str], timestamp: str, use_cache: bool, reference_file: Optional[str]
) -> Dict[str, str]:
    logger.info(f"Speichere Ergebnisse '{output_format}' unter: {output_base_path}*"); output_files: Dict[str, str] = {}
    if output_format in ["csv", "all"] and tfidf_df is not None and not tfidf_df.empty:
        try:
            tfidf_file = f"{output_base_path}_tfidf.csv"; tfidf_df.to_csv(tfidf_file, index=False, encoding='utf-8-sig')
            logger.info(f"-> CSV gespeichert: {os.path.basename(tfidf_file)}"); output_files["tfidf_csv"] = tfidf_file
        except Exception as e: logger.error(f"Fehler Speichern CSV: {e}", exc_info=True)
    elif output_format in ["csv", "all"]: logger.warning("Überspringe CSV (keine Daten).")
    try:
        summary_json_file = f"{output_base_path}_summary.json"; json_data = { "query": query, "language": language, "timestamp": timestamp, "num_results_requested": num_results_requested, "num_results_processed": num_valid_urls, "reference_file_used": os.path.basename(reference_file) if reference_file else "Nein", "cache_used": use_cache, "analysis_options": analysis_options, "analysis_summary": analysis_summary, "related_questions": related_questions, "recommendations": recommendations, "failed_urls": failed_urls, "wordcloud_file": os.path.basename(wordcloud_file_path) if wordcloud_file_path else None }
        with open(summary_json_file, 'w', encoding='utf-8') as f: json.dump(json_data, f, ensure_ascii=False, indent=4)
        if output_format in ["json", "all"]: logger.info(f"-> JSON gespeichert: {os.path.basename(summary_json_file)}")
        output_files["summary_json"] = summary_json_file
    except Exception as e: logger.error(f"Fehler Speichern JSON: {e}", exc_info=True)
    if recommendations and "Fehler" not in recommendations and "Übersprungen" not in recommendations:
         try:
             reco_file = f"{output_base_path}_recommendations.txt";
             with open(reco_file, 'w', encoding='utf-8') as f: f.write(f"Empfehlungen für: {query}\n{timestamp}\n{'='*30}\n\n{recommendations}")
             logger.info(f"-> TXT gespeichert: {os.path.basename(reco_file)}"); output_files["recommendations_txt"] = reco_file
         except Exception as e: logger.error(f"Fehler Speichern TXT: {e}", exc_info=True)
    if output_format in ["html", "all"]:
        if HTML_TEMPLATE:
            try:
                summary_data = analysis_summary if isinstance(analysis_summary, dict) else {}; sentiment_score = summary_data.get("overall_sentiment"); overall_sentiment_str = f"{sentiment_score:.2f}" if sentiment_score is not None else "N/A"
                render_data = { "query": query, "timestamp": timestamp, "language": language, "num_urls_processed": num_valid_urls, "num_urls_failed": len(failed_urls), "use_cache": use_cache, "reference_file_used": os.path.basename(reference_file) if reference_file else "Nein", "include_ner": analysis_options.get("ner", False), "include_clustering": analysis_options.get("cluster", False), "include_sentiment": analysis_options.get("sentiment", False), "overall_top_terms_with_scores": summary_data.get("overall_top_terms_with_scores", []), "top_terms_by_url": summary_data.get("top_terms_by_url", {}), "missing_terms": summary_data.get("missing_terms", []), "overall_entities": summary_data.get("overall_entities", {}), "clusters": summary_data.get("clusters", {}), "sentiment_by_url": summary_data.get("sentiment_by_url", {}), "overall_sentiment": overall_sentiment_str, "related_questions": related_questions, "recommendations": recommendations, "failed_urls": failed_urls, "wordcloud_file": os.path.basename(wordcloud_file_path) if wordcloud_file_path else None }
                html_content = HTML_TEMPLATE.render(**render_data); html_file = f"{output_base_path}_report.html"
                with open(html_file, 'w', encoding='utf-8') as f: f.write(html_content)
                logger.info(f"-> HTML Report gespeichert: {os.path.basename(html_file)}"); output_files["report_html"] = html_file
            except Exception as e: logger.exception("Fehler Erstellen HTML Report")
        else: logger.warning("Überspringe HTML Report (Template fehlt).")
    return output_files

# --- Hauptanalysefunktion (Orchestrierung) ---
def run_analysis(
    query: str, language: str = "de", num_results: int = 10,
    output_prefix: Optional[str] = None, reference_file: Optional[str] = None,
    use_cache: bool = True, include_ner: bool = False, include_clustering: bool = False,
    include_sentiment: bool = False, max_workers: int = 5, output_format: str = "all"
) -> Dict[str, Any]:
    start_time = time.time(); timestamp = time.strftime('%Y%m%d-%H%M%S')
    logger.info("-" * 50); logger.info(f"Starte Analyse für: '{query}' (Sprache: {language}, Zeit: {timestamp})")
    logger.info(f"Parameter: Num Results={num_results}, Workers={max_workers}, Cache={'an' if use_cache else 'aus'}, Format={output_format}")
    analysis_options = {"ner": include_ner, "cluster": include_clustering, "sentiment": include_sentiment}
    # DEBUG LOG: Zeige die empfangenen Optionen
    logger.debug(f"Analyse-Optionen für diesen Lauf: {analysis_options}")
    logger.info(f"Optionen aktiviert: {', '.join(f'{k}={v}' for k, v in analysis_options.items() if v)}")
    logger.info("-" * 50)

    nlp = _setup_analysis(language)
    if not nlp: return {"success": False, "error": f"Spacy-Modell '{language}' nicht geladen.", "query": query, "language": language}

    texts, valid_urls, failed_urls, related_questions = _fetch_data(query, num_results, language, use_cache, max_workers)
    if not texts:
        err_msg = "; ".join([f"{url}: {reason}" for url, reason in failed_urls]) if failed_urls else "Keine Texte/SERPs."
        logger.error(f"Keine Texte zur Analyse verfügbar. Fehler: {err_msg}")
        return {"success": False, "error": f"Keine Texte zur Analyse verfügbar. Details: {err_msg}", "query": query, "language": language, "failed_urls": failed_urls}

    reference_text = _load_reference_text(reference_file)
    tfidf_df, analysis_summary = _perform_core_analysis(texts, valid_urls, nlp, reference_text, include_ner, include_clustering, include_sentiment)

    # DEBUG LOG: Gib die Keys des Summarys nach der Kernanalyse aus
    logger.debug(f"Keys im analysis_summary nach _perform_core_analysis: {analysis_summary.keys() if isinstance(analysis_summary, dict) else 'Kein Dict'}")

    if analysis_summary is None or analysis_summary.get("error"):
        error = analysis_summary.get("error", "Unbek. Fehler Kernanalyse.") if analysis_summary else "Unbek. Fehler Kernanalyse."
        logger.error(f"Kernanalyse fehlgeschlagen: {error}")
        return {"success": False, "error": error, "query": query, "language": language, "failed_urls": failed_urls}

    output_prefix_sanitized = sanitize_filename(output_prefix or query)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_base_path = os.path.join(config.OUTPUT_DIR, f"{output_prefix_sanitized}_{timestamp}")

    recommendations, wordcloud_file_path = _generate_additional_outputs(analysis_summary, query, reference_text, related_questions, output_base_path)

    output_files = _save_results(
        output_format=output_format, output_base_path=output_base_path, query=query, language=language,
        num_results_requested=num_results, num_valid_urls=len(valid_urls), analysis_options=analysis_options,
        tfidf_df=tfidf_df, analysis_summary=analysis_summary, related_questions=related_questions,
        failed_urls=failed_urls, recommendations=recommendations, wordcloud_file_path=wordcloud_file_path,
        timestamp=timestamp, use_cache=use_cache, reference_file=reference_file
    )

    end_time = time.time(); duration = end_time - start_time
    logger.info("-" * 50); logger.info(f"Analyse für '{query}' abgeschlossen! Dauer: {duration:.2f} Sek."); logger.info(f"Ergebnisse gespeichert: {output_base_path}*"); logger.info("-" * 50)

    result_dict = {
        "success": True, "query": query, "language": language, "output_files": output_files,
        "tfidf_dataframe": tfidf_df.to_dict('records') if tfidf_df is not None and not tfidf_df.empty else [],
        "analysis_summary": analysis_summary, "related_questions": related_questions,
        "recommendations": recommendations, "failed_urls": failed_urls, "wordcloud_file_path": wordcloud_file_path,
        "duration_seconds": duration
    }
    return result_dict