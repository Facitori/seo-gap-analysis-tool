# SEO-GAP-ANALYSIS/modules/serp_api.py
import os
import requests
import sys
import json
# KORREKTUR/ERGÄNZUNG: Typing für Rückgabewert anpassen
from typing import List, Dict, Any, Optional, TypedDict
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from cache_utils import get_cache_key, get_cache_path, load_from_cache, save_to_cache

# --- Retry Konfiguration (bleibt) ---
RETRY_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)

def log_retry_attempt(retry_state):
    logger.warning(
        f"Netzwerkfehler (Versuch {retry_state.attempt_number}): Erneuter Versuch nach {retry_state.outcome.exception()}. "
        f"Warte {retry_state.next_action.sleep:.2f}s..."
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # Prüft auf Netzwerkfehler ODER 5xx Serverfehler
    retry=retry_if_exception(
        lambda e: isinstance(e, RETRY_EXCEPTIONS) or \
                  (isinstance(e, requests.exceptions.HTTPError) and e.response is not None and e.response.status_code >= 500)
    ),
    before_sleep=log_retry_attempt
)
def _make_serp_api_request(url: str, params: Dict) -> requests.Response:
    """Führt den eigentlichen API-Request durch (wird von tenacity wiederholt)."""
    logger.debug(f"Sende Anfrage an {url} mit Parametern: {params}")
    response = requests.get(url, params=params, timeout=20)
    # raise_for_status prüft auf 4xx/5xx Fehler. Tenacity wiederholt nur bei 5xx (gemäß retry-Bedingung)
    response.raise_for_status()
    return response

# --- NEU: TypedDict für klareren Rückgabewert ---
class SerpResults(TypedDict):
    organic_results: List[Dict[str, Any]]
    related_questions: List[str] # Nur die Fragen als Liste von Strings
    error: Optional[str] # Fehlermeldung, falls etwas schiefgeht

# --- Hauptfunktion angepasst ---
def get_serp_results(
    query: str,
    num_results: int = config.RESULTS_COUNT,
    use_cache: bool = True,
    language: str = config.LANGUAGE
) -> SerpResults: # Rückgabetyp angepasst
    """
    Ruft SERP-Ergebnisse (organisch + PAA-Fragen) ab, verwendet Cache und Retries.
    Gibt ein Dictionary mit 'organic_results', 'related_questions' und 'error' zurück.
    """
    default_return: SerpResults = {"organic_results": [], "related_questions": [], "error": None}
    logger.debug(f"get_serp_results aufgerufen für query='{query}', num={num_results}, lang={language}, cache={use_cache}")

    # Cache-Schlüssel bleibt gleich, aber Inhalt ändert sich
    cache_key = get_cache_key("serp_v2", query, num_results, language) # v2 für neuen Inhalt
    cache_file = get_cache_path("serp_v2", cache_key, extension="json")

    if use_cache:
        cached_data = load_from_cache(cache_file)
        # Prüfen, ob Cache-Daten das erwartete Format haben
        if isinstance(cached_data, dict) and "organic_results" in cached_data and "related_questions" in cached_data:
            logger.info(f"-> SERP-Daten (inkl. PAA) aus Cache für '{query}'.")
            # Füge optional 'error: None' hinzu, falls es im Cache fehlt
            cached_data.setdefault("error", None)
            return cached_data # Type Hint erwartet hier SerpResults
        elif cached_data is not None:
            logger.warning(f"Ungültiges Format im SERP-Cache für '{query}' gefunden. Ignoriere Cache.")

    if not config.SERP_API_KEY:
        logger.error("Kein SERP_API_KEY konfiguriert.")
        default_return["error"] = "SerpApi API Key fehlt."
        return default_return

    # Parameter für SerpApi
    params = {
        "q": query,
        "num": num_results * 2, # Frage etwas mehr an, um genug organische zu bekommen
        "api_key": config.SERP_API_KEY,
        "engine": "google",
        "hl": language,
        "gl": language.upper() if len(language) == 2 else 'DE', # Ländercode
        "lr": f"lang_{language}" # Sprachrestriktion
        # Weitere Parameter könnten hier hinzugefügt werden, z.B. für Ort
    }

    try:
        logger.info("-> Führe API-Anfrage an SerpApi durch (mit Retries)...")
        response = _make_serp_api_request(config.SERP_API_URL, params)
        data = response.json()

        # Extrahiere organische Ergebnisse
        organic_results = []
        if "organic_results" in data:
            count = 0
            for item in data["organic_results"]:
                # Filtere Ergebnisse ohne Link oder Titel oder nicht-http(s) URLs
                link = item.get("link")
                title = item.get("title")
                if link and title and link.startswith(('http://', 'https://')):
                    organic_results.append({"title": title, "url": link})
                    count += 1
                    if count >= num_results: # Begrenze auf die gewünschte Anzahl
                        break
        else:
            logger.warning("Keine 'organic_results' in SerpApi-Antwort.")

        # Extrahiere "People Also Ask" Fragen
        related_questions = []
        # WICHTIG: Der Schlüssel kann variieren! Prüfe deine API-Antwort.
        # Mögliche Schlüssel: 'related_questions', 'people_also_ask'
        paa_key = None
        if 'related_questions' in data and isinstance(data['related_questions'], list):
            paa_key = 'related_questions'
        elif 'people_also_ask' in data and isinstance(data['people_also_ask'], list):
             paa_key = 'people_also_ask'
        # Weitere Variationen prüfen, falls nötig

        if paa_key:
            for item in data[paa_key]:
                if isinstance(item, dict) and item.get("question"):
                    related_questions.append(item["question"])
            logger.info(f"-> {len(related_questions)} PAA-Fragen gefunden.")
        else:
            logger.info("-> Keine PAA-Fragen ('related_questions' oder 'people_also_ask') in SerpApi-Antwort gefunden.")
            # Optional: Logge die Top-Level-Keys der Antwort zum Debuggen
            # logger.debug(f"SerpApi Antwort-Keys: {list(data.keys())}")


        logger.info(f"-> {len(organic_results)} organische Ergebnisse von API für '{query}'.")
        result_data: SerpResults = {
            "organic_results": organic_results,
            "related_questions": related_questions,
            "error": None
        }
        if use_cache:
            save_to_cache(result_data, cache_file) # Speichere das gesamte Dictionary
        return result_data

    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
        err_msg = f"Fehler beim Abrufen der Suchergebnisse nach Retries. Status: {status_code}. Fehler: {e}"
        logger.error(err_msg, exc_info=True)
        default_return["error"] = err_msg
        return default_return
    except json.JSONDecodeError as e:
        err_msg = f"Ungültige JSON-Antwort von SerpApi für '{query}': {e}"
        logger.error(err_msg)
        default_return["error"] = err_msg
        return default_return
    except Exception as e:
        err_msg = f"Unerwarteter Fehler beim Verarbeiten der SerpApi-Antwort für '{query}'"
        logger.exception(err_msg) # Logge den Traceback
        default_return["error"] = f"{err_msg}: {e}"
        return default_return