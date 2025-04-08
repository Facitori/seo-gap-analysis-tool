# SEO-GAP-ANALYSIS/modules/extractor.py
import os
import requests
import trafilatura
import sys
import tenacity
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
    RetryError, retry_if_exception
)
from typing import Optional, Tuple, Dict
import traceback
import logging
import requests.exceptions

logger = logging.getLogger(__name__)

try:
    import config # Importiere das config-Modul
    from cache_utils import get_cache_key, get_cache_path, load_from_cache, save_to_cache
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    from cache_utils import get_cache_key, get_cache_path, load_from_cache, save_to_cache

# KORREKTUR: Verwende den Wert direkt aus dem config-Modul
MIN_TEXT_LENGTH = config.MIN_EXTRACT_LENGTH

# Retry Konfiguration (bleibt gleich)
RETRY_EXCEPTIONS_EXTRACTOR = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)

def log_retry_extractor(retry_state):
    exception = retry_state.outcome.exception() if retry_state.outcome else "Unknown Exception"
    logger.warning(
        f"Netzwerkfehler beim Extrahieren (Versuch {retry_state.attempt_number}): {exception.__class__.__name__}. "
        f"Warte {retry_state.next_action.sleep:.2f}s..."
    )

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception(
        lambda e: isinstance(e, RETRY_EXCEPTIONS_EXTRACTOR) or \
                  (isinstance(e, requests.exceptions.HTTPError) and e.response is not None and e.response.status_code >= 500)
    ),
    before_sleep=log_retry_extractor
)
def _fetch_url_content(url: str, headers: Dict) -> requests.Response:
    logger.debug(f"-> Versuche Download von {url}...")
    response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
    response.raise_for_status()
    return response

def extract_text_from_url(url: str, use_cache: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """Extrahiert Textinhalt von URL mit Trafilatura und Retries für Download."""
    logger.debug(f"extract_text_from_url aufgerufen für '{url}', cache={use_cache}")
    cache_key = get_cache_key("text_v2", url)
    cache_file = get_cache_path("text_v2", cache_key, extension="json")

    if use_cache:
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
             if isinstance(cached_data, (list, tuple)) and len(cached_data) == 2:
                 logger.debug(f"Cache hit für {url}")
                 return cached_data[0], cached_data[1]
             else:
                 logger.warning(f"Ungültiges Cache-Format für {url} gefunden, ignoriere Cache.")

    downloaded_content = None; error_msg = None; text_content = None; response = None
    try:
        headers = { "User-Agent": "...", "Accept": "...", "Accept-Language": "...", "Referer": "..." } # Gekürzt
        response = _fetch_url_content(url, headers)
        content_type = response.headers.get('Content-Type', '').lower()
        if 'html' not in content_type:
            error_msg = f"Inhaltstyp ist kein HTML ({content_type})"
            logger.warning(f"{error_msg} für {url}")
            if use_cache: save_to_cache([None, error_msg], cache_file)
            return None, error_msg

        downloaded_content = response.content
        if not downloaded_content:
             error_msg = "Kein Inhalt heruntergeladen."
             logger.warning(error_msg + f" für {url}")
             if use_cache: save_to_cache([None, error_msg], cache_file)
             return None, error_msg

        logger.debug(f"-> Extrahiere Text mit Trafilatura für {url}...")
        try:
            text_content = trafilatura.extract(downloaded_content, include_comments=False, include_tables=False, include_formatting=False)
        except Exception as trafila_error:
            error_msg = f"Trafilatura Fehler: {trafila_error}"
            logger.error(f"{error_msg} für {url}", exc_info=True)
            if use_cache: save_to_cache([None, error_msg], cache_file)
            return None, error_msg

        if not text_content:
            error_msg = "Trafilatura konnte keinen Hauptinhalt extrahieren."
            logger.warning(error_msg + f" für {url}")
            if use_cache: save_to_cache([None, error_msg], cache_file)
            return None, error_msg

        # Verwende MIN_TEXT_LENGTH aus config
        if len(text_content) < MIN_TEXT_LENGTH:
            error_msg = f"Extrahierter Text zu kurz ({len(text_content)}/{MIN_TEXT_LENGTH})"
            logger.warning(error_msg + f" für {url}")
            if use_cache: save_to_cache([None, error_msg], cache_file)
            return None, error_msg

        logger.debug(f"-> Erfolgreich Text ({len(text_content)} Zeichen) extrahiert von {url}")
        if use_cache: save_to_cache([text_content, None], cache_file)
        return text_content, None

    except requests.exceptions.RequestException as e:
        response_obj = getattr(e, 'response', None)
        status_code = response_obj.status_code if response_obj is not None else 'N/A'
        if isinstance(e, requests.exceptions.HTTPError) and 400 <= status_code < 500:
            error_msg = f"HTTP Client Fehler {status_code} ({getattr(response_obj, 'reason', 'N/A')})"
            logger.warning(f"{error_msg} beim Abruf von {url}")
        else:
            error_msg = f"Netzwerkfehler (RequestException): {e.__class__.__name__}"
            logger.error(f"Fehler bei Extraktion von {url}: {error_msg}", exc_info=True)
        response = response_obj
        if use_cache: save_to_cache([None, error_msg], cache_file)
        return None, error_msg
    except RetryError as e:
        original_exception = e.__cause__ if e.__cause__ else e
        error_msg = f"Netzwerkfehler nach Retries: {original_exception.__class__.__name__}"
        logger.error(f"Fehler bei Extraktion von {url} nach allen Retries: {error_msg}", exc_info=e)
        if use_cache: save_to_cache([None, error_msg], cache_file)
        return None, error_msg
    except Exception as e:
        error_msg = f"Unerwarteter Extraktionsfehler: {e}"
        logger.exception(f"Schwerwiegender Fehler bei Extraktion von {url}")
        if use_cache: save_to_cache([None, error_msg], cache_file)
        return None, error_msg
    finally:
        if response is not None:
            try:
                response.close()
                logger.debug(f"Response-Verbindung für {url} geschlossen.")
            except Exception as close_err:
                logger.warning(f"Fehler beim Schließen der Response für {url}: {close_err}")