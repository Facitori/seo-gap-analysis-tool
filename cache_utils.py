# SEO-GAP-ANALYSIS/cache_utils.py
import os
import hashlib
import time
import json
from typing import Any
import config
import logging # NEU
import shutil # Für clear_all_cache

# Logger für dieses Modul
logger = logging.getLogger(__name__)

def get_cache_key(*args) -> str:
    """Erzeugt einen MD5-Cache-Schlüssel."""
    key_string = "_".join(map(str, args)).lower()
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def get_cache_path(cache_type: str, key: str, extension: str = "json") -> str:
    """Erstellt den vollständigen Pfad zu einer Cache-Datei."""
    filename = f"{cache_type}_{key}.{extension}"
    return os.path.join(config.CACHE_DIR, filename)

def is_cache_valid(cache_file: str) -> bool:
    """Überprüft, ob die Cache-Datei existiert und noch gültig ist."""
    if not os.path.exists(cache_file): return False
    try:
        file_age = time.time() - os.path.getmtime(cache_file)
        is_valid = file_age < config.MAX_CACHE_AGE_SECONDS
        logger.debug(f"Cache check für {os.path.basename(cache_file)}: Alter={file_age:.0f}s, Gültig={is_valid}") # DEBUG Level
        return is_valid
    except OSError as e: logger.warning(f"Fehler beim Prüfen des Cache-Alters für {cache_file}: {e}"); return False

def load_from_cache(cache_file: str) -> Any | None:
    """Lädt Daten aus einer Cache-Datei."""
    if not is_cache_valid(cache_file): return None
    try:
        logger.debug(f"Lade aus Cache: {os.path.basename(cache_file)}") # Log statt print
        with open(cache_file, 'r', encoding='utf-8') as f:
            if cache_file.endswith(".json"): return json.load(f)
            else: return f.read()
    except FileNotFoundError: logger.warning(f"Cache-Datei nicht gefunden (während Lesen): {cache_file}"); return None # Log statt print
    except json.JSONDecodeError as e: logger.error(f"Fehler beim Dekodieren JSON-Cache {cache_file}: {e}"); return None # Log statt print
    except Exception as e: logger.error(f"Allg. Fehler beim Lesen Cache {cache_file}: {e}", exc_info=True); return None # Log statt print

def save_to_cache(data: Any, cache_file: str):
    """Speichert Daten in einer Cache-Datei."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            if cache_file.endswith(".json"): json.dump(data, f, ensure_ascii=False, indent=4)
            else: f.write(str(data))
        logger.debug(f"Im Cache gespeichert: {os.path.basename(cache_file)}") # Log statt print
    except Exception as e: logger.error(f"Fehler beim Speichern im Cache {cache_file}: {e}", exc_info=True) # Log statt print

def clear_cache_for_query(query: str, num_results: int, language: str):
    """Löscht spezifische Cache-Dateien für eine Abfrage."""
    serp_key = get_cache_key("serp", query, num_results, language); serp_file = get_cache_path("serp", serp_key, extension="json")
    if os.path.exists(serp_file):
        try: os.remove(serp_file); logger.info(f"SERP Cache gelöscht: {os.path.basename(serp_file)}") # Log statt print
        except OSError as e: logger.error(f"Fehler beim Löschen SERP Cache {serp_file}: {e}") # Log statt print
    # Hinweis zum Text-Cache bleibt relevant
    logger.info("Hinweis: Text-Caches werden nicht automatisch gelöscht. Ggf. --clear-cache verwenden.") # Log statt print


def clear_all_cache():
    """Löscht den gesamten Inhalt des Cache-Verzeichnisses."""
    if os.path.exists(config.CACHE_DIR):
        try:
            # Sicherer: Inhalt löschen statt Verzeichnis selbst, falls Rechte fehlen
            for filename in os.listdir(config.CACHE_DIR):
                file_path = os.path.join(config.CACHE_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e: logger.error(f'Fehler beim Löschen von {file_path} im Cache. Grund: {e}') # Log statt print
            logger.info(f"Cache-Verzeichnis '{config.CACHE_DIR}' wurde geleert.") # Log statt print
        except Exception as e: logger.error(f"Fehler beim Leeren des Cache-Verzeichnisses '{config.CACHE_DIR}': {e}", exc_info=True) # Log statt print
    else: logger.warning("Cache-Verzeichnis existiert nicht.") # Log statt print