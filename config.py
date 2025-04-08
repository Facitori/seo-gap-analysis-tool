# SEO-GAP-ANALYSIS/config.py
import os
import json
from dotenv import load_dotenv
import logging # Logging hinzufügen

logger = logging.getLogger(__name__) # Logger für config.py

# Lade Umgebungsvariablen aus .env-Datei zuerst
load_dotenv()

# --- Kernkonfiguration (aus .env oder Standardwerte) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# --- OpenAI Konfiguration ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.7))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 1000))

# --- SerpApi Konfiguration ---
SERP_API_URL = os.getenv("SERP_API_URL", "https://serpapi.com/search")
RESULTS_COUNT = int(os.getenv("RESULTS_COUNT", 10)) # Standardanzahl Ergebnisse

# --- Sprach- und NLP-Konfiguration ---
LANGUAGE = os.getenv("LANGUAGE", "de")  # Standardsprache
SPACY_MODEL_MAP = {
    "de": "de_core_news_sm",
    "en": "en_core_web_sm"
    # Füge hier weitere Sprachen und Modelle hinzu
}
# Modell basierend auf Sprache oder expliziter Angabe in .env
SPACY_MODEL = os.getenv("SPACY_MODEL", SPACY_MODEL_MAP.get(LANGUAGE, "de_core_news_sm"))

# --- Verzeichnisse und Caching ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
MAX_CACHE_AGE_SECONDS = int(os.getenv("MAX_CACHE_AGE_SECONDS", 7 * 24 * 60 * 60)) # 7 Tage Standard

# --- NEU: Extraktionskonfiguration ---
# Mindestlänge des extrahierten Textes, damit er als gültig betrachtet wird
MIN_EXTRACT_LENGTH = int(os.getenv("MIN_EXTRACT_LENGTH", 150))

# --- Sicherstellen, dass Verzeichnisse existieren ---
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
except OSError as e:
    logger.error(f"Fehler beim Erstellen der Verzeichnisse {OUTPUT_DIR} oder {CACHE_DIR}: {e}")
    # Hier könnte man überlegen, ob die Anwendung beendet werden soll

def load_config_from_json(config_path: str = "config.json"):
    """
    Lädt zusätzliche Konfigurationseinstellungen aus einer JSON-Datei.
    Überschreibt Standardwerte oder .env-Werte für NICHT-SENSITIVE Daten.
    API-Schlüssel und sensible Daten werden NICHT aus JSON geladen.
    """
    global LANGUAGE, RESULTS_COUNT, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, \
           SERP_API_URL, SPACY_MODEL, OUTPUT_DIR, CACHE_DIR, MAX_CACHE_AGE_SECONDS, \
           SPACY_MODEL_MAP, MIN_EXTRACT_LENGTH # MIN_EXTRACT_LENGTH hinzugefügt

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            logger.info(f"Lade zusätzliche Konfiguration aus {config_path}...") # Log statt print

            # Lade bekannte, nicht-sensitive Schlüssel (überschreibt .env/Defaults)
            LANGUAGE = config_data.get("LANGUAGE", LANGUAGE)
            RESULTS_COUNT = int(config_data.get("RESULTS_COUNT", RESULTS_COUNT)) # Sicherstellen, dass int
            OPENAI_MODEL = config_data.get("OPENAI_MODEL", OPENAI_MODEL)
            OPENAI_TEMPERATURE = float(config_data.get("OPENAI_TEMPERATURE", OPENAI_TEMPERATURE)) # Sicherstellen, dass float
            OPENAI_MAX_TOKENS = int(config_data.get("OPENAI_MAX_TOKENS", OPENAI_MAX_TOKENS)) # Sicherstellen, dass int
            SERP_API_URL = config_data.get("SERP_API_URL", SERP_API_URL)
            # Spacy-Modell basierend auf der (ggf. aus JSON geladenen) Sprache aktualisieren
            SPACY_MODEL = config_data.get("SPACY_MODEL", SPACY_MODEL_MAP.get(LANGUAGE, "de_core_news_sm"))
            OUTPUT_DIR = config_data.get("OUTPUT_DIR", OUTPUT_DIR)
            MAX_CACHE_AGE_SECONDS = int(config_data.get("MAX_CACHE_AGE_SECONDS", MAX_CACHE_AGE_SECONDS)) # Sicherstellen, dass int
            # NEU: MIN_EXTRACT_LENGTH laden
            MIN_EXTRACT_LENGTH = int(config_data.get("MIN_EXTRACT_LENGTH", MIN_EXTRACT_LENGTH)) # Sicherstellen, dass int

            # Cache-Verzeichnis neu berechnen, falls OUTPUT_DIR geändert wurde
            CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

            # Sicherstellen, dass Verzeichnisse existieren, falls geändert
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            os.makedirs(CACHE_DIR, exist_ok=True)

            # Warnung vor unbekannten Schlüsseln oder API-Schlüsseln in JSON
            known_keys = {
                "LANGUAGE", "RESULTS_COUNT", "OPENAI_MODEL", "OPENAI_TEMPERATURE",
                "OPENAI_MAX_TOKENS", "SERP_API_URL", "SPACY_MODEL", "OUTPUT_DIR",
                "MAX_CACHE_AGE_SECONDS", "MIN_EXTRACT_LENGTH" # MIN_EXTRACT_LENGTH hinzugefügt
            }
            for key in config_data:
                if "API_KEY" in key.upper():
                     logger.warning(f"Sicherheitswarnung: API-Schlüssel '{key}' in {config_path} gefunden und ignoriert. API-Keys nur in .env!")
                elif key not in known_keys:
                    logger.warning(f"Unbekannter Konfigurationsschlüssel '{key}' in {config_path}")

        except json.JSONDecodeError as e:
            logger.error(f"Fehler beim Lesen der JSON-Konfigurationsdatei {config_path}: {e}")
        except ValueError as e:
             logger.error(f"Fehler beim Konvertieren eines Wertes aus {config_path}: {e}")
        except Exception as e:
            logger.error(f"Allgemeiner Fehler beim Laden der Konfiguration aus {config_path}: {e}", exc_info=True)
    else:
        logger.info(f"Keine JSON-Konfigurationsdatei '{config_path}' gefunden. Verwende Werte aus .env und Standardwerte.")

def get_spacy_model_for_language(lang_code: str) -> str:
    """Gibt den Namen des Spacy-Modells für den gegebenen Sprachcode zurück."""
    return SPACY_MODEL_MAP.get(lang_code, SPACY_MODEL_MAP.get("de")) # Fallback auf Deutsch