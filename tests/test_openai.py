# SEO-GAP-ANALYSIS/modules/openai_helper.py
import openai
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, AuthenticationError
from typing import Dict, Any, List, Optional
import sys
import os
import traceback
import logging
# KORREKTUR: RetryError wieder importieren
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, RetryError

logger = logging.getLogger(__name__)

try: import config
except ImportError:
    logger.error("Konnte config nicht importieren."); sys.exit(1)

# --- Retry Konfiguration & Logging Callback ---
def log_retry_openai(retry_state):
    logger.warning(f"OpenAI API Fehler (Versuch {retry_state.attempt_number}): {retry_state.outcome.exception()}. Warte {retry_state.next_action.sleep:.2f}s...")

def _should_retry_openai_exception(e: BaseException) -> bool:
    """Prüft, ob eine OpenAI Exception einen Retry auslösen soll."""
    if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(e, APIStatusError) and not isinstance(e, AuthenticationError):
         status_code = getattr(e, 'status_code', None)
         if status_code is not None and status_code >= 500:
             return True
    return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=retry_if_exception(_should_retry_openai_exception),
    before_sleep=log_retry_openai
)
def _call_openai_api(client: OpenAI, model: str, messages: List[Dict], temperature: float, max_tokens: int):
    """Macht den eigentlichen API Call (wird von tenacity wiederholt)."""
    logger.debug(f"Sende Anfrage an OpenAI Model {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response

# --- Funktion generate_recommendations ---
def generate_recommendations(
    analysis_summary: Dict[str, Any],
    query: str,
    reference_text: Optional[str] = None,
    related_questions: Optional[List[str]] = None
) -> str:
    """Generiert SEO-Empfehlungen für die Artikelerstellung via OpenAI."""
    if not config.OPENAI_API_KEY:
        logger.warning("Kein OpenAI API-Schlüssel konfiguriert.")
        return "Übersprungen (kein API-Schlüssel)."

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)

        # --- Daten für den Prompt extrahieren ---
        top_terms_scores = analysis_summary.get("overall_top_terms_with_scores", [])
        top_terms = ", ".join([f"{t} ({s:.2f})" for t, s in top_terms_scores[:20]]) if top_terms_scores else "Keine gefunden."
        missing = ", ".join(analysis_summary.get("missing_terms", [])) or "Keine (oder kein Referenztext vorhanden)."
        clusters = analysis_summary.get("clusters", {}); cluster_info = "\n".join([f"  - Cluster {c}: {', '.join(t[:8])}{'...' if len(t)>8 else ''}" for c, t in clusters.items()]) if clusters else "Keine Cluster gefunden."
        entities = analysis_summary.get("overall_entities", {}); entity_info = ""
        if entities:
            for label, ents in entities.items():
                if ents: entity_info += f"  - {label}: {', '.join([f'{e} ({c}x)' for e, c in ents[:5]])}\n"
        if not entity_info: entity_info = "Keine relevanten Entitäten gefunden.\n"
        sentiment = analysis_summary.get("overall_sentiment"); sentiment_desc = f"{sentiment:.2f}" if sentiment is not None else "N/A."
        ref_snippet = f"'{reference_text[:200]}...'" if reference_text and len(reference_text)>200 else (f"'{reference_text}'" if reference_text else 'Keiner vorhanden.')
        paa_list_str = "\n".join([f"  - {q}" for q in related_questions]) if related_questions else "Keine 'People Also Ask'-Fragen gefunden."

        # --- Prompt ---
        system_prompt = """Du bist ein exzellenter SEO-Redakteur und Content-Stratege...""" # Gekürzt
        user_prompt = f"""
**Ziel:** Erstellung eines herausragenden Artikels zum Keyword "{query}".
**Basis:** SEO-Gap-Analyse der Top-Wettbewerber und häufig gestellte Fragen (PAA).
**Analysedaten der Wettbewerber & SERP:**
1.  **Wichtigste Begriffe (TF-IDF):** {top_terms}
2.  **Potenziell fehlende Begriffe (im Vgl. zum Referenztext):** {missing}
3.  **Relevante Entitäten (NER):**
{entity_info}
4.  **Thematische Keyword-Cluster:**
{cluster_info}
5.  **Häufig gestellte Fragen (People Also Ask):**
{paa_list_str}
6.  **Durchschnittliches Sentiment der Wettbewerber:** {sentiment_desc}
7.  **Referenztext (Auszug, falls vorhanden):** {ref_snippet}
**Deine Aufgabe:**
Entwickle basierend auf **allen** oben genannten Daten ein **detailliertes Konzept und konkrete Handlungsempfehlungen...**
**Liefere insbesondere Vorschläge für:**
1.  **Artikel-Struktur / Gliederung:** ...
2.  **Inhaltliche Schwerpunkte & Content Gaps:** ...
3.  **Titel & Meta-Description:** ...
4.  **Einleitung & Kernaussagen:** ...
5.  **(Optional) Tonalität & Stil:** ...
**Strukturiere deine Empfehlungen klar und übersichtlich...**
""" # Gekürzt
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # --- API Aufruf ---
        response = _call_openai_api(client=client, model=config.OPENAI_MODEL, messages=messages, temperature=config.OPENAI_TEMPERATURE, max_tokens=config.OPENAI_MAX_TOKENS)

        # --- Antwortverarbeitung ---
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content; logger.debug("Antwort von OpenAI erhalten."); return content.strip()
        else: logger.warning(f"Keine gültige Antwort von OpenAI: {response}"); return "Fehler: Ungültige Antwort von OpenAI."

    # --- Fehlerbehandlung ---
    except openai.AuthenticationError as e: logger.error(f"OpenAI Authentifizierungsfehler: {e}"); return "Fehler: OpenAI API-Schlüssel ungültig oder Berechtigung fehlt."
    except APIStatusError as e:
         if 400 <= e.status_code < 500: logger.error(f"OpenAI Client Error (Status {e.status_code}): {e.response}"); return f"Fehler: Ungültige Anfrage an OpenAI (Status: {e.status_code}). Prüfe die Eingabedaten/Prompt."
         else: logger.exception(f"Unerwarteter APIStatusError ({e.status_code}) von OpenAI"); return f"Fehler bei OpenAI-Anfrage (APIStatusError): {str(e)}"
    # KORREKTUR: RetryError hier wieder hinzugefügt
    except (RetryError, RateLimitError, APITimeoutError, APIConnectionError) as e:
        logger.exception("Fehler bei OpenAI-Anfrage nach Retries oder nicht-wiederholbarer Netzwerkfehler")
        final_exception = e.__cause__ if hasattr(e, '__cause__') and e.__cause__ else e
        return f"Fehler bei OpenAI-Verbindung: {str(final_exception)}"
    except Exception as e:
        logger.exception("Unerwarteter Fehler in generate_recommendations")
        final_exception = e.__cause__ if hasattr(e, '__cause__') and e.__cause__ else e
        return f"Fehler bei Empfehlungsgenerierung: {str(final_exception)}"