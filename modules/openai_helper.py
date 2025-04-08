# SEO-GAP-ANALYSIS/modules/openai_helper.py
# ... (Imports, Retry-Config, _call_openai_api bleiben gleich) ...
import openai
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from typing import Dict, Any, List, Optional
import sys
import os
import traceback
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger(__name__)

try: import config
except ImportError:
    logger.error("Konnte config nicht importieren."); sys.exit(1) # Beenden, wenn config fehlt

# --- Retry Konfiguration & Logging Callback (bleibt) ---
def log_retry_openai(retry_state):
    logger.warning(f"OpenAI API Fehler (Versuch {retry_state.attempt_number}): {retry_state.outcome.exception()}. Warte {retry_state.next_action.sleep:.2f}s...")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=retry_if_exception(
        lambda e: isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)) or \
                  (isinstance(e, APIStatusError) and e.status_code >= 500)
    ),
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


# ANGEPASST: Nimmt jetzt optional related_questions entgegen
def generate_recommendations(
    analysis_summary: Dict[str, Any],
    query: str,
    reference_text: Optional[str] = None,
    related_questions: Optional[List[str]] = None # NEU: Optionaler Parameter
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

        # NEU: Formatiere PAA-Fragen für den Prompt
        paa_list_str = ""
        if related_questions:
            paa_list_str = "\n".join([f"  - {q}" for q in related_questions])
        else:
            paa_list_str = "Keine 'People Also Ask'-Fragen gefunden."

        # --- Optimierter Prompt für Artikelerstellung (mit PAA) ---
        system_prompt = """Du bist ein exzellenter SEO-Redakteur und Content-Stratege. Deine Aufgabe ist es, basierend auf einer SEO-Wettbewerbsanalyse detaillierte Empfehlungen und konkrete Vorschläge für die **Erstellung eines neuen, umfassenden Artikels** zu liefern, der das Potenzial hat, besser zu ranken als die analysierten Wettbewerber. Sprich Deutsch, sei kreativ, präzise und praxisorientiert."""

        user_prompt = f"""
**Ziel:** Erstellung eines herausragenden Artikels zum Keyword "{query}".

**Basis:** SEO-Gap-Analyse der Top-Wettbewerber und häufig gestellte Fragen (PAA).

**Analysedaten der Wettbewerber & SERP:**
1.  **Wichtigste Begriffe (TF-IDF):** {top_terms}
    *   *(Kernbegriffe, die im Artikel vorkommen sollten)*
2.  **Potenziell fehlende Begriffe (im Vgl. zum Referenztext):** {missing}
    *   *(Diese Lücken sollten im neuen Artikel geschlossen werden)*
3.  **Relevante Entitäten (NER):**
{entity_info}
    *   *(Wichtige Personen, Orte, Organisationen, Produkte etc., die Relevanz signalisieren)*
4.  **Thematische Keyword-Cluster:**
{cluster_info}
    *   *(Deuten auf wichtige Unterthemen und Struktur für den Artikel hin)*
5.  **Häufig gestellte Fragen (People Also Ask):**
{paa_list_str}
    *   *(Direkte Nutzerfragen, die im Artikel beantwortet werden sollten, idealerweise in einem FAQ-Abschnitt)*
6.  **Durchschnittliches Sentiment der Wettbewerber:** {sentiment_desc}
    *   *(Gibt Orientierung für die Tonalität)*
7.  **Referenztext (Auszug, falls vorhanden):** {ref_snippet}
    *   *(Kann als Ausgangspunkt oder zum Vergleich dienen)*

**Deine Aufgabe:**
Entwickle basierend auf **allen** oben genannten Daten ein **detailliertes Konzept und konkrete Handlungsempfehlungen für die Erstellung eines neuen Artikels** zum Thema "{query}".

**Liefere insbesondere Vorschläge für:**

1.  **Artikel-Struktur / Gliederung:**
    *   Schlage eine logische Gliederung mit Haupt- (H2) und Unterüberschriften (H3) vor.
    *   Nutze die **Keyword-Cluster** und die **Häufig gestellten Fragen (PAA)**, um sinnvolle Abschnitte (inkl. eines FAQ-Abschnitts) zu definieren.
    *   Berücksichtige die **Top-Begriffe** und **Entitäten** bei der Themenfindung für die Abschnitte.

2.  **Inhaltliche Schwerpunkte & Content Gaps:**
    *   Welche **fehlenden Begriffe** müssen unbedingt thematisiert werden? In welchen Abschnitten?
    *   Welche der **Häufig gestellten Fragen (PAA)** sollten direkt beantwortet werden?
    *   Welche **Entitäten** sollten prominent erwähnt oder näher erläutert werden?
    *   Gibt es **Themen aus den Clustern**, die bei Wettbewerbern vorkommen und im neuen Artikel vertieft werden sollten, um umfassender zu sein?

3.  **Titel & Meta-Description:**
    *   Schlage 2-3 attraktive und SEO-optimierte **Titelvorschläge** (ca. 50-60 Zeichen) vor, die das Keyword "{query}" und wichtige Begriffe/Fragen aufgreifen.
    *   Formuliere einen Vorschlag für eine **Meta-Description** (ca. 150-160 Zeichen), die zum Klicken anregt und relevante Keywords/Entitäten/Fragen aufgreift.

4.  **Einleitung & Kernaussagen:**
    *   Skizziere eine mögliche Einleitung, die den Nutzer abholt und das Thema umreißt.
    *   Was sollten die zentralen Botschaften oder Alleinstellungsmerkmale des Artikels sein?

5.  **(Optional) Tonalität & Stil:**
    *   Gib eine kurze Empfehlung zur Tonalität basierend auf dem **Sentiment** der Wettbewerber und dem Thema.

**Strukturiere deine Empfehlungen klar und übersichtlich (z.B. mit Markdown).** Sei spezifisch und gib nicht nur Keywords wieder, sondern erkläre, *wie* sie im Artikelkontext verwendet werden können.
"""
        # --- API Aufruf ---
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = _call_openai_api(client=client, model=config.OPENAI_MODEL, messages=messages, temperature=config.OPENAI_TEMPERATURE, max_tokens=config.OPENAI_MAX_TOKENS) # Evtl. max_tokens erhöhen

        # --- Antwortverarbeitung ---
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content
            logger.debug("Antwort von OpenAI erfolgreich erhalten.")
            return content.strip()
        else:
            logger.warning(f"Keine gültige Antwort von OpenAI erhalten: {response}"); return "Fehler: Ungültige oder leere Antwort von OpenAI."

    # --- Fehlerbehandlung ---
    except openai.AuthenticationError as e: logger.error(f"OpenAI Authentifizierungsfehler: {e}"); return "Fehler: OpenAI API-Schlüssel ungültig oder Berechtigung fehlt."
    except APIStatusError as e:
         if 400 <= e.status_code < 500: logger.error(f"OpenAI Client Error (Status {e.status_code}): {e.response}"); return f"Fehler: Ungültige Anfrage an OpenAI (Status: {e.status_code}). Prüfe die Eingabedaten/Prompt."
         else: logger.exception("Unerwarteter APIStatusError von OpenAI"); return f"Fehler bei OpenAI-Anfrage (APIStatusError): {str(e)}"
    except Exception as e:
        logger.exception("Fehler in generate_recommendations nach Retries oder anderer Fehler")
        final_exception = e.__cause__ if hasattr(e, '__cause__') and e.__cause__ else e; return f"Fehler bei Empfehlungsgenerierung: {str(final_exception)}"