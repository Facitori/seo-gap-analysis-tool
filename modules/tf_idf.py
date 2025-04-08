# SEO-GAP-ANALYSIS/modules/tf_idf.py
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import re
import sys
import os
from collections import Counter
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__) # Logger für dieses Modul

try: import config
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

def load_spacy_model(model_name: str) -> Optional[spacy.language.Language]:
    try: nlp = spacy.load(model_name); logger.info(f"-> Spacy-Modell '{model_name}' geladen."); return nlp
    except OSError: logger.error(f"Spacy-Modell '{model_name}' nicht gefunden."); return None
    except Exception as e: logger.exception(f"Fehler Laden Spacy-Modell '{model_name}'"); return None

def preprocess_text(text: str, nlp: spacy.language.Language) -> str:
    if not text or not nlp: return ""
    text = re.sub(r'https?://\S+', ' ', text); text = re.sub(r'\d+', ' ', text)
    text = text.replace('"', ' ').replace("'", " ").replace("-", " ")
    text = re.sub(r'\s+', ' ', text).strip(); doc = nlp(text); tokens = []
    for token in doc:
        if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and not token.is_stop and
                not token.is_punct and token.lemma_ not in ['-pron-'] and len(token.lemma_) > 2):
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)

def extract_entities(text: str, nlp: spacy.language.Language) -> List[Tuple[str, str, int]]:
    if not text or not nlp: return []
    if "ner" not in nlp.pipe_names: logger.warning("NER-Pipe nicht aktiv."); return []
    try: doc = nlp(text)
    except Exception as e: logger.error(f"Fehler Spacy-Verarbeitung NER: {e}", exc_info=True); return []
    entity_counter = Counter(); entity_labels = {}; allowed_labels = {"PERSON", "ORG", "GPE", "LOC"}
    for ent in doc.ents:
        if ent.label_ in allowed_labels:
            entity_text = ent.text.strip()
            if entity_text and len(entity_text) > 2 and not entity_text.isdigit():
                 entity_text_normalized = re.sub(r'\s+', ' ', entity_text)
                 entity_counter[entity_text_normalized] += 1
                 entity_labels.setdefault(entity_text_normalized, ent.label_)
    sorted_entities = entity_counter.most_common()
    result = [(entity, entity_labels[entity], count) for entity, count in sorted_entities if entity in entity_labels]
    return result

def perform_keyword_clustering(tfidf_matrix, feature_names, num_clusters: int = 5) -> Dict[int, List[str]]:
    if tfidf_matrix.shape[0] < num_clusters:
        logger.warning(f"Docs ({tfidf_matrix.shape[0]}) < Cluster ({num_clusters}). Reduziere."); num_clusters = max(1, tfidf_matrix.shape[0])
    if num_clusters <= 0: logger.warning("Anzahl Cluster <= 0."); return {}
    if tfidf_matrix.shape[1] == 0: logger.warning("Keine Features für Clustering."); return {}
    try: kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto'); kmeans.fit(tfidf_matrix)
    except Exception as e: logger.error(f"Fehler K-Means: {e}", exc_info=True); return {}
    clusters: Dict[int, List[str]] = {}
    if hasattr(kmeans, 'cluster_centers_'):
         order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
         for i in range(num_clusters):
             cluster_terms = []
             for ind in order_centroids[i, :10]:
                 if ind < len(feature_names): cluster_terms.append(feature_names[ind])
             if cluster_terms: clusters[i] = cluster_terms
    else: logger.warning("Cluster-Zentren nicht verfügbar."); return {}
    return {k: v for k, v in clusters.items() if v}

def perform_sentiment_analysis(texts: List[str]) -> Tuple[Dict[int, float], float]:
    sentiment_by_index: Dict[int, float] = {}; total_score = 0.0; valid_texts_count = 0
    for i, text in enumerate(texts):
        if text and len(text) > 10:
            try: blob = TextBlob(text); score = blob.sentiment.polarity; sentiment_by_index[i] = score; total_score += score; valid_texts_count += 1
            except Exception as e: logger.error(f"Fehler Sentiment Text Index {i}: {e}"); sentiment_by_index[i] = 0.0
        else: sentiment_by_index[i] = 0.0
    overall_sentiment = total_score / valid_texts_count if valid_texts_count > 0 else 0.0
    return sentiment_by_index, overall_sentiment

def perform_tf_idf_analysis(texts: List[str], urls: List[str], nlp: spacy.language.Language,
                           reference_text: Optional[str] = None, include_ner: bool = False,
                           include_clustering: bool = False, include_sentiment: bool = False
                           ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    if not nlp: return None, {"error": "Spacy Modell nicht geladen."}
    logger.info("-> Starte Textvorverarbeitung...")
    preprocessed_texts = [preprocess_text(text, nlp) for text in texts]
    valid_indices = [i for i, txt in enumerate(preprocessed_texts) if txt and len(txt.split()) > 1]
    if not valid_indices: return None, {"error": "Keine verwertbaren Texte nach Vorverarbeitung."}
    preprocessed_texts_filtered = [preprocessed_texts[i] for i in valid_indices]
    urls_filtered = [urls[i] for i in valid_indices]; original_texts_filtered = [texts[i] for i in valid_indices]
    logger.info(f"-> {len(preprocessed_texts_filtered)} Texte analysiert (von {len(texts)}).")
    logger.info("-> Berechne TF-IDF...")
    tfidf_matrix = None; feature_names = []
    try:
        vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=2)
        tfidf_matrix = vectorizer.fit_transform(preprocessed_texts_filtered)
        feature_names = vectorizer.get_feature_names_out()
        if tfidf_matrix.shape[1] == 0: logger.warning("TF-IDF: Keine Features gefunden."); return pd.DataFrame({'url': urls_filtered}), {"error": "Keine TF-IDF Features."}
    except Exception as e: logger.error(f"Fehler TF-IDF Vektorisierung: {e}", exc_info=True); return None, {"error": f"Fehler TF-IDF: {e}"}
    try:
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=urls_filtered)
        tfidf_df.reset_index(inplace=True); tfidf_df.rename(columns={'index': 'url'}, inplace=True)
    except Exception as e: logger.error(f"Fehler TF-IDF DataFrame: {e}", exc_info=True); return pd.DataFrame({'url': urls_filtered}), {"error": f"Fehler TF-IDF DF: {e}"}
    logger.info("-> Ermittle Top-Begriffe...")
    top_terms_by_url: Dict[str, List[str]] = {}; all_scores: Dict[str, List[float]] = {}
    for i, url in enumerate(urls_filtered):
        scores = tfidf_matrix[i].toarray().flatten(); sorted_idx = np.argsort(scores)[::-1]; url_top_terms = []
        for idx in sorted_idx[:15]:
             term = feature_names[idx]; score = scores[idx]
             if score > 0.01: url_top_terms.append(term); all_scores.setdefault(term, []).append(score)
        top_terms_by_url[url] = url_top_terms
    overall_avg_scores = {term: np.mean(scores) for term, scores in all_scores.items()}
    sorted_overall_terms = sorted(overall_avg_scores.items(), key=lambda item: item[1], reverse=True)
    overall_top_terms_with_scores = sorted_overall_terms[:50]
    missing_terms: List[str] = []
    if reference_text:
        logger.info("-> Vergleiche mit Referenztext...")
        try:
            preprocessed_reference = preprocess_text(reference_text, nlp); ref_tokens_set = set(preprocessed_reference.split())
            logger.debug(f"Ref Tokens: {ref_tokens_set}"); logger.debug(f"Top Terms: {[t for t,s in overall_top_terms_with_scores]}")
            for term, score in overall_top_terms_with_scores:
                 if term not in preprocessed_reference:
                     parts = term.split(); all_parts_present = all(p in ref_tokens_set for p in parts)
                     if not all_parts_present: missing_terms.append(term); logger.debug(f"Begriff '{term}' fehlt in Ref.")
        except Exception as e: logger.warning(f"Fehler Vergleich Ref-Text: {e}", exc_info=True)

    analysis_summary = { "overall_top_terms_with_scores": overall_top_terms_with_scores, "top_terms_by_url": top_terms_by_url, "missing_terms": missing_terms }

    if include_ner:
        logger.info("-> Führe NER durch...") # DEBUG LOG
        entities_by_url: Dict[str, List[Tuple[str, str, int]]] = {}; overall_entity_counter = Counter(); overall_entity_labels = {}
        combined_ner_labels = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "GPE/LOC", "LOC": "GPE/LOC"}
        for i, text in enumerate(original_texts_filtered):
            url = urls_filtered[i]
            try:
                entities = extract_entities(text, nlp); entities_by_url[url] = entities
                for entity, label, count in entities:
                    if label in combined_ner_labels:
                        combined_label = combined_ner_labels[label]; key = (entity.lower(), combined_label)
                        overall_entity_counter[key] += count; overall_entity_labels.setdefault(key, combined_label)
            except Exception as e: logger.error(f"Fehler NER URL {url}: {e}", exc_info=True)
        aggregated_entities: Dict[str, List[Tuple[str, int]]] = {}; temp_agg: Dict[str, Counter] = {lbl: Counter() for lbl in set(combined_ner_labels.values())}
        lower_to_original_case = {};
        for url_entities in entities_by_url.values():
            for entity_text, _, _ in url_entities: lower_to_original_case.setdefault(entity_text.lower(), entity_text)
        for (ent_lower, lbl), count in overall_entity_counter.items():
            original_case_entity = lower_to_original_case.get(ent_lower, ent_lower); temp_agg[lbl][original_case_entity] += count
        for lbl, counter in temp_agg.items():
            if counter: aggregated_entities[lbl] = counter.most_common(15)
        # DEBUG LOG
        logger.debug(f"NER Ergebnis (overall_entities): {aggregated_entities}")
        analysis_summary["overall_entities"] = aggregated_entities
        # analysis_summary["entities_by_url"] = entities_by_url # Optional hinzufügen

    if include_clustering:
        logger.info("-> Führe Clustering durch...") # DEBUG LOG
        n_clusters = min(5, max(2, len(urls_filtered) // 2)) if len(urls_filtered) > 3 else max(1, len(urls_filtered))
        clusters = perform_keyword_clustering(tfidf_matrix, feature_names, num_clusters=n_clusters)
        # DEBUG LOG
        logger.debug(f"Clustering Ergebnis (clusters): {clusters}")
        analysis_summary["clusters"] = clusters

    if include_sentiment:
        logger.info("-> Führe Sentiment Analyse durch...") # DEBUG LOG
        sentiment_by_index, overall_sentiment = perform_sentiment_analysis(original_texts_filtered)
        sentiment_by_url = {urls_filtered[i]: score for i, score in sentiment_by_index.items() if i < len(urls_filtered)}
        # DEBUG LOG
        logger.debug(f"Sentiment Ergebnis (sentiment_by_url): {sentiment_by_url}")
        logger.debug(f"Sentiment Ergebnis (overall_sentiment): {overall_sentiment}")
        analysis_summary["sentiment_by_url"] = sentiment_by_url
        analysis_summary["overall_sentiment"] = overall_sentiment

    return tfidf_df, analysis_summary