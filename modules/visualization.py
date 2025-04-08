# SEO-GAP-ANALYSIS/modules/visualization.py
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import logging

# Logger für dieses Modul
logger = logging.getLogger(__name__)

# Setze das Backend, bevor pyplot importiert wird (oder hier, falls nicht anderswo geschehen)
# 'Agg' ist gut für Server ohne GUI
try:
    matplotlib.use('Agg')
except ImportError:
    logger.warning("Matplotlib Backend 'Agg' konnte nicht gesetzt werden. Probleme bei GUI-loser Ausführung möglich.")

def generate_wordcloud(terms, output_file):
    """Generiert eine Wortwolke und speichert sie."""
    if not terms:
        logger.warning("Keine Begriffe für Wortwolke übergeben.")
        return

    fig = None # Initialisiere fig außerhalb des try-Blocks
    try:
        logger.debug(f"Generiere Wortwolke für {len(terms)} Begriffe...")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(terms))

        # Erstelle die Figur und Achse
        # Wichtig: speichere das Figure-Objekt
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        # Speichern der Figur
        plt.savefig(output_file)

        logger.info(f"Wortwolke gespeichert unter: {output_file}")
    except Exception as e:
        logger.error(f"Fehler beim Generieren der Wortwolke für {output_file}: {e}", exc_info=True)
    finally:
        # Stelle sicher, dass die Figur geschlossen wird, wenn sie erstellt wurde
        # Dies gibt Speicher frei, auch wenn ein Fehler aufgetreten ist.
        if fig is not None:
             try:
                 # Schließe die spezifische Figur, die erstellt wurde
                 plt.close(fig)
                 logger.debug(f"Matplotlib Figur für {output_file} geschlossen.")
             except Exception as close_err:
                 # Logge Fehler beim Schließen, aber fahre fort
                 logger.warning(f"Fehler beim Schließen der Matplotlib Figur für {output_file}: {close_err}")