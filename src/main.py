###
## Main entry point for the URL Migration Tool.
## .\amb\Scripts\activate
###
import os
import sys
from pathlib import Path
import multiprocessing as mp
# Update the import statement to include highlight_score_columns

# Add the project root directory to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f'project_root: {project_root}')

# librerie locali
from src.mig_tool import MigTool
from src.logger import setup_logger, log_error
from src.matching_algorithms import MatchingAlgorithms
from src.excel_formatter import highlight_matching_urls, highlight_score_columns

from config.config import HTTP_CONFIG

# selezione lingua per spacy 
def get_language_selection():
    """
    Richiede all'utente di selezionare la lingua per l'elaborazione.
    Accetta le lingue IT, EN, DE, FR o EXIT per uscire dal programma.
    Returns:
        str: La lingua selezionata in formato minuscolo
    """
    while True:
        language = input("Scegli la lingua (IT/EN/DE/FR) o EXIT per uscire: ").strip().lower()
        if language in ['it', 'en', 'de', 'fr']:
            return language
        elif language in ['exit', 'quit', 'close']:
            print("Chiusura del programma...")
            sys.exit(0)
        print("Scelta non valida. Riprova.")

def get_parallel_jobs():
    """
    Richiede all'utente di inserire il numero di processi paralleli da utilizzare.
    Accetta -1 per utilizzare tutti i core disponibili o un numero positivo.
    Returns:
        int: Numero di processi paralleli da utilizzare
    """
    while True:
        try:
            n_jobs = input("Inserisci il numero di processi paralleli (-1 per tutti i core disponibili): ").strip()
            n_jobs = int(n_jobs)
            if n_jobs == -1 or n_jobs > 0:
                return n_jobs
            print("Inserisci un numero positivo o -1")
        except ValueError:
            print("Inserisci un numero valido")

def get_parallel_threads():
    """
    Richiede all'utente di inserire il numero di thread paralleli per la verifica degli URL.
    Accetta valori da 1 a 10.
    Returns:
        int: Numero di thread paralleli per la verifica URL
    """
    while True:
        try:
            threads = input("Inserisci il numero di thread paralleli per la verifica URL (1-10): ").strip()
            threads = int(threads)
            if 1 <= threads <= 10:
                return threads
            print("Inserisci un numero tra 1 e 10")
        except ValueError:
            print("Inserisci un numero valido")

def get_use_playwright():
    """
    Richiede all'utente se utilizzare Playwright per la verifica degli URL.
    Returns:
        bool: True per usare Playwright, False per usare httpx
    """
    while True:
        choice = input("Utilizzare Playwright per la verifica degli URL? (s/n): ").strip().lower()
        if choice in ['s', 'si', 'sì', 'y', 'yes']:
            return True
        elif choice in ['n', 'no', 'not']:
            return False
        print("Scelta non valida. Rispondi 's' o 'n'.")

def get_playwright_http2_option():
    """
    Richiede all'utente se disabilitare il protocollo HTTP/2 in Playwright.
    
    Returns:
        bool: True per disabilitare HTTP/2, False per usare le impostazioni predefinite
    """
    while True:
        choice = input("Disabilitare HTTP/2 per evitare errori di protocollo? (s/n): ").strip().lower()
        if choice in ['s', 'si', 'sì', 'y', 'yes']:
            return True
        elif choice in ['n', 'no', 'not']:
            return False
        print("Scelta non valida. Rispondi 's' o 'n'.")

def get_follow_redirects_option():
    """
    Richiede all'utente se Playwright deve seguire i redirect e restituire lo status code finale.
    
    Returns:
        bool: True per seguire i redirect e restituire lo status code finale,
              False per non seguire i redirect e restituire lo status code iniziale
    """
    while True:
        choice = input("Seguire i redirect e restituire lo status code finale? (s/n): ").strip().lower()
        if choice in ['s', 'si', 'sì', 'y', 'yes']:
            return True
        elif choice in ['n', 'no', 'not']:
            return False
        print("Scelta non valida. Rispondi 's' o 'n'.")

# Main entry point
def main():
    """
    Punto di ingresso principale del programma.
    Inizializza il logger, richiede le configurazioni all'utente e avvia il MigTool.
    Gestisce le eccezioni e registra eventuali errori fatali.
    """
    try:
        # Initialize logger
        logger = setup_logger()
        
        # Get language selection first
        language = get_language_selection()
        
        # Get number of parallel jobs
        n_jobs = get_parallel_jobs()
        
        # Get number of parallel threads for URL verification
        parallel_threads = get_parallel_threads()
        HTTP_CONFIG['parallel_threads'] = parallel_threads
        
        # Ask if HTTP/2 should be disabled in Playwright
        disable_http2 = get_playwright_http2_option()
        
        # Ask if redirects should be followed
        follow_redirects = get_follow_redirects_option()
        
        # Create and run MigTool with language selection and parallel jobs
        mig_tool = MigTool(language, n_jobs=n_jobs)
        
        # Set the configuration flags in the url_handler
        mig_tool.url_handler.disable_http2 = disable_http2
        mig_tool.url_handler.follow_redirects = follow_redirects
        
        # Run the tool and get the path to the generated Excel file
        excel_path = mig_tool.run()
        
        ### format excel
        
        # Apply formatting to the generated Excel file
        if excel_path and excel_path.exists():
            logger.info(f"Applying formatting to {excel_path}")
            highlight_matching_urls(excel_path, logger)
            highlight_score_columns(excel_path, logger)  
            logger.info("Formatting complete!")
        else:
            logger.warning("No Excel file was generated. Skipping Excel formatting.")
        
    except Exception as e:
        log_error(e, logger, "Fatal error in main")
        raise

# Questo costrutto verifica se lo script viene eseguito direttamente (non importato come modulo)
# Quando il file viene eseguito direttamente, __name__ è "__main__"
# Quando il file viene importato come modulo, __name__ è il nome del modulo
if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for Windows compatibility
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    main() 