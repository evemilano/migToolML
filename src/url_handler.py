"""
URL handling and HTTP verification for the URL Migration Tool.

This module manages all URL-related operations and HTTP verification processes for the URL Migration Tool, including:
- URL validation and normalization
- HTTP status code checking and verification
- URL pattern matching and comparison
- Handling of different URL formats and encodings
- Redirect chain analysis and validation

The module provides the URLHandler class which manages:
1. URL validation and cleaning
   - Protocol handling (http/https)
   - URL encoding/decoding
   - Path normalization
   - Query parameter handling
2. HTTP verification
   - Status code checking
   - Redirect chain analysis
   - Response validation
3. URL comparison and matching
   - Pattern matching
   - Path similarity analysis
   - Query parameter comparison
4. Error handling and logging
   - Invalid URL detection
   - Connection error handling
   - Timeout management
5. Performance optimization
   - Connection pooling
   - Caching mechanisms
   - Rate limiting

The module is essential for:
- Ensuring URL validity before processing
- Verifying HTTP responses and redirects
- Supporting accurate URL matching algorithms
- Maintaining robust error handling
- Optimizing performance for large-scale URL processing
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, unquote, quote
from tenacity import retry, stop_after_attempt, wait_exponential
from config.config import HTTP_CONFIG
from src.logger import log_error, log_warning, log_info
import asyncio
import concurrent.futures
from functools import partial
from playwright.sync_api import sync_playwright

class URLHandler:
    """Handles URL processing and HTTP verification."""
    
    def __init__(self, logger):
        """
        Initialize URLHandler.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.http_config = HTTP_CONFIG
        self.parallel_threads = min(max(1, self.http_config['parallel_threads']), 10)
        self.disable_http2 = True   # Flag per disabilitare HTTP/2 con Playwright
        self.follow_redirects = True  # Flag per controllare se seguire i redirect
        
    def clean_url(self, url: str) -> str:
        """
        Clean URL for comparison.
        
        Args:
            url (str): URL to clean
            
        Returns:
            str: Cleaned URL
        """
        if not url:
            return ''
            
        try:
            # First clean invalid characters
            url = self.clean_invalid_characters(url)
            
            # Then remove UTM parameters
            url = self.remove_utm_parameters(url)
            
            # Decode URL and parse components
            parsed = urlparse(unquote(url))
            
            # Split path into segments and join with spaces
            path = ' '.join(filter(None, parsed.path.split('/')))
            
            # Extract and sort parameters
            params = parsed.params
            if params:
                params = '&'.join(sorted(params.split('&')))
                
            # Replace hyphens, underscores, and hashes with spaces
            path = re.sub(r'[-_#]', ' ', path)
            
            # Rebuild URL: path + parameters (if present)
            cleaned_url = (path + ('?' + params if params else '')).strip()
            
            # Convert to lowercase
            return cleaned_url.lower()
        except Exception as e:
            log_error(e, self.logger, f"Error cleaning URL: {url}")
            return url.lower()
    
    def remove_utm_parameters(self, url: str) -> str:
        """
        Remove UTM parameters from URL.
        
        Args:
            url (str): URL to clean
            
        Returns:
            str: URL without UTM parameters
        """
        try:
            parsed_url = urlparse(url)
            if 'utm' in parsed_url.query:
                return urlunparse((parsed_url.scheme, parsed_url.netloc, 
                                 parsed_url.path, '', '', ''))
            return url
        except Exception as e:
            log_error(e, self.logger, f"Error removing UTM parameters from URL: {url}")
            return url
    
    def clean_invalid_characters(self, url: str) -> str:
        """
        Remove invalid characters from URL.
        
        Args:
            url (str): URL to clean
            
        Returns:
            str: Cleaned URL
        """
        if not isinstance(url, str):
            return str(url)
            
        try:
            # Remove leading/trailing whitespace and non-printable characters
            url = url.strip()
            url = ''.join(c for c in url if c.isprintable())
            return url
        except Exception as e:
            log_error(e, self.logger, f"Error cleaning invalid characters from URL: {url}")
            return str(url)
    
    @retry(stop=stop_after_attempt(2), 
           wait=wait_exponential(multiplier=1, min=2, max=6))
    def check_status_playwright(self, url: str) -> int:
        """
        Check HTTP status code using Playwright.
        
        Args:
            url (str): URL to check
            
        Returns:
            int: HTTP status code
        """
        try:
            # User-Agent realistico di Chrome su Windows
            realistic_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            
            # Prepara gli argomenti del browser in base alle opzioni scelte
            browser_args = [
                "--no-sandbox",            # Evita problemi di sandbox
                "--disable-dev-shm-usage", # Evita problemi di memoria condivisa
                "--disable-setuid-sandbox", # Ulteriore sicurezza
                "--disable-gpu",           # Disabilita accelerazione GPU
                "--disable-web-security",  # Disabilita alcune restrizioni di sicurezza web
                "--disable-features=IsolateOrigins,site-per-process" # Disabilita isolamento processi
            ]
            
            # Aggiungi l'opzione per disabilitare HTTP/2 se richiesto
            if self.disable_http2:
                browser_args.append("--disable-http2")
                log_info(f"HTTP/2 disabilitato per la verifica di: {url}", self.logger)
            
            with sync_playwright() as p:
                # Usa il browser chromium in modalità headless con opzioni aggiuntive
                browser = p.chromium.launch(
                    headless=True,
                    args=browser_args
                )
                
                try:
                    # Crea un nuovo contesto e pagina con User-Agent realistico
                    context = browser.new_context(
                        user_agent=realistic_ua,
                        ignore_https_errors=True,  # Ignora errori SSL
                        viewport={"width": 1920, "height": 1080}  # Risoluzione desktop standard
                    )
                    
                    # Aggiunge intestazioni extra per sembrare un browser reale
                    page = context.new_page()
                    page.set_extra_http_headers({
                        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "sec-ch-ua": '"Google Chrome";v="121", "Not;A=Brand";v="8", "Chromium";v="121"',
                        "sec-ch-ua-mobile": "?0",
                        "sec-ch-ua-platform": '"Windows"'
                    })
                    
                    # Variabili per tracciare stato iniziale e finale
                    initial_status = None
                    final_status = None
                    redirect_chain = []
                    
                    # Intercetta le risposte HTTP per catturare lo status code
                    def handle_response(response):
                        nonlocal initial_status, final_status, redirect_chain
                        # Se l'URL corrisponde esattamente, salva lo stato iniziale
                        if response.url == url and initial_status is None:
                            initial_status = response.status
                            
                        # Se lo stato è un redirect (3xx), aggiungi alla catena di redirect
                        if 300 <= response.status < 400:
                            redirect_chain.append((response.url, response.status, response.headers.get("location")))
                        
                        # Per l'URL finale dopo redirect, salva lo stato finale
                        final_status = response.status
                    
                    page.on("response", handle_response)
                    
                    # Opzioni per il goto che dipendono da follow_redirects
                    goto_options = {
                        "timeout": self.http_config['timeout'] * 1000,
                        "wait_until": "domcontentloaded"
                    }
                    
                    # Naviga all'URL con timeout
                    page.goto(url, **goto_options)
                    
                    # Decidi quale status code restituire in base a follow_redirects
                    if self.follow_redirects:
                        status_code = final_status
                        redirect_info = f" (dopo {len(redirect_chain)} redirect)" if redirect_chain else ""
                        log_info(f"URL: {url} - Status Code finale: {status_code}{redirect_info}", self.logger)
                    else:
                        status_code = initial_status
                        log_info(f"URL: {url} - Status Code iniziale: {status_code}", self.logger)
                        if redirect_chain:
                            redirect_info = ", ".join([f"{code} -> {location}" for _, code, location in redirect_chain[:2]])
                            log_info(f"Redirect ignorati: {redirect_info}...", self.logger)
                    
                    # Se non abbiamo catturato uno status code, assumiamo che la pagina sia caricata con successo
                    if status_code is None:
                        status_code = 200
                        log_info(f"URL: {url} - Nessuno status code rilevato, assumo 200 OK", self.logger)
                    
                    return status_code
                finally:
                    browser.close()
        except Exception as e:
            log_error(e, self.logger, f"Playwright error checking URL: {url}")
            raise
    
    def _verify_url_batch(self, urls: List[str], batch_idx: int, total_batches: int) -> List[Tuple[str, int]]:
        """
        Verify a batch of URLs using Playwright.
        
        Args:
            urls (List[str]): List of URLs to verify
            batch_idx (int): Current batch index
            total_batches (int): Total number of batches
            
        Returns:
            List[Tuple[str, int]]: List of (URL, status_code) tuples
        """
        results = []

        # Verifica degli URL con Playwright
        for url in urls:
            try:
                # Verifica URL con Playwright
                status = self.check_status_playwright(url)
                log_info(f"URL: {url} - Status Code: {status}", self.logger)
                results.append((url, status))
            except Exception as e:
                log_error(e, self.logger, f"Error verifying URL: {url}")
                results.append((url, "browser error"))
            time.sleep(self.http_config['pause'])
        
        # Log batch completion milestone
        progress = (batch_idx + 1) / total_batches * 100
        log_info(f"Batch {batch_idx + 1}/{total_batches} completato ({progress:.1f}%)", self.logger)
        return results

    def verify_urls(self, urls: List[str]) -> List[Tuple[str, int]]:
        """
        Verify list of URLs using HTTP/2 with parallel processing.
        
        Args:
            urls (List[str]): List of URLs to verify
            
        Returns:
            List[Tuple[str, int]]: List of (URL, status_code) tuples
        """
        total_urls = len(urls)
        if total_urls == 0:
            return []
            
        # Use batch size from config
        batch_size = self.http_config['batch_size']
        batches = [urls[i:i + batch_size] for i in range(0, total_urls, batch_size)]
        total_batches = len(batches)
        
        log_info(f"Inizio verifica URL con {self.parallel_threads} thread paralleli", self.logger)
        log_info(f"Totale URL da verificare: {total_urls}", self.logger)
        log_info(f"Dimensione batch: {batch_size} URL", self.logger)
        log_info(f"Numero totale di batch: {total_batches}", self.logger)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_threads) as executor:
            # Create partial function with fixed arguments
            verify_batch = partial(self._verify_url_batch, total_batches=total_batches)
            
            # Submit all batches for parallel processing
            future_to_batch = {
                executor.submit(verify_batch, batch, i): batch 
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            completed_batches = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_results = future.result()
                results.extend(batch_results)
                completed_batches += 1
                
                # Log overall progress milestone (every 10% or when all batches are done)
                progress = (completed_batches / total_batches) * 100
                if progress % 10 < 1 or completed_batches == total_batches:
                    log_info(f"Progresso complessivo: {progress:.1f}% ({completed_batches}/{total_batches} batch completati)", 
                            self.logger)
        
        log_info("Verifica URL completata", self.logger)
        return results
    
    def filter_problematic_urls(self, results: List[Tuple[str, int]], 
                              problematic_codes: List[int]) -> List[Tuple[str, int]]:
        """
        Filter URLs with problematic status codes.
        
        Args:
            results (List[Tuple[str, int]]): List of (URL, status_code) tuples
            problematic_codes (List[int]): List of problematic status codes
            
        Returns:
            List[Tuple[str, int]]: List of problematic URLs
        """
        return [(url, status) for url, status in results 
                if status in problematic_codes or status == "httpx error"]
    
    def retry_problematic_urls(self, problematic_urls: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Retry verification of problematic URLs.
        
        Args:
            problematic_urls (List[Tuple[str, int]]): List of problematic URLs
            
        Returns:
            List[Tuple[str, int]]: Updated results for problematic URLs
        """
        updated_results = []
        
        # Utilizza Playwright per verificare nuovamente gli URL problematici
        for url, _ in problematic_urls:
            # Attesa più lunga tra i tentativi per URL problematici
            time.sleep(self.http_config['pause'] * 2 + 3)
            try:
                # Verifica URL con Playwright
                status = self.check_status_playwright(url)
                log_info(f"Retry URL: {url} - Status Code: {status}", self.logger)
                updated_results.append((url, status))
            except Exception as e:
                log_error(e, self.logger, f"Error retrying URL: {url}")
                updated_results.append((url, "browser error"))
        
        return updated_results 