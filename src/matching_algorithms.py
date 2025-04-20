"""
URL matching algorithms for the URL Migration Tool.

This module implements various string matching algorithms to find the best matching URLs
between a set of 404 (broken) URLs and a set of live URLs. It provides multiple matching
strategies to ensure robust URL matching:

1. Vectorized Algorithms:
   - Jaccard Similarity: Measures similarity between sets of characters
   - Tversky Index: Asymmetric similarity measure with configurable parameters

2. Simple Algorithms:
   - Hamming Distance: Modified version that handles different length URLs
   - Ratcliff/Obershelp: Pattern matching algorithm

3. Complex Algorithms:
   - Levenshtein Distance: Measures minimum number of single-character edits
   - Fuzzy String Matching: Token-based similarity using fuzzywuzzy
   - Jaro-Winkler: Similarity measure optimized for short strings

4. Machine Learning Algorithms:
   - spaCy: Natural language processing based similarity
   - BERTopic: Topic modeling based matching
   - TF-IDF Vectorization: Character n-gram based similarity
   - ML: Continuous learning algorithm based on previous matches

Features:
- Parallel processing support for improved performance
- Configurable algorithm selection
- Detailed logging of matching process
- Support for multiple languages (IT, EN, DE, FR)
- Batch processing with optimal size calculation
- Comprehensive similarity scoring and normalization

The module is designed to be part of a larger URL migration tool, helping to identify
the most appropriate redirects for broken URLs to their live counterparts.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
from fuzzywuzzy import fuzz
import Levenshtein as lev
from difflib import SequenceMatcher
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jellyfish import jaro_winkler_similarity
from bertopic import BERTopic
from config.config import SIMILARITY_CONFIG, ALGORITHM_CONFIG
from src.logger import log_error, log_info
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

class MatchingAlgorithms:
    """Handles URL matching using various algorithms."""
    
    def __init__(self, logger, language: str, n_jobs: int = -1):
        """
        Initialize MatchingAlgorithms.
        
        Args:
            logger: Logger instance
            language (str): Selected language ('it' or 'en')
            n_jobs (int): Number of jobs for parallel processing (-1 for all cores)
        """
        self.logger = logger
        self.similarity_config = SIMILARITY_CONFIG
        self.algorithm_config = ALGORITHM_CONFIG
        self.nlp = None
        self.vectorizer = None
        self.live_vectors = None
        self.bertopic_model = None
        self.language = language
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.ml_matcher = None
        
        # Cache for BERTopic embeddings
        self._live_url_topics = None
        self._live_url_probs = None
        self._url_404_topics = {}
        self._url_404_probs = {}
        self._bertopic_fitted = False
        
        # Initialize preprocessing
        self._initialize_preprocessing()
        
    def _initialize_preprocessing(self):
        """Initialize all preprocessing steps and models."""
        if self.algorithm_config['use_spacy']:
            self.nlp = self.load_spacy_model()
            
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 3),
            min_df=2
        )
        
        # Initialize BERTopic model if enabled
        if self.algorithm_config['use_bertopic']:
            try:
                # Initialize BERTopic with reduced dimensions for efficiency
                self.bertopic_model = BERTopic(
                    #embedding_model="all-MiniLM-L6-v2",
                    embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
                    min_topic_size=2,
                    verbose=False
                )
                log_info("BERTopic model initialized", self.logger)
            except Exception as e:
                log_error(e, self.logger, "Failed to initialize BERTopic model")
                self.bertopic_model = None
        
        # Initialize ML matcher if enabled
        if self.algorithm_config['use_ml']:
            try:
                from src.ml import AIMatchingAlgorithm
                self.ml_matcher = AIMatchingAlgorithm(self.logger)
                log_info("ML matching algorithm initialized", self.logger)
            except Exception as e:
                log_error(e, self.logger, "Failed to initialize ML matching algorithm")
                self.ml_matcher = None
        
    def _vectorized_jaccard(self, url: str, live_urls: np.ndarray) -> Tuple[str, float]:
        """Vectorized Jaccard similarity implementation."""
        url_set = set(url)
        live_sets = [set(u) for u in live_urls]
        
        # Vectorized intersection and union
        intersections = np.array([len(url_set.intersection(s)) for s in live_sets])
        unions = np.array([len(url_set.union(s)) for s in live_sets])
        
        scores = np.divide(intersections, unions, where=unions > 0)
        best_idx = np.argmax(scores)
        
        return live_urls[best_idx], scores[best_idx]
    
    def _vectorized_tversky(self, url: str, live_urls: np.ndarray) -> Tuple[str, float]:
        """Vectorized Tversky index implementation."""
        url_set = set(url)
        live_sets = [set(u) for u in live_urls]
        alpha = self.similarity_config['tversky_alpha']
        beta = self.similarity_config['tversky_beta']
        
        # Vectorized calculations
        commons = np.array([len(url_set.intersection(s)) for s in live_sets])
        unique_to_url = np.array([len(url_set - s) for s in live_sets])
        unique_to_live = np.array([len(s - url_set) for s in live_sets])
        
        denominators = commons + alpha * unique_to_url + beta * unique_to_live
        scores = np.divide(commons, denominators, where=denominators > 0)
        best_idx = np.argmax(scores)
        
        return live_urls[best_idx], scores[best_idx]
    
    def _process_batch(self, batch_urls: List[str], live_urls: List[str]) -> List[Dict]:
        """Process a batch of URLs using all algorithms."""
        results = []
        live_urls_array = np.array(live_urls)
        
        # Process BERTopic in batch if enabled
        if self.algorithm_config['use_bertopic'] and self.bertopic_model and not self._bertopic_fitted:
            try:
                self._initialize_bertopic_embeddings(batch_urls, live_urls)
            except Exception as e:
                log_error(e, self.logger, "Error initializing BERTopic embeddings")
        
        for url in batch_urls:
            matches = []
            
            # 1. Vectorized algorithms
            if self.algorithm_config['use_jaccard']:
                match, score = self._vectorized_jaccard(url, live_urls_array)
                matches.append(('jaccard', match, score))
                
            if self.algorithm_config['use_tversky']:
                match, score = self._vectorized_tversky(url, live_urls_array)
                matches.append(('tversky', match, score))
                
            # 2. Simple algorithms
            if self.algorithm_config['use_hamming']:
                match, score = self.hamming_matching(url, live_urls)
                matches.append(('hamming', match, score))
                
            if self.algorithm_config['use_ratcliff']:
                match, score = self.ratcliff_matching(url, live_urls)
                matches.append(('ratcliff', match, score))
                
            # 3. Complex algorithms
            if self.algorithm_config['use_levenshtein']:
                match, score = self.levenshtein_matching(url, live_urls)
                matches.append(('levenshtein', match, score))
                
            if self.algorithm_config['use_fuzzy']:
                match, score = self.fuzzy_matching(url, live_urls)
                matches.append(('fuzzy', match, score))
                
            # Add Jaro-Winkler matching
            if self.algorithm_config['use_jaro_winkler']:
                match, score = self.jaro_winkler_matching(url, live_urls)
                matches.append(('jaro_winkler', match, score))
                
            # Add vector matching
            if self.algorithm_config['use_vector']:
                # Initialize vectorizer if not already done
                if self.live_vectors is None and live_urls:
                    try:
                        # Fit vectorizer on live URLs
                        self.live_vectors = self.vectorizer.fit_transform(live_urls)
                        log_info("Vectorizer initialized and fitted", self.logger)
                    except Exception as e:
                        log_error(e, self.logger, "Error initializing vectorizer")
                
                if self.live_vectors is not None:
                    match, score = self.vector_matching(url, live_urls, self.vectorizer, self.live_vectors)
                    matches.append(('vector', match, score))
                
            # 4. ML algorithms
            if self.algorithm_config['use_bertopic'] and self.bertopic_model and self._bertopic_fitted:
                match, score = self.bertopic_matching(url, live_urls)
                matches.append(('bertopic', match, score))
                
            if self.algorithm_config['use_ml'] and self.ml_matcher and self.ml_matcher.model_initialized:
                match, score = self.ml_matching(url, live_urls)
                matches.append(('ML', match, score))
                
            results.append({
                'url_404': url,
                'matches': matches
            })
            
        return results
    
    def _initialize_bertopic_embeddings(self, batch_urls: List[str], live_urls: List[str]):
        """
        Initialize BERTopic model and pre-compute embeddings for both 404 and live URLs.
        This significantly improves performance by avoiding recomputing embeddings.
        
        Args:
            batch_urls (List[str]): A batch of 404 URLs
            live_urls (List[str]): List of live URLs
        """
        if not self.bertopic_model or not live_urls or self._bertopic_fitted:
            return
            
        try:
            log_info("Pre-computing BERTopic embeddings for all URLs...", self.logger)
            
            # Combine all URLs for model fitting
            all_urls = live_urls.copy()
            all_urls.extend(batch_urls)
            
            # Fit model once on all URLs
            self.bertopic_model.fit(all_urls)
            self._bertopic_fitted = True
            log_info("BERTopic model fitted with all URLs", self.logger)
            
            # Pre-compute embeddings for live URLs (done once)
            log_info("Pre-computing topics for live URLs...", self.logger)
            self._live_url_topics, self._live_url_probs = self.bertopic_model.transform(live_urls)
            log_info(f"Completed topic modeling for {len(live_urls)} live URLs", self.logger)
            
            # Pre-compute embeddings for 404 URLs in this batch
            log_info(f"Pre-computing topics for {len(batch_urls)} 404 URLs...", self.logger)
            batch_topics, batch_probs = self.bertopic_model.transform(batch_urls)
            
            # Store pre-computed embeddings in cache
            for i, url in enumerate(batch_urls):
                if i < len(batch_topics):
                    self._url_404_topics[url] = batch_topics[i]
                    self._url_404_probs[url] = batch_probs[i]
            
            log_info("BERTopic embeddings pre-computation complete", self.logger)
            
        except Exception as e:
            log_error(e, self.logger, "Error pre-computing BERTopic embeddings")
            self._bertopic_fitted = False
    
    def find_matches(self, df_404: pd.DataFrame, df_live: pd.DataFrame) -> List[Dict]:
        """
        Find matches for all 404 URLs using parallel processing.
        
        Args:
            df_404 (pd.DataFrame): DataFrame containing 404 URLs
            df_live (pd.DataFrame): DataFrame containing live URLs
            
        Returns:
            List[Dict]: List of matches for each 404 URL
        """
        live_urls = df_live['URL'].tolist()
        urls_404 = df_404['URL'].tolist()
        total_urls = len(urls_404)
        
        log_info(f"\n{'='*50}", self.logger)
        log_info("INIZIO PROCESSO DI MATCHING", self.logger)
        log_info(f"{'='*50}", self.logger)
        log_info(f"Totale URL 404 da processare: {total_urls}", self.logger)
        log_info(f"Totale URL live disponibili: {len(live_urls)}", self.logger)
        log_info(f"Numero di processi paralleli: {self.n_jobs}", self.logger)
        
        # Calculate optimal batch size based on number of URLs
        batch_size = max(1, total_urls // (self.n_jobs * 4))
        batches = [urls_404[i:i + batch_size] for i in range(0, total_urls, batch_size)]
        total_batches = len(batches)
        
        log_info(f"\nConfigurazione batch:", self.logger)
        log_info(f"- Dimensione batch: {batch_size} URL", self.logger)
        log_info(f"- Numero totale di batch: {total_batches}", self.logger)
        log_info(f"- URL per processo: {batch_size * self.n_jobs}", self.logger)
        
        log_info("\nAlgoritmi attivi:", self.logger)
        for algo, enabled in self.algorithm_config.items():
            if enabled:
                log_info(f"- {algo.replace('use_', '')}", self.logger)
        
        log_info(f"\n{'='*50}", self.logger)
        
        # Process batches in parallel
        process_batch = partial(self._process_batch, live_urls=live_urls)
        # verbose is used to set the verbosity level of the parallel processing
        # verbose value is between 0 and 100
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(process_batch)(batch) for batch in batches
        )
        
        # Flatten results
        results = [item for sublist in results for item in sublist]
        
        # Process spaCy matching separately (not in parallel)
        if self.algorithm_config['use_spacy'] and self.nlp:
            log_info(f"\n{'='*50}", self.logger)
            log_info("INIZIO MATCHING CON SPACY", self.logger)
            log_info(f"{'='*50}", self.logger)
            log_info(f"Processamento sequenziale di {len(results)} URL", self.logger)
            
            for idx, result in enumerate(results, 1):
                url = result['url_404']
                match, score = self.spacy_matching(url, live_urls)
                if match:
                    result['matches'].append(('spacy', match, score))
                if idx % 10 == 0:  # Log ogni 10 URL
                    progress = (idx / len(results)) * 100
                    log_info(f"Progresso spaCy: {progress:.1f}% ({idx}/{len(results)} URL processati)", self.logger)
            
            log_info("Matching spaCy completato", self.logger)
        
        # Log riepilogo finale
        log_info(f"\n{'='*50}", self.logger)
        log_info("RIEPILOGO FINALE", self.logger)
        log_info(f"{'='*50}", self.logger)
        log_info(f"Totale URL processati: {len(results)}", self.logger)
        log_info(f"Totale match trovati: {sum(1 for r in results if r['matches'])}", self.logger)
        
        # Log distribuzione match per algoritmo
        log_info("\nDistribuzione match per algoritmo:", self.logger)
        algo_counts = {}
        for result in results:
            for algo, _, _ in result['matches']:
                algo_counts[algo] = algo_counts.get(algo, 0) + 1
        
        for algo, count in algo_counts.items():
            log_info(f"- {algo}: {count} match", self.logger)
        
        log_info(f"{'='*50}\n", self.logger)
        return results
    
    def load_spacy_model(self):
        """
        Load spaCy model based on selected language.
        
        Returns:
            spacy.lang: Loaded spaCy model or None if loading fails
        """
        try:
            if self.language == "it":
                nlp = spacy.load("it_core_news_lg", disable=["tagger", "parser", "ner"])
                self.logger.info("Modello italiano caricato")
            elif self.language == "de":
                nlp = spacy.load("de_core_news_lg", disable=["tagger", "parser", "ner"])
                self.logger.info("Deutsches Modell geladen")
            elif self.language == "fr":
                nlp = spacy.load("fr_core_news_lg", disable=["tagger", "parser", "ner"])
                self.logger.info("Modèle français chargé")
            else:
                nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner"])
                self.logger.info("English model loaded")
            return nlp
        except OSError as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            print(f"Errore nel caricamento del modello. Assicurati di aver installato il modello corretto.")
            print("Per installare i modelli, esegui:")
            print("python -m spacy download it_core_news_lg")
            print("python -m spacy download en_core_web_lg")
            print("python -m spacy download de_core_news_lg")
            print("python -m spacy download fr_core_news_lg")
            return None
    
    def fuzzy_matching(self, url: str, live_urls: List[str]) -> Tuple[str, float]:
        """
        Find best match using fuzzy matching.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        best_score = 0
        best_match = None
        
        for live_url in live_urls:
            score = fuzz.token_sort_ratio(url, live_url)
            if score > best_score:
                best_score = score
                best_match = live_url
                
        return best_match, best_score / 100
    
    def levenshtein_matching(self, url: str, live_urls: List[str]) -> Tuple[str, float]:
        """
        Find best match using Levenshtein distance.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        best_score = float('inf')
        best_match = None
        
        for live_url in live_urls:
            score = lev.distance(url, live_url)
            if score < best_score:
                best_score = score
                best_match = live_url
                
        max_len = max(len(url), len(best_match))
        normalized_score = 1 - (best_score / max_len)
        return best_match, normalized_score
    
    def jaccard_matching(self, url: str, live_urls: List[str]) -> Tuple[str, float]:
        """
        Find best match using Jaccard similarity.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        url_set = set(url)
        best_score = 0
        best_match = None
        
        for live_url in live_urls:
            live_set = set(live_url)
            intersection = len(url_set.intersection(live_set))
            union = len(url_set.union(live_set))
            score = intersection / union if union > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = live_url
                
        return best_match, best_score
    
    def hamming_matching(self, url: str, live_urls: List[str]) -> Tuple[str, float]:
        """
        Find best match using Modified Hamming distance that handles different length URLs.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        best_score = 0
        best_match = None
        
        for live_url in live_urls:
            # Calculate length penalty (0 if same length, up to 0.5 for very different lengths)
            length_penalty = 0.5 * (abs(len(url) - len(live_url)) / max(len(url), len(live_url)))
            
            # Calculate Hamming distance on overlapping part
            min_length = min(len(url), len(live_url))
            hamming_distance = sum(a != b for a, b in zip(url[:min_length], live_url[:min_length]))
            
            # Normalize Hamming score (1 for perfect match, 0 for completely different)
            hamming_score = 1 - (hamming_distance / min_length)
            
            # Combine scores (hamming_score has more weight than length penalty)
            final_score = hamming_score * (1 - length_penalty)
            
            if final_score > best_score:
                best_score = final_score
                best_match = live_url
                
        return best_match, best_score
    
    def ratcliff_matching(self, url: str, live_urls: List[str]) -> Tuple[str, float]:
        """
        Find best match using Ratcliff/Obershelp algorithm.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        best_score = 0
        best_match = None
        
        for live_url in live_urls:
            score = SequenceMatcher(None, url, live_url).ratio()
            if score > best_score:
                best_score = score
                best_match = live_url
                
        return best_match, best_score
    
    def tversky_matching(self, url: str, live_urls: List[str]) -> Tuple[str, float]:
        """
        Find best match using Tversky index.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        url_set = set(url)
        best_score = 0
        best_match = None
        alpha = self.similarity_config['tversky_alpha']
        beta = self.similarity_config['tversky_beta']
        
        for live_url in live_urls:
            live_set = set(live_url)
            common = len(url_set.intersection(live_set))
            unique_to_url = len(url_set - live_set)
            unique_to_live = len(live_set - url_set)
            
            denominator = common + alpha * unique_to_url + beta * unique_to_live
            if denominator > 0:
                score = common / denominator
                if score > best_score:
                    best_score = score
                    best_match = live_url
                    
        return best_match, best_score
    
    def spacy_matching(self, url: str, live_urls: List[str]) -> Tuple[Optional[str], float]:
        """
        Match URLs using spaCy language model.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[Optional[str], float]: Best matching URL and similarity score
        """
        try:
            if self.nlp is None:
                log_error(ValueError("spaCy model not initialized"), self.logger, "spaCy model not initialized")
                return None, 0.0
                
            # Preprocess URLs to make them more suitable for NLP
            # Split by common URL separators to make them more like text
            def preprocess_url(url_str):
                # Replace certain characters with spaces to help spaCy
                return url_str.replace('/', ' ').replace('-', ' ').replace('_', ' ').replace('.', ' ')
                
            # Process the 404 URL with spaCy after preprocessing
            preprocessed_url = preprocess_url(url)
            url_doc = self.nlp(preprocessed_url)
            
            best_match = None
            best_score = 0.0
            
            # Calculate similarity with each live URL
            for live_url in live_urls:
                preprocessed_live_url = preprocess_url(live_url)
                live_doc = self.nlp(preprocessed_live_url)
                
                # Calculate similarity
                try:
                    score = url_doc.similarity(live_doc)
                except ValueError:
                    # Handle empty documents
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_match = live_url
            
            # Apply threshold
            spacy_threshold = self.similarity_config['spacy_threshold']
            if best_score >= spacy_threshold:
                return best_match, best_score
            else:
                return None, 0.0
                
        except Exception as e:
            log_error(e, self.logger, "Error in spaCy matching")
            return None, 0.0
    
    def vector_matching(self, url: str, live_urls: List[str], 
                       vectorizer: TfidfVectorizer, live_vectors: np.ndarray) -> Tuple[str, float]:
        """
        Find best match using TF-IDF vectors.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            vectorizer (TfidfVectorizer): Fitted vectorizer
            live_vectors (np.ndarray): Vectorized live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        query_vector = vectorizer.transform([url])
        similarities = cosine_similarity(query_vector, live_vectors).flatten()
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        best_match = live_urls[best_match_idx]
        
        return best_match, best_score
    
    def jaro_winkler_matching(self, url: str, live_urls: List[str]) -> Tuple[str, float]:
        """
        Find best match using Jaro-Winkler similarity.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        best_score = 0
        best_match = None
        
        for live_url in live_urls:
            score = jaro_winkler_similarity(url, live_url)
            if score > best_score:
                best_score = score
                best_match = live_url
                
        return best_match, best_score
    
    def bertopic_matching(self, url: str, live_urls: List[str]) -> Tuple[str, float]:
        """
        Find best match using BERTopic with pre-computed embeddings.
        Uses vectorized operations for improved performance.
        
        Args:
            url (str): URL to match
            live_urls (List[str]): List of live URLs
            
        Returns:
            Tuple[str, float]: (best_match, score)
        """
        if not self.bertopic_model or not live_urls or not self._bertopic_fitted:
            return None, 0
            
        try:
            # Use pre-computed topics from cache
            if url not in self._url_404_topics:
                # If not in cache, compute on-the-fly (should be rare)
                url_topics, url_probs = self.bertopic_model.transform([url])
                if url_topics is None or url_probs is None or len(url_topics) == 0 or len(url_probs) == 0:
                    return None, 0
                url_topic = url_topics[0]
                url_prob = url_probs[0]
            else:
                # Use cached values
                url_topic = self._url_404_topics[url]
                url_prob = self._url_404_probs[url]
                
            # Use pre-computed live URL topics
            if self._live_url_topics is None or self._live_url_probs is None:
                return None, 0
                
            # Vectorized matching
            # 1. Create a numpy array of all topics from live URLs
            live_topics = np.array(self._live_url_topics)
            live_probs = np.array(self._live_url_probs)
            
            # Handle case when live_topics is empty
            if len(live_topics) == 0 or len(live_urls) == 0:
                return None, 0
                
            # 2. Find matches based on topic equality (vectorized)
            # Create a mask for matching topics
            topic_matches = (live_topics == url_topic)
            
            # 3. Calculate scores using vectorized operations
            scores = np.zeros(len(live_urls))
            
            # For matching topics, use the probability directly
            if np.any(topic_matches):
                # Use the probability as score for matching topics
                matching_indices = np.where(topic_matches)[0]
                scores[matching_indices] = live_probs[matching_indices]
            
            # For non-matching topics with topic_representations
            if hasattr(self.bertopic_model, 'topic_representations_') and not np.all(topic_matches):
                # Apply the 0.3 scaling factor for different topics
                non_matching_indices = np.where(~topic_matches)[0]
                scores[non_matching_indices] = 0.3 * live_probs[non_matching_indices]
            
            # 4. Find the best match
            if np.max(scores) > 0:
                best_idx = np.argmax(scores)
                if best_idx < len(live_urls):
                    return live_urls[best_idx], scores[best_idx]
            
            return None, 0
            
        except Exception as e:
            log_error(e, self.logger, "Error in BERTopic matching")
            return None, 0
    
    def ml_matching(self, url: str, live_urls: List[str]) -> Tuple[Optional[str], float]:
        """
        ML-based URL matching algorithm.
        
        Args:
            url (str): The 404 URL to find a match for
            live_urls (List[str]): List of live URLs to match against
            
        Returns:
            Tuple[Optional[str], float]: Best matching URL and confidence score
        """
        try:
            # First check if ML matching is disabled in config
            if not self.algorithm_config['use_ml']:
                return None, 0.0
                
            # Then check if the ML matcher exists and is initialized
            if not self.ml_matcher or not self.ml_matcher.model_initialized:
                return None, 0.0
                
            # Use the AIMatchingAlgorithm to find the best match
            best_match, confidence_score = self.ml_matcher.find_best_match(url, live_urls)
            
            # Apply threshold
            ml_threshold = self.similarity_config['ml_threshold']
            if confidence_score >= ml_threshold:
                return best_match, confidence_score
            else:
                return None, 0.0
                
        except Exception as e:
            log_error(e, self.logger, f"Error in ML matching for URL: {url}")
            return None, 0.0 