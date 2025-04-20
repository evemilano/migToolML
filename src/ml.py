"""
AI/ML matching algorithm with continuous learning for the URL Migration Tool.

This module implements a machine learning system that continuously learns from the results
of other matching algorithms to improve URL matching over time. Key features:

1. Incremental learning: Model updates with each script execution
2. No separate training phase: Learning happens during normal operation
3. Feature extraction from URLs for effective matching
4. Persistent storage of model and match history
5. Integration with existing matching algorithms

The module provides the AIMatchingAlgorithm class which:
- Loads/creates ML models for URL matching
- Extracts meaningful features from URLs
- Makes predictions for the best matching URLs
- Updates itself based on agreement between other algorithms
- Saves state between script executions for continuous improvement
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Union
import joblib
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import urlparse, parse_qs
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix, hstack
from src.logger import log_error, log_info, log_warning

class AIMatchingAlgorithm:
    """Handles URL matching using machine learning with continuous learning."""
    
    def __init__(self, logger):
        """
        Initialize AIMatchingAlgorithm.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        
        # Define file paths for persistence
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.model_file = self.data_dir / "ai_model.joblib"
        self.tfidf_file = self.data_dir / "tfidf_vectorizer.joblib"
        self.scaler_file = self.data_dir / "feature_scaler.joblib"
        self.history_file = self.data_dir / "match_history.csv"
        
        # Model components
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.match_history = None
        
        # Model parameters
        self.min_training_samples = 100  # Minimum samples needed before making predictions
        self.agreement_threshold = 8    # Minimum agreement score to consider for training
        self.high_confidence_threshold = 10  # High confidence threshold for stronger learning
        self.model_initialized = False
        self.max_history_size = 1000000  # Maximum number of records to keep in history
        
        # Load existing model and history if available
        self._load_model_and_history()
        
    def _load_model_and_history(self):
        """Load existing model and match history if available, otherwise initialize new ones."""
        try:
            # Check if model exists
            if self.model_file.exists() and self.tfidf_file.exists() and self.scaler_file.exists():
                self.model = joblib.load(self.model_file)
                self.tfidf_vectorizer = joblib.load(self.tfidf_file)
                self.feature_scaler = joblib.load(self.scaler_file)
                self.model_initialized = True
                log_info("AI model loaded successfully", self.logger)
            else:
                self._initialize_model()
                log_info("New AI model initialized", self.logger)
            
            # Load match history
            if self.history_file.exists():
                self.match_history = pd.read_csv(self.history_file)
                log_info(f"Loaded match history with {len(self.match_history)} records", self.logger)
                # Clean history if it exceeds maximum size
                self._clean_history_if_needed()
            else:
                # Create empty match history DataFrame
                self.match_history = pd.DataFrame(columns=[
                    'timestamp', 'url_404', 'redirect_url', 'agreement_score',
                    'total_score', 'algorithm', 'is_best_match'
                ])
                log_info("Created new match history", self.logger)
                
        except Exception as e:
            log_error(e, self.logger, "Error loading AI model or history")
            self._initialize_model()
            # Create new history if loading failed
            self.match_history = pd.DataFrame(columns=[
                'timestamp', 'url_404', 'redirect_url', 'agreement_score',
                'total_score', 'algorithm', 'is_best_match'
            ])
    
    def _clean_history_if_needed(self):
        """Clean history using a sophisticated strategy to maintain quality and diversity."""
        if self.match_history is not None and len(self.match_history) > self.max_history_size:
            # Convert timestamp to datetime if not already
            self.match_history['timestamp'] = pd.to_datetime(self.match_history['timestamp'])
            
            # 1. Mantieni tutti i match ad alta confidenza
            high_confidence_mask = self.match_history['agreement_score'] >= self.high_confidence_threshold
            high_confidence_matches = self.match_history[high_confidence_mask].copy()
            
            # 2. Per i rimanenti record, implementa una strategia bilanciata
            other_matches = self.match_history[~high_confidence_mask].copy()
            
            # Estrai il codice lingua dall'URL (assumendo formato standard tipo /it/, /en/, /fr/, ecc.)
            def extract_language(url):
                import re
                lang_match = re.search(r'/([a-z]{2})/', url)
                return lang_match.group(1) if lang_match else 'unknown'
            
            other_matches['language'] = other_matches['url_404'].apply(extract_language)
            
            # Calcola quanti record mantenere per lingua
            remaining_slots = self.max_history_size - len(high_confidence_matches)
            records_per_lang = remaining_slots // other_matches['language'].nunique()
            
            # Seleziona i record più recenti per ogni lingua
            balanced_matches = []
            for lang in other_matches['language'].unique():
                lang_matches = other_matches[other_matches['language'] == lang]
                # Prendi i più recenti per questa lingua
                recent_lang_matches = lang_matches.nlargest(records_per_lang, 'timestamp')
                balanced_matches.append(recent_lang_matches)
            
            # Combina tutti i record selezionati
            if balanced_matches:
                balanced_df = pd.concat(balanced_matches)
                final_history = pd.concat([high_confidence_matches, balanced_df])
            else:
                final_history = high_confidence_matches
                
            # Ordina per timestamp e rimuovi la colonna temporanea 'language'
            self.match_history = final_history.sort_values('timestamp', ascending=False)
            if 'language' in self.match_history.columns:
                self.match_history = self.match_history.drop('language', axis=1)
                
            log_info(f"Cleaned match history: {len(high_confidence_matches)} high confidence matches, "
                    f"{len(self.match_history) - len(high_confidence_matches)} balanced matches", 
                    self.logger)
    
    def _initialize_model(self):
        """Initialize a new model and vectorizers."""
        try:
            # Initialize TF-IDF vectorizer for text features
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                min_df=1,
                max_features=100
            )
            
            # Initialize feature scaler
            self.feature_scaler = StandardScaler()
            
            # Initialize RandomForest model with default parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced classes better
            )
            
            self.model_initialized = False  # Not fully initialized until trained
            
        except Exception as e:
            log_error(e, self.logger, "Error initializing AI model")
            raise
    
    def _save_model_and_history(self):
        """Save the current model and match history."""
        try:
            # Create data directory if it doesn't exist
            self.data_dir.mkdir(exist_ok=True)
            
            # Create temporary filenames for atomic saving
            temp_model_file = self.model_file.with_suffix('.tmp')
            temp_tfidf_file = self.tfidf_file.with_suffix('.tmp')
            temp_scaler_file = self.scaler_file.with_suffix('.tmp')
            temp_history_file = self.history_file.with_suffix('.tmp')
            
            # Save model components
            if self.model_initialized:
                joblib.dump(self.model, temp_model_file)
                joblib.dump(self.tfidf_vectorizer, temp_tfidf_file)
                joblib.dump(self.feature_scaler, temp_scaler_file)
                
                # Rename temporary files to final filenames (atomic operation)
                temp_model_file.replace(self.model_file)
                temp_tfidf_file.replace(self.tfidf_file)
                temp_scaler_file.replace(self.scaler_file)
                log_info("AI model saved successfully", self.logger)
            
            # Save match history
            if self.match_history is not None and not self.match_history.empty:
                self.match_history.to_csv(temp_history_file, index=False)
                temp_history_file.replace(self.history_file)
                log_info(f"Saved match history with {len(self.match_history)} records", self.logger)
                
        except Exception as e:
            log_error(e, self.logger, "Error saving AI model or history")
    
    def _extract_path_segments(self, url: str) -> List[str]:
        """Extract path segments from URL."""
        try:
            if not url or not isinstance(url, str):
                return []
                
            parsed = urlparse(url)
            path = parsed.path
            # Split path into segments and filter empty segments
            segments = [s for s in path.split('/') if s]
            return segments
        except Exception as e:
            log_error(e, self.logger, f"Error extracting path segments from URL: {url}")
            return []
    
    def _extract_extensions(self, url: str) -> str:
        """Extract file extensions from URL."""
        try:
            if not url or not isinstance(url, str):
                return ""
                
            path = urlparse(url).path
            filename = os.path.basename(path)
            extension = os.path.splitext(filename)[1]
            return extension.lower() if extension else ""
        except Exception as e:
            log_error(e, self.logger, f"Error extracting extension from URL: {url}")
            return ""
    
    def _extract_query_params(self, url: str) -> Dict:
        """Extract query parameters from URL."""
        try:
            if not url or not isinstance(url, str):
                return {}
                
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            return {k: v[0] if len(v) == 1 else v for k, v in params.items()}
        except Exception as e:
            log_error(e, self.logger, f"Error extracting query parameters from URL: {url}")
            return {}
    
    def _compute_path_similarity(self, url1: str, url2: str) -> float:
        """Compute similarity between URL paths."""
        try:
            if not url1 or not url2 or not isinstance(url1, str) or not isinstance(url2, str):
                return 0.0
                
            path1 = urlparse(url1).path
            path2 = urlparse(url2).path
            return SequenceMatcher(None, path1, path2).ratio()
        except Exception as e:
            log_error(e, self.logger, f"Error computing path similarity between URLs: {url1}, {url2}")
            return 0.0
    
    def _compute_segment_matches(self, url1: str, url2: str) -> float:
        """Compute matching segments between URLs."""
        try:
            if not url1 or not url2 or not isinstance(url1, str) or not isinstance(url2, str):
                return 0.0
                
            segments1 = self._extract_path_segments(url1)
            segments2 = self._extract_path_segments(url2)
            
            if not segments1 or not segments2:
                return 0.0
            
            # Count matching segments
            matches = sum(1 for s1 in segments1 if s1 in segments2)
            total = max(len(segments1), len(segments2))
            
            return matches / total if total > 0 else 0.0
        except Exception as e:
            log_error(e, self.logger, f"Error computing segment matches between URLs: {url1}, {url2}")
            return 0.0
    
    def _extract_features(self, url_404: str, live_url: str) -> List[float]:
        """
        Extract features from a pair of URLs.
        
        Args:
            url_404: The 404 URL
            live_url: A candidate live URL
            
        Returns:
            List[float]: Feature vector
        """
        try:
            if not url_404 or not live_url or not isinstance(url_404, str) or not isinstance(live_url, str):
                return [0.0] * 6
                
            # Text-based features
            path_similarity = self._compute_path_similarity(url_404, live_url)
            segment_match_ratio = self._compute_segment_matches(url_404, live_url)
            
            # Length-based features
            len_ratio = min(len(url_404), len(live_url)) / max(len(url_404), len(live_url)) if max(len(url_404), len(live_url)) > 0 else 0
            
            # Extension matching
            ext_404 = self._extract_extensions(url_404)
            ext_live = self._extract_extensions(live_url)
            ext_match = 1.0 if ext_404 and ext_live and ext_404 == ext_live else 0.0
            
            # Query parameter features
            params_404 = self._extract_query_params(url_404)
            params_live = self._extract_query_params(live_url)
            
            # Common parameters - fix calculation
            common_params = set(params_404.keys()) & set(params_live.keys())
            all_params = set(params_404.keys()) | set(params_live.keys())
            param_overlap = len(common_params) / max(len(all_params), 1)
            
            # Domain features
            domain_404 = urlparse(url_404).netloc
            domain_live = urlparse(live_url).netloc
            domain_match = 1.0 if domain_404 == domain_live else 0.0
            
            # Common numerical feature array
            features = [
                path_similarity,
                segment_match_ratio,
                len_ratio,
                ext_match,
                param_overlap,
                domain_match
            ]
            
            return features
            
        except Exception as e:
            log_error(e, self.logger, f"Error extracting features from URLs: {url_404}, {live_url}")
            return [0.0] * 6  # Return zeros if feature extraction fails
    
    def _get_tfidf_similarity(self, url_404: str, live_url: str) -> float:
        """
        Calculate TF-IDF similarity between two URLs.
        
        Args:
            url_404: The 404 URL
            live_url: A candidate live URL
            
        Returns:
            float: TF-IDF similarity score
        """
        try:
            if not url_404 or not live_url or not isinstance(url_404, str) or not isinstance(live_url, str):
                return 0.0
                
            # Prepare TF-IDF features
            tfidf_features = self._prepare_tfidf_features([url_404, live_url])
            
            if tfidf_features is not None:
                # Get vectors for both URLs
                url_404_tfidf = tfidf_features[0]
                live_url_tfidf = tfidf_features[1]
                
                # Calculate cosine similarity
                similarity = url_404_tfidf.dot(live_url_tfidf.T).toarray()[0][0]
                return similarity
            else:
                return 0.0
                
        except Exception as e:
            log_error(e, self.logger, f"Error calculating TF-IDF similarity between URLs: {url_404}, {live_url}")
            return 0.0

    def extract_all_features(self, url_404: str, live_url: str) -> List[float]:
        """
        Extract all features (basic + TF-IDF) from a pair of URLs.
        
        Args:
            url_404: The 404 URL
            live_url: A candidate live URL
            
        Returns:
            List[float]: Complete feature vector
        """
        # Get basic features
        basic_features = self._extract_features(url_404, live_url)
        
        # Get TF-IDF similarity feature
        tfidf_similarity = self._get_tfidf_similarity(url_404, live_url)
        
        # Combine features
        all_features = basic_features + [tfidf_similarity]
        
        return all_features

    def _prepare_tfidf_features(self, urls: List[str]) -> Optional[csr_matrix]:
        """
        Prepare TF-IDF features for URLs.
        
        Args:
            urls: List of URLs to vectorize
            
        Returns:
            Optional[csr_matrix]: TF-IDF matrix or None if error
        """
        try:
            if not urls:
                return None
                
            # Fit or transform with TF-IDF vectorizer
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                return self.tfidf_vectorizer.fit_transform(urls)
            else:
                return self.tfidf_vectorizer.transform(urls)
        except Exception as e:
            log_error(e, self.logger, "Error preparing TF-IDF features")
            return None
            
    def update_from_matches(self, matches: List[Dict], df_results: pd.DataFrame):
        """
        Update AI model from match results.
        
        Args:
            matches: List of match dictionaries
            df_results: DataFrame with processed results including agreement scores
        """
        log_info("Updating AI model from match results", self.logger)
        
        try:
            # Extract match information
            new_records = []
            training_data = []
            training_labels = []
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Find best matches from processed results
            for idx, row in df_results.iterrows():
                url_404 = row['404 URL'] if '404 URL' in row else row.get('url_404', None)
                if url_404 is None:
                    continue
                    
                best_redirect = row.get('Best redirect', None)
                if best_redirect is None:
                    continue
                    
                agreement_score = row.get('Agreement', 0)
                total_score = row.get('TotScore', 0)
                
                # Add to training data if agreement is high enough
                if agreement_score >= self.agreement_threshold:
                    # Find all candidate redirects from matches
                    match_entry = next((m for m in matches if m['url_404'] == url_404), None)
                    if not match_entry:
                        continue
                        
                    # Get all unique redirect candidates
                    all_redirects = set()
                    for algo_name, redirect_url, score in match_entry['matches']:
                        if redirect_url and redirect_url not in all_redirects:
                            all_redirects.add(redirect_url)
                            
                            # Add to history
                            new_records.append({
                                'timestamp': current_time,
                                'url_404': url_404,
                                'redirect_url': redirect_url,
                                'agreement_score': agreement_score,
                                'total_score': total_score,
                                'algorithm': algo_name,
                                'is_best_match': (redirect_url == best_redirect)
                            })
                    
                    # Extract features and labels for all redirects
                    for redirect_url in all_redirects:
                        # Extract all features (basic + TF-IDF)
                        features = self.extract_all_features(url_404, redirect_url)
                        training_data.append(features)
                        
                        # Label 1 for best match, 0 for others
                        is_best_match = (redirect_url == best_redirect)
                        training_labels.append(1 if is_best_match else 0)
                        
                        # Apply higher weight to high confidence matches
                        if agreement_score >= self.high_confidence_threshold and is_best_match:
                            # Add duplicate entries for high confidence matches to increase their weight
                            for _ in range(2):  # Add two more copies
                                training_data.append(features)
                                training_labels.append(1)
            
            # Update match history
            if new_records:
                new_history = pd.DataFrame(new_records)
                if self.match_history is not None and not self.match_history.empty:
                    self.match_history = pd.concat([self.match_history, new_history], ignore_index=True)
                else:
                    self.match_history = new_history
                
                log_info(f"Added {len(new_records)} new records to match history", self.logger)
                # Clean history if it exceeds maximum size
                self._clean_history_if_needed()
            
            # Train model if we have enough training data
            if training_data and len(training_data) >= self.min_training_samples:
                # Convert to numpy arrays
                X = np.array(training_data)
                y = np.array(training_labels)
                
                log_info(f"Training model with {X.shape[1]} features", self.logger)
                
                # Scale features
                if not hasattr(self.feature_scaler, 'n_features_in_'):
                    X_scaled = self.feature_scaler.fit_transform(X)
                else:
                    X_scaled = self.feature_scaler.transform(X)
                
                # Train the model
                self.model.fit(X_scaled, y)
                self.model_initialized = True
                
                # Log training summary
                positive_samples = sum(y)
                log_info(f"Model trained with {len(y)} samples ({positive_samples} positive, {len(y) - positive_samples} negative)", self.logger)
            elif training_data:
                log_warning(f"Not enough training samples yet: {len(training_data)}/{self.min_training_samples}", self.logger)
            else:
                log_warning("No training data available from current matches", self.logger)
            
            # Save updated model and history
            self._save_model_and_history()
            
        except Exception as e:
            log_error(e, self.logger, "Error updating AI model from matches")
    
    def find_best_match(self, url_404: str, live_urls: List[str]) -> Tuple[Optional[str], float]:
        """
        Find best match for a 404 URL among live URLs using the AI model.
        
        Args:
            url_404: The 404 URL to find a match for
            live_urls: List of candidate live URLs
            
        Returns:
            Tuple[Optional[str], float]: (best_match_url, confidence_score)
        """
        if not self.model_initialized or not live_urls:
            return None, 0.0
            
        try:
            # Validate inputs
            if not url_404 or not isinstance(url_404, str):
                log_error(ValueError("Invalid 404 URL"), self.logger, "Invalid 404 URL provided")
                return None, 0.0
                
            # Filter valid live URLs
            valid_live_urls = [url for url in live_urls if url and isinstance(url, str)]
            if not valid_live_urls:
                return None, 0.0
                
            # Extract features for all candidate pairs
            feature_vectors = []
            for live_url in valid_live_urls:
                features = self.extract_all_features(url_404, live_url)
                feature_vectors.append(features)
            
            # Convert to array
            X = np.array(feature_vectors)
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Get predictions and probabilities
            probas = self.model.predict_proba(X_scaled)
            
            # Extract positive class probability (confidence score)
            if probas.shape[1] > 1:  # Binary classification
                confidence_scores = probas[:, 1]  # Get probability of positive class
            else:  # Single class - handle this case explicitly
                if self.model.classes_[0] == 1:  # If the only class is positive
                    confidence_scores = probas[:, 0]
                else:  # If the only class is negative
                    confidence_scores = 1 - probas[:, 0]
                
            # Find best match
            best_idx = np.argmax(confidence_scores)
            best_score = confidence_scores[best_idx]
            best_match = valid_live_urls[best_idx]
            
            return best_match, float(best_score)
            
        except Exception as e:
            log_error(e, self.logger, f"Error finding best match with AI for URL: {url_404}")
            return None, 0.0
            
    def get_model_stats(self) -> Dict[str, Union[int, float, bool]]:
        """
        Get statistics about the current model.
        
        Returns:
            Dict: Model statistics
        """
        stats = {
            "model_initialized": self.model_initialized,
            "history_records": len(self.match_history) if self.match_history is not None else 0,
            "min_training_samples": self.min_training_samples,
            "model_features": self.feature_scaler.n_features_in_ if hasattr(self.feature_scaler, 'n_features_in_') else 0
        }
        
        return stats 