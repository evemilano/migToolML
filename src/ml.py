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
                # Ensure 'source' column exists, default to 'auto'
                if 'source' not in self.match_history.columns:
                    log_warning("\'source\' column not found in history, defaulting existing records to \'auto\'.", self.logger)
                    self.match_history['source'] = 'auto'
                else:
                    # Fill potential NaN values in source with 'auto'
                    self.match_history['source'] = self.match_history['source'].fillna('auto')
                # Clean history if it exceeds maximum size
                self._clean_history_if_needed()
            else:
                # Create empty match history DataFrame
                self.match_history = pd.DataFrame(columns=[
                    'timestamp', 'url_404', 'redirect_url', 'agreement_score',
                    'total_score', 'algorithm', 'is_best_match', 'source'
                ])
                log_info("Created new match history", self.logger)
                
        except Exception as e:
            log_error(e, self.logger, "Error loading AI model or history")
            self._initialize_model()
            # Create new history if loading failed
            self.match_history = pd.DataFrame(columns=[
                'timestamp', 'url_404', 'redirect_url', 'agreement_score',
                'total_score', 'algorithm', 'is_best_match', 'source'
            ])
    
    def _clean_history_if_needed(self):
        """Clean history using a sophisticated strategy to maintain quality and diversity."""
        if self.match_history is not None and len(self.match_history) > self.max_history_size:
            log_info(f"History size ({len(self.match_history)}) exceeds limit ({self.max_history_size}). Cleaning...", self.logger)
            # Ensure source column exists and has no NaNs before cleaning
            if 'source' not in self.match_history.columns:
                self.match_history['source'] = 'auto'
            else:
                self.match_history['source'] = self.match_history['source'].fillna('auto')

            # Convert timestamp to datetime if not already
            try:
                # Use errors='coerce' to handle potential invalid date formats gracefully
                self.match_history['timestamp'] = pd.to_datetime(self.match_history['timestamp'], errors='coerce')
                # Drop rows where timestamp conversion failed
                self.match_history.dropna(subset=['timestamp'], inplace=True)
            except Exception as e:
                log_error(e, self.logger, "Error converting timestamp column during cleaning. Some history might be lost.")
                # Attempt to proceed without timestamp sorting if conversion fails drastically
                pass

            # 1. Separate human-verified matches
            human_matches = self.match_history[self.match_history['source'] == 'human'].copy()
            auto_matches = self.match_history[self.match_history['source'] != 'human'].copy()

            log_info(f"Found {len(human_matches)} human-verified records to keep.", self.logger)

            # 2. Calculate remaining slots for auto matches
            remaining_slots = self.max_history_size - len(human_matches)

            if remaining_slots <= 0:
                # If human matches already exceed the limit, keep only human matches
                # (Optionally, could prune oldest human matches, but let's keep all for now)
                self.match_history = human_matches.sort_values('timestamp', ascending=False)
                log_warning(f"Human-verified matches ({len(human_matches)}) exceed max history size ({self.max_history_size}). Keeping only human data.", self.logger)
                return # Cleaning finished

            if len(auto_matches) <= remaining_slots:
                 # No need to clean auto_matches further, just combine and return
                self.match_history = pd.concat([human_matches, auto_matches]).sort_values('timestamp', ascending=False)
                log_info(f"No cleaning needed for 'auto' records ({len(auto_matches)} <= {remaining_slots} slots).", self.logger)
                return # Cleaning finished

            # 3. Apply original cleaning logic ONLY to auto_matches to fill remaining_slots
            log_info(f"Applying cleaning logic to {len(auto_matches)} 'auto' records to fit {remaining_slots} slots.", self.logger)

            # Ensure 'agreement_score' exists and is numeric before filtering
            if 'agreement_score' not in auto_matches.columns:
                 log_warning("Cannot perform high-confidence filtering: 'agreement_score' column missing in auto_matches.", self.logger)
                 high_confidence_auto = pd.DataFrame(columns=auto_matches.columns) # Empty DF
                 other_auto_matches = auto_matches.copy()
            else:
                 # Convert agreement_score to numeric, coercing errors
                 auto_matches['agreement_score'] = pd.to_numeric(auto_matches['agreement_score'], errors='coerce')
                 # Separate high confidence auto matches
                 high_confidence_mask = auto_matches['agreement_score'] >= self.high_confidence_threshold
                 high_confidence_auto = auto_matches[high_confidence_mask].copy()
                 other_auto_matches = auto_matches[~high_confidence_mask].copy()


            # Check if high confidence auto matches alone fill the remaining slots
            if len(high_confidence_auto) >= remaining_slots:
                # Keep only the most recent high-confidence auto matches
                kept_auto_matches = high_confidence_auto.nlargest(remaining_slots, 'timestamp')
                log_info(f"Keeping {len(kept_auto_matches)} most recent high-confidence 'auto' records.", self.logger)
            else:
                # Keep all high-confidence auto matches and fill the rest with balanced 'other' auto matches
                slots_for_others = remaining_slots - len(high_confidence_auto)

                if 'url_404' not in other_auto_matches.columns:
                     log_warning("Cannot perform language balancing: 'url_404' column missing in other_auto_matches.", self.logger)
                     balanced_other_matches_df = pd.DataFrame(columns=other_auto_matches.columns) # Empty DF
                elif slots_for_others > 0 and not other_auto_matches.empty:
                    # Extract language (handle potential errors)
                    def extract_language(url):
                        import re
                        try:
                            if pd.isna(url): return 'unknown'
                            lang_match = re.search(r'/([a-z]{2})/', str(url))
                            return lang_match.group(1) if lang_match else 'unknown'
                        except Exception:
                            return 'unknown' # Robustness against non-string URLs etc.

                    other_auto_matches['language'] = other_auto_matches['url_404'].apply(extract_language)
                    unique_languages = other_auto_matches['language'].nunique()

                    if unique_languages > 0:
                         records_per_lang = max(1, slots_for_others // unique_languages) # Ensure at least 1 per lang if possible
                         balanced_other_matches = []
                         # Select the most recent records per language
                         for lang, group in other_auto_matches.groupby('language'):
                             # Ensure timestamp column exists and is sortable before nlargest
                             if 'timestamp' in group.columns:
                                 recent_lang_matches = group.nlargest(records_per_lang, 'timestamp')
                                 balanced_other_matches.append(recent_lang_matches)
                             else:
                                 # Fallback if timestamp is missing or problematic
                                 balanced_other_matches.append(group.head(records_per_lang))

                         if balanced_other_matches:
                             balanced_other_matches_df = pd.concat(balanced_other_matches).head(slots_for_others) # Ensure we don't exceed slots_for_others
                         else:
                             balanced_other_matches_df = pd.DataFrame(columns=other_auto_matches.columns) # Empty DF
                    else:
                         # If no unique languages found, just take the most recent overall
                         balanced_other_matches_df = other_auto_matches.nlargest(slots_for_others, 'timestamp')

                    # Drop the temporary language column
                    if 'language' in balanced_other_matches_df.columns:
                        balanced_other_matches_df = balanced_other_matches_df.drop('language', axis=1)

                    log_info(f"Keeping {len(balanced_other_matches_df)} balanced 'other' auto records.", self.logger)

                else: # No slots left for others or no others exist
                     balanced_other_matches_df = pd.DataFrame(columns=other_auto_matches.columns) # Empty DF


                # Combine high-confidence auto and balanced other auto matches
                kept_auto_matches = pd.concat([high_confidence_auto, balanced_other_matches_df])


            # 4. Combine kept human matches and kept auto matches
            final_history = pd.concat([human_matches, kept_auto_matches])

            # Sort final history by timestamp
            self.match_history = final_history.sort_values('timestamp', ascending=False)

            # Drop temporary language column if it somehow persisted
            if 'language' in self.match_history.columns:
                 self.match_history = self.match_history.drop('language', axis=1)

            log_info(f"Finished cleaning history. Kept {len(human_matches)} human records and {len(kept_auto_matches)} auto records. Total: {len(self.match_history)}.", self.logger)
    
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
        Update AI model from match results, prioritizing human-verified data from history.

        Args:
            matches: List of match dictionaries from the current run.
            df_results: DataFrame with processed results from the current run, including agreement scores.
        """
        log_info("Updating AI model from match results (prioritizing human data)", self.logger)

        try:
            # --- 1. Prepare Human-Verified Data ---
            human_training_data = []
            human_training_labels = []
            if self.match_history is not None and not self.match_history.empty:
                human_entries = self.match_history[self.match_history['source'] == 'human']
                if not human_entries.empty:
                    log_info(f"Found {len(human_entries)} human-verified records in history for training.", self.logger)
                    for _, row in human_entries.iterrows():
                        url_404 = row['url_404']
                        redirect_url = row['redirect_url']
                        # Ensure URLs are valid strings before feature extraction
                        if isinstance(url_404, str) and isinstance(redirect_url, str):
                            features = self.extract_all_features(url_404, redirect_url)
                            human_training_data.append(features)
                            human_training_labels.append(1) # Human verified is always a positive match
                        else:
                             log_warning(f"Skipping human record due to invalid URL types: 404='{url_404}', Redirect='{redirect_url}'", self.logger)

            # --- 2. Prepare Auto-Verified Data from Current Run ---
            auto_training_data = []
            auto_training_labels = []
            new_history_records = [] # Keep track of new records to add *after* training prep
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            processed_404s_for_history = set() # Avoid duplicate history entries from the same run

            for idx, row in df_results.iterrows():
                url_404 = row['404 URL'] if '404 URL' in row else row.get('url_404', None)
                if url_404 is None:
                    continue

                best_redirect = row.get('Best redirect', None)
                if best_redirect is None:
                    continue

                agreement_score = row.get('Agreement', 0)
                total_score = row.get('TotScore', 0)

                # Find the corresponding match entry from the *current* run's matches
                match_entry = next((m for m in matches if m.get('url_404') == url_404), None)
                if not match_entry:
                    log_warning(f"Could not find original match entry for 404 URL: {url_404} in current run data.", self.logger)
                    continue

                # Add to training data *only* if agreement is high enough (current run)
                if agreement_score >= self.agreement_threshold:
                    # Get all unique redirect candidates *from this specific match_entry*
                    current_run_redirects = set()
                    # Use match_entry['matches'] which should be List[Tuple[str, str, float]]
                    if 'matches' in match_entry and isinstance(match_entry['matches'], list):
                        for algo_name, redirect_url, score in match_entry['matches']:
                            if redirect_url and isinstance(redirect_url, str) and redirect_url not in current_run_redirects:
                                current_run_redirects.add(redirect_url)

                                # Add to new history records (only once per 404 in this run)
                                if url_404 not in processed_404s_for_history:
                                     new_history_records.append({
                                         'timestamp': current_time,
                                         'url_404': url_404,
                                         'redirect_url': redirect_url,
                                         'agreement_score': agreement_score, # Agreement for the *best* match
                                         'total_score': total_score,         # Total score for the *best* match
                                         'algorithm': algo_name,             # Algorithm suggesting *this* redirect
                                         'is_best_match': (redirect_url == best_redirect), # Is *this* redirect the best?
                                         'source': 'auto'
                                     })


                    # Extract features and labels for all redirects suggested in *this run*
                    for redirect_url in current_run_redirects:
                        # Ensure redirect_url is valid before proceeding
                        if not isinstance(redirect_url, str):
                            log_warning(f"Skipping invalid redirect URL type: '{redirect_url}' for 404: '{url_404}'", self.logger)
                            continue

                        features = self.extract_all_features(url_404, redirect_url)
                        auto_training_data.append(features)

                        is_best_match = (redirect_url == best_redirect)
                        auto_training_labels.append(1 if is_best_match else 0)

                        # Apply higher weight (by duplication) only to high confidence *positive* matches
                        if agreement_score >= self.high_confidence_threshold and is_best_match:
                            for _ in range(2): # Add two more copies for weighting
                                auto_training_data.append(features)
                                auto_training_labels.append(1)

                    processed_404s_for_history.add(url_404) # Mark this 404 as processed for history

            # --- 3. Update Match History (after processing current run) ---
            if new_history_records:
                new_history = pd.DataFrame(new_history_records)
                if self.match_history is not None and not self.match_history.empty:
                    # Ensure columns match before concatenating
                    for col in new_history.columns:
                         if col not in self.match_history.columns:
                              self.match_history[col] = pd.NA # Add missing columns to existing history
                    for col in self.match_history.columns:
                         if col not in new_history.columns:
                              new_history[col] = pd.NA # Add missing columns to new history batch
                    new_history = new_history[self.match_history.columns] # Ensure order matches

                    self.match_history = pd.concat([self.match_history, new_history], ignore_index=True)
                else:
                    self.match_history = new_history

                log_info(f"Added {len(new_history_records)} new 'auto' records to match history.", self.logger)
                self._clean_history_if_needed() # Clean history if needed after adding new records

            # --- 4. Combine Data and Prepare for Training ---
            combined_training_data = human_training_data + auto_training_data
            combined_training_labels = human_training_labels + auto_training_labels

            if not combined_training_data:
                log_warning("No training data available (neither human nor auto) for this update.", self.logger)
                self._save_model_and_history() # Still save history even if no training
                return # Exit if no data

            # Minimum samples check applies to the combined data
            if len(combined_training_data) >= self.min_training_samples:
                X = np.array(combined_training_data)
                y = np.array(combined_training_labels)

                log_info(f"Preparing combined training data: {len(human_training_data)} human samples, {len(auto_training_data)} auto samples. Total: {len(X)}")

                # --- 5. Create Sample Weights ---
                human_weight = 10.0 # Assign higher weight to human data
                auto_weight = 1.0
                sample_weights = np.array(
                    [human_weight] * len(human_training_data) +
                    [auto_weight] * len(auto_training_data)
                )

                log_info(f"Using sample weights: {human_weight} for human, {auto_weight} for auto.")

                # --- 6. Scale Features ---
                try:
                    if not hasattr(self.feature_scaler, 'n_features_in_') or self.feature_scaler.n_features_in_ != X.shape[1]:
                        # Check if scaler needs fitting/refitting (e.g., first time or feature dimension changed)
                         log_info("Fitting feature scaler.", self.logger)
                         X_scaled = self.feature_scaler.fit_transform(X)
                    else:
                         # Check for potential NaN/inf values before transforming
                         if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                              log_warning("NaN or Inf found in feature data before scaling. Attempting to handle...", self.logger)
                              # Simple imputation: replace NaN with 0, Inf with large finite number
                              X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)

                         log_info("Transforming features using existing scaler.", self.logger)
                         X_scaled = self.feature_scaler.transform(X)

                except ValueError as ve:
                     log_error(ve, self.logger, f"ValueError during feature scaling. X shape: {X.shape}. Scaler features expected: {getattr(self.feature_scaler, 'n_features_in_', 'N/A')}. Skipping training for this run.")
                     self._save_model_and_history() # Save history
                     return
                except Exception as e:
                    log_error(e, self.logger, "Error during feature scaling. Skipping training for this run.")
                    self._save_model_and_history() # Save history
                    return


                # --- 7. Train the Model ---
                log_info(f"Training model with {X_scaled.shape[0]} samples and {X_scaled.shape[1]} features.", self.logger)

                try:
                     # Ensure model exists
                     if self.model is None:
                          log_error(ValueError("Model is None, cannot train."), self.logger, "Attempted to train a non-existent model.")
                          self._initialize_model() # Try re-initializing
                          if self.model is None: # If still None, fatal error
                               raise RuntimeError("Failed to initialize model for training.")

                     # Check class balance in the current batch
                     unique_labels, counts = np.unique(y, return_counts=True)
                     label_counts = dict(zip(unique_labels, counts))
                     log_info(f"Training batch label distribution: {label_counts}")
                     if len(label_counts) < 2:
                         log_warning(f"Training batch contains only one class ({label_counts}). Model fitting might be suboptimal or fail.")
                         # Optional: Add logic here if training requires multiple classes (e.g., skip training)
                         # For RandomForest, it might still run but won't learn effectively.

                     self.model.fit(X_scaled, y, sample_weight=sample_weights)
                     self.model_initialized = True # Mark as initialized/trained

                     positive_samples = sum(y)
                     log_info(f"Model updated successfully using combined data ({positive_samples} positive labels in batch).", self.logger)

                except Exception as e:
                     log_error(e, self.logger, "Error occurred during model fitting. Model state might be inconsistent.")
                     # Decide if we should save potentially corrupted model? Maybe not.
                     # Let's save history only in case of training failure.
                     self.match_history.to_csv(self.history_file.with_suffix('.err.csv'), index=False) # Save history to error file
                     log_warning("Saving history to .err.csv due to training failure. Model files not updated.")
                     return # Don't proceed to saving potentially bad model state


            elif combined_training_data: # Data exists but not enough samples
                log_warning(f"Not enough combined training samples to update model: {len(combined_training_data)}/{self.min_training_samples}", self.logger)
            else: # Should have been caught earlier, but defensive check
                 log_warning("No combined training data available.", self.logger)


            # --- 8. Save Updated Model and History ---
            # Save only if training was successful or if no training occurred but history was updated
            self._save_model_and_history()

        except Exception as e:
            log_error(e, self.logger, "Unhandled error updating AI model from matches")
            # Attempt to save history even if there was a major error
            try:
                 if self.match_history is not None:
                      self.match_history.to_csv(self.history_file.with_suffix('.err.csv'), index=False)
                      log_warning("Saving history to .err.csv due to unhandled error in update_from_matches.")
            except Exception as save_err:
                 log_error(save_err, self.logger, "Failed to save history during error handling.")

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

    def add_human_verified_matches(self, verified_pairs: List[Tuple[str, str]]):
        """
        Adds human-verified matches to the history, marks them, and saves.
        Does not retrain the model immediately.

        Args:
            verified_pairs: A list of (url_404, verified_live_url) tuples.
        """
        if not verified_pairs:
            log_info("No human-verified pairs provided to add.", self.logger)
            return

        log_info(f"Adding {len(verified_pairs)} human-verified matches to history.", self.logger)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        human_records = []

        for url_404, verified_live_url in verified_pairs:
            # Validate URLs before proceeding
            if not url_404 or not verified_live_url or not isinstance(url_404, str) or not isinstance(verified_live_url, str):
                 log_warning(f"Skipping invalid human-verified pair: ({url_404}, {verified_live_url})", self.logger)
                 continue

            # Create a history record for the human-verified pair
            # Set high agreement/total score placeholders? Or keep NaN/None? Let's use placeholders for now.
            human_records.append({
                'timestamp': current_time,
                'url_404': url_404,
                'redirect_url': verified_live_url,
                'agreement_score': 999, # Placeholder for high confidence
                'total_score': 999,     # Placeholder for high confidence
                'algorithm': 'human',   # Mark algorithm as human
                'is_best_match': True,  # Assumed true as it's human verified
                'source': 'human'       # Mark source as human
            })

        if not human_records:
             log_info("No valid human-verified pairs to add after validation.", self.logger)
             return

        # Create DataFrame from the new human records
        df_human_history = pd.DataFrame(human_records)

        # Ensure columns match the main history DataFrame before concatenating
        # This handles cases where the main history might have different columns initially
        if self.match_history is None:
             # If history was None, initialize it with the columns from human data
             self.match_history = pd.DataFrame(columns=df_human_history.columns)

        # Add missing columns to human history df if they exist in main history, fill with default
        for col in self.match_history.columns:
             if col not in df_human_history.columns:
                 # Determine appropriate fill value based on expected type (could be more sophisticated)
                 fill_value = pd.NA
                 if col in ['agreement_score', 'total_score']:
                     fill_value = 0
                 elif col == 'is_best_match':
                     fill_value = False
                 elif col == 'source':
                      fill_value = 'auto' # Should not happen here, but safe default

                 df_human_history[col] = fill_value

        # Add missing columns to main history df if they exist in human history, fill with default
        for col in df_human_history.columns:
              if col not in self.match_history.columns:
                   fill_value = pd.NA
                   if col in ['agreement_score', 'total_score']:
                       fill_value = 0
                   elif col == 'is_best_match':
                       fill_value = False
                   elif col == 'source':
                        fill_value = 'auto'

                   self.match_history[col] = fill_value


        # Ensure column order consistency before concatenation
        df_human_history = df_human_history[self.match_history.columns]


        # Concatenate new human records with existing history
        if self.match_history is not None and not self.match_history.empty:
             self.match_history = pd.concat([self.match_history, df_human_history], ignore_index=True)
        else:
             self.match_history = df_human_history


        # Ensure no duplicate entries based on 404/redirect pair (keep latest if duplicates exist)
        # Optional: Decide if we want to deduplicate. Let's keep duplicates for now,
        # as multiple human verifications might be valid over time.
        # self.match_history.sort_values('timestamp', ascending=False, inplace=True)
        # self.match_history.drop_duplicates(subset=['url_404', 'redirect_url'], keep='first', inplace=True)


        log_info(f"Added {len(df_human_history)} new human records. History size now: {len(self.match_history)}.", self.logger)

        # Clean history again in case adding human records pushed it over the limit
        self._clean_history_if_needed()

        # Save the updated history (model is not retrained here)
        self._save_model_and_history() 