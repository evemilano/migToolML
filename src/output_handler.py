"""
Output handling and export for the URL Migration Tool.

This module handles all output-related operations for the URL Migration Tool, including:
- Generation of Excel reports with multiple sheets (Mapping, Redirects, Metrics)
- Calculation of quality metrics for URL matches
- Auto-saving functionality for periodic backup
- Data preparation and cleaning for export
- Summary statistics generation and display

The module provides the OutputHandler class which manages:
1. Excel file generation with timestamped filenames
2. Multiple sheet exports (Mapping, Redirects, Metrics)
3. Quality metrics calculation including:
   - Total URLs processed
   - Unique 404 URLs and redirects
   - Average agreement scores
   - High confidence matches
4. Data preparation for redirects
5. Auto-save functionality
6. Summary statistics printing
7. Match data processing and cleaning

The output Excel file contains:
- Mapping sheet: Complete matching results with all algorithm scores
- Redirects sheet: Simplified view with best redirect matches
- Metrics sheet: Quality metrics and statistics
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import datetime
from config.config import OUTPUT_DIR, OUTPUT_CONFIG, ALGORITHM_CONFIG
from src.logger import log_error, log_info

class OutputHandler:
    """Handles output generation and export."""
    
    def __init__(self, logger):
        """
        Initialize OutputHandler.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.output_dir = OUTPUT_DIR
        self.config = OUTPUT_CONFIG
        self.algorithm_list = self._get_algorithm_list()
        
    def _get_algorithm_list(self) -> List[str]:
        """
        Extract algorithm list from configuration.
        
        Returns:
            List[str]: List of algorithm names
        """
        # Extract algorithm names from ALGORITHM_CONFIG keys
        # Remove 'use_' prefix and filter for enabled algorithms
        algorithms = [
            key.replace('use_', '') 
            for key, enabled in ALGORITHM_CONFIG.items() 
            if enabled and key.startswith('use_')
        ]
        
        # Special case for 404check which might not be used in scoring
        if '404check' in algorithms:
            algorithms.remove('404check')
    
        # Non convertire più in maiuscolo i nomi degli algoritmi
        # algorithms = [algo.upper() for algo in algorithms]
            
        return algorithms
    
    def generate_output_filename(self, domain=None) -> str:
        """
        Generate output filename with timestamp and optional domain.
        Args: domain (str, optional): Domain name to include in filename
        Returns: str: Generated filename
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        if domain:
            # Clean domain name (remove http://, https://, www., etc.)
            clean_domain = domain.replace('http://', '').replace('https://', '').replace('www.', '')
            # Remove any trailing slashes
            clean_domain = clean_domain.rstrip('/')
            # Replace dots with underscores for better filename compatibility
            clean_domain = clean_domain.replace('.', '_')
            return f"{timestamp}_{clean_domain}_redirect_map.xlsx"
        else:
            return f"{timestamp}_redirect_map.xlsx"
    
    def prepare_redirect_df(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for redirects sheet.
        Args: final_df (pd.DataFrame): Final mapping DataFrame
        Returns: pd.DataFrame: Prepared redirects DataFrame
        """
        columns_to_keep = [
            '404 URL',
            'status_code',
            'TotScore',
            'Agreement',
            'Best redirect',
            'Best redirect TotScore'
        ]
        
        try:
            redirect_df = final_df[columns_to_keep].copy()
            log_info("Successfully prepared redirects DataFrame", self.logger)
            return redirect_df
        except Exception as e:
            log_error(e, self.logger, "Error preparing redirects DataFrame")
            raise
    
    def calculate_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate quality metrics for the results.
        Args: df (pd.DataFrame): Results DataFrame
        Returns: Dict: Dictionary of quality metrics
        """
        try:
            metrics = {
                'total_urls': len(df),
                'unique_404s': df['404 URL'].nunique(),
                'unique_redirects': df['Best redirect'].nunique(),
                'avg_agreement': df['Agreement'].mean(),
                'avg_score': df['TotScore'].mean(),
                'high_confidence_redirects': len(df[df['Agreement'] >= 3]),
                'avg_best_redirect_score': df['Best redirect TotScore'].mean(),
                'total_high_confidence': len(df[df['Best redirect TotScore'] >= 0.8])
            }
            log_info("Successfully calculated quality metrics", self.logger)
            return metrics
        except Exception as e:
            log_error(e, self.logger, "Error calculating quality metrics")
            raise
    
    def export_to_excel(self, mapping_df: pd.DataFrame, redirect_df: pd.DataFrame) -> Optional[Path]:
        """
        Export results to Excel file.
        Args: mapping_df (pd.DataFrame): Mapping DataFrame
              redirect_df (pd.DataFrame): Redirects DataFrame
        Returns: Optional[Path]: Path to exported file
        """
        try:
            # Extract domain from the first 404 URL
            domain = None
            if not mapping_df.empty:
                # Check both possible column names for 404 URLs
                url_column = 'url_404' if 'url_404' in mapping_df.columns else '404 URL'
                if url_column in mapping_df.columns and not mapping_df[url_column].empty:
                    first_url = mapping_df[url_column].iloc[0]
                    from urllib.parse import urlparse
                    domain = urlparse(first_url).netloc
            
            # Generate filename with or without domain
            output_filename = self.generate_output_filename(domain=domain)
            output_path = self.output_dir / output_filename
            
            with pd.ExcelWriter(output_path) as writer:
                # Export mapping sheet
                mapping_df.to_excel(
                    writer,
                    sheet_name=self.config['excel_sheets']['mapping'],
                    index=False
                )
                log_info(f"Exported mapping sheet to {output_filename}", self.logger)
                
                # Export redirects sheet
                redirect_df.to_excel(
                    writer,
                    sheet_name=self.config['excel_sheets']['redirects'],
                    index=False
                )
                log_info(f"Exported redirects sheet to {output_filename}", self.logger)
                
                # Export metrics sheet
                metrics = self.calculate_quality_metrics(mapping_df)
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_excel(
                    writer,
                    sheet_name='Metrics',
                    index=False
                )
                log_info(f"Exported metrics sheet to {output_filename}", self.logger)
            
            return output_path
        except Exception as e:
            log_error(e, self.logger, "Error exporting to Excel")
            return None
    
    def auto_save(self, mapping_df: pd.DataFrame, redirect_df: pd.DataFrame) -> None:
        """
        Auto-save results periodically.
        
        Args:
            mapping_df (pd.DataFrame): Mapping DataFrame
            redirect_df (pd.DataFrame): Redirects DataFrame
        """
        try:
            output_path = self.export_to_excel(mapping_df, redirect_df)
            if output_path:
                log_info(f"Auto-saved results to {output_path}", self.logger)
        except Exception as e:
            log_error(e, self.logger, "Error during auto-save")
    
    def print_summary(self, metrics: Dict) -> None:
        """
        Print summary of results.
        Args: metrics (Dict): Quality metrics
        """
        print("\n=== Results Summary ===")
        print(f"Total URLs processed: {metrics['total_urls']:,}")
        print(f"Unique 404 URLs: {metrics['unique_404s']:,}")
        print(f"Unique redirects: {metrics['unique_redirects']:,}")
        print(f"Average agreement score: {metrics['avg_agreement']:.2f}")
        print(f"Average total score: {metrics['avg_score']:.2f}")
        print(f"High confidence redirects: {metrics['high_confidence_redirects']:,}")
        print(f"Average best redirect score: {metrics['avg_best_redirect_score']:.2f}")
        print(f"Total high confidence matches: {metrics['total_high_confidence']:,}")
        print("=====================")

    def prepare_matches_df(self, matches: List[Dict]) -> pd.DataFrame:
        """
        Convert matches list to a clean DataFrame.
        Args: matches (List[Dict]): List of matches with scores and metadata
        Returns: pd.DataFrame: Clean DataFrame with unpacked matches
        """
        # Initialize empty lists for each column
        data = {
            'url_404': [],
        }
        
        # Add columns for each algorithm
        for algo in self.algorithm_list:
            data[algo] = []
            data[f'{algo}_score'] = []
        
        # Process each match
        for match_dict in matches:
            url_404 = match_dict['url_404']
            match_results = {algo: (None, 0.0) for algo in self.algorithm_list}
            
            # Extract matches for each algorithm
            for algo_name, url, score in match_dict['matches']:
                # Trova il nome dell'algoritmo nella lista, ignorando case sensitivity
                algo_key = algo_name
                if algo_key not in self.algorithm_list:
                    # Prova a cercare l'algoritmo ignorando case
                    algo_matches = [a for a in self.algorithm_list if a.lower() == algo_key.lower()]
                    if algo_matches:
                        algo_key = algo_matches[0]
                    else:
                        # Se non troviamo corrispondenza, lo saltiamo
                        log_error(ValueError(f"Algorithm {algo_name} not found in algorithm list"), 
                                 self.logger, f"Algorithm {algo_name} not found in algorithm list")
                        continue
                
                match_results[algo_key] = (str(url) if url is not None else None, float(score))
            
            # Add data to columns
            data['url_404'].append(url_404)
            for algo in self.algorithm_list:
                if algo in match_results:
                    url, score = match_results[algo]
                    data[algo].append(url)
                    data[f'{algo}_score'].append(score)
                else:
                    # Assicuriamoci che tutte le colonne abbiano lo stesso numero di righe
                    data[algo].append(None)
                    data[f'{algo}_score'].append(0.0)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate total score and agreement
        score_columns = [col for col in df.columns if col.endswith('_score')]
        df['TotScore'] = df[score_columns].sum(axis=1)
        
        # Calculate agreement (number of algorithms suggesting the same URL)
        def count_agreement(row):
            urls = [row[algo] for algo in self.algorithm_list if pd.notna(row[algo])]
            if not urls:
                return 0, None, 0.0
            url_counts = {url: urls.count(url) for url in set(urls)}
            best_redirect = max(url_counts, key=url_counts.get)
            max_count = url_counts[best_redirect]
            
            # Calculate total score for best redirect
            total_score = sum(
                row[f'{algo}_score'] 
                for algo in self.algorithm_list
                if pd.notna(row[algo]) and row[algo] == best_redirect
            )
            
            return max_count, best_redirect, total_score
        
        agreements = df.apply(count_agreement, axis=1)
        df['Agreement'], df['Best redirect'], df['Best redirect TotScore'] = zip(*agreements)
        
        return df

    def generate_output(self, matches: List[Dict]) -> Optional[Path]:
        """
        Generate output files from matches.
        Args: 
            matches (List[Dict]): List of matches with scores and metadata
        Returns:
            Optional[Path]: Path to the created Excel file
        """
        try:
            if not matches:
                self.logger.warning("No matches found to generate output")
                return None
                
            # Create clean DataFrame from matches
            final_mig_df = self.prepare_matches_df(matches)
            
            # Sort by Best redirect TotScore
            final_mig_df = final_mig_df.sort_values(by='Best redirect TotScore', ascending=False)


            ### plug ml.py
            # AI/ML integration: Add ML predictions and update scores
            try:
                # Initialize AIMatchingAlgorithm
                from src.ml import AIMatchingAlgorithm
                ai_matcher = AIMatchingAlgorithm(self.logger)
                
                log_info("Applying AI matching algorithm to results", self.logger)
                
                # Check if 'ml' column already exists (lowercase version)
                if 'ml' in final_mig_df.columns and 'ml_score' in final_mig_df.columns:
                    # Use existing ml/ml_score columns instead of creating new ones
                    log_info("Using existing ml columns", self.logger)
                else:
                    # Initialize ml and ml_score columns with None and 0.0
                    final_mig_df['ml'] = None
                    final_mig_df['ml_score'] = 0.0
                
                # Process each 404 URL to find ML-based match
                url_404_list = final_mig_df['url_404'].unique().tolist()
                for url_404 in url_404_list:
                    # Get all candidate URLs from other algorithms 
                    # (only include valid URLs with non-zero scores)
                    candidate_urls = []
                    for algo in self.algorithm_list:
                        if algo.lower() != 'ml':  # Skip ML algorithm itself
                            urls = final_mig_df.loc[final_mig_df['url_404'] == url_404, algo].tolist()
                            urls = [url for url in urls if pd.notna(url) and url]
                            candidate_urls.extend(urls)
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    candidate_urls = [x for x in candidate_urls if not (x in seen or seen.add(x))]
                    
                    if candidate_urls and ai_matcher.model_initialized:
                        # Get ML prediction
                        best_match, confidence_score = ai_matcher.find_best_match(url_404, candidate_urls)
                        
                        # Update the DataFrame with ML prediction
                        if best_match is not None:
                            final_mig_df.loc[final_mig_df['url_404'] == url_404, 'ml'] = best_match
                            final_mig_df.loc[final_mig_df['url_404'] == url_404, 'ml_score'] = confidence_score
                
                # Re-calculate total score and agreement including ML
                # Only include columns ending with '_score' and exclude any uppercase 'ML_score'
                score_columns = [col for col in final_mig_df.columns if col.endswith('_score') and col != 'ML_score']
                final_mig_df['TotScore'] = final_mig_df[score_columns].sum(axis=1)
                
                # Recalculate agreement (number of algorithms suggesting the same URL)
                def count_agreement(row):
                    # Use only existing algorithms in self.algorithm_list and ensure no duplicates
                    algo_list = [algo.lower() for algo in self.algorithm_list]
                    urls = [row[algo] for algo in algo_list if algo in final_mig_df.columns and pd.notna(row[algo])]
                    if not urls:
                        return 0, None, 0.0
                    url_counts = {url: urls.count(url) for url in set(urls)}
                    best_redirect = max(url_counts, key=url_counts.get)
                    max_count = url_counts[best_redirect]
                    
                    # Calculate total score for best redirect
                    total_score = sum(
                        row[f'{algo}_score'] 
                        for algo in algo_list
                        if f'{algo}_score' in final_mig_df.columns and pd.notna(row[algo]) and row[algo] == best_redirect
                    )
                    
                    return max_count, best_redirect, total_score
                
                # Apply recalculation
                agreements = final_mig_df.apply(count_agreement, axis=1)
                final_mig_df['Agreement'], final_mig_df['Best redirect'], final_mig_df['Best redirect TotScore'] = zip(*agreements)
                
                # Update ML model from matches (continuous learning)
                ai_matcher.update_from_matches(matches, final_mig_df)
                
                log_info("AI matching algorithm applied successfully", self.logger)
                
            except Exception as e:
                log_error(e, self.logger, "Error applying AI matching algorithm")
                # Ensure the code continues even if ML fails
                pass
            ### enf plug.py

            # Prepare redirects DataFrame (moved after ML processing)
            columns_to_keep = [
                'url_404',
                'TotScore',
                'Agreement',
                'Best redirect',
                'Best redirect TotScore'
            ]
            redirect_df = final_mig_df[columns_to_keep].copy()
            
            # Sort both DataFrames by Best redirect TotScore
            final_mig_df = final_mig_df.sort_values(by='Best redirect TotScore', ascending=False)
            redirect_df = redirect_df.sort_values(by='Best redirect TotScore', ascending=False)
            

             # Estrai il dominio dal primo URL 404
            domain = None
            if not final_mig_df.empty:
                if 'url_404' in final_mig_df.columns and len(final_mig_df['url_404']) > 0:
                    first_url = final_mig_df['url_404'].iloc[0]
                    from urllib.parse import urlparse
                    domain = urlparse(first_url).netloc
    
            # Generate timestamp for filename
            output_filename = self.generate_output_filename(domain=domain)
            output_path = self.output_dir / output_filename
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_path) as writer:
                final_mig_df.to_excel(writer, sheet_name='Mapping', index=False)
                redirect_df.to_excel(writer, sheet_name='Redirects', index=False)
                # Add algorithm stats sheet
                algo_stats_df = self.prepare_algorithm_stats_df(final_mig_df)
                algo_stats_df.to_excel(writer, sheet_name='Algorithm Stats', index=False)


            # Print summary
            print(f"\nFound {len(matches)} matches")
            print(f"Results saved to: {output_path}")
            
            # Return the path to the created file
            return output_path
            
        except Exception as e:
            log_error(e, self.logger, "Error generating output")
            raise



        
    # aggiungito per il calcolo delle statistiche degli algoritmi
    def prepare_algorithm_stats_df(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a DataFrame showing how many times each algorithm's URL matched the Best redirect URL.
        
        Args:
            final_df (pd.DataFrame): Final mapping DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with algorithm stats
        """
        try:
            # Get the Best redirect column
            best_redirects = final_df["Best redirect"]
            
            # Dictionary to store algorithm stats
            algo_stats = {}
            
            # Find all algorithm columns (excluding score columns and other non-algorithm columns)
            excluded_keywords = ["score", "agreement", "totscore", "best redirect", "status", "count", "url_404"]
            algorithm_columns = []
            
            for col in final_df.columns:
                if not any(keyword in col.lower() for keyword in excluded_keywords):
                    algorithm_columns.append(col)
            
            # Count matches for each algorithm
            for algo_col in algorithm_columns:
                # Count how many times this algorithm's URL matches the Best redirect
                matches = sum(final_df[algo_col] == best_redirects)
                algo_stats[algo_col] = matches
            
            # Create DataFrame with algorithm stats
            stats_df = pd.DataFrame({
                'Algorithm': list(algo_stats.keys()),
                'Best Redirect Matches': list(algo_stats.values())
            })
            
            # Sort by number of matches in descending order
            stats_df = stats_df.sort_values('Best Redirect Matches', ascending=False).reset_index(drop=True)
            
            # Add percentage column
            total_urls = len(best_redirects[best_redirects.notna()])
            if total_urls > 0:
                stats_df['Percentage'] = (stats_df['Best Redirect Matches'] / total_urls * 100).round(2).astype(str) + '%'
            else:
                stats_df['Percentage'] = '0%'
                
            log_info("Successfully prepared algorithm stats DataFrame", self.logger)
            return stats_df
            
        except Exception as e:
            log_error(e, self.logger, "Error preparing algorithm stats DataFrame")
            raise