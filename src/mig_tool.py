"""
URL Migration Tool - Main Module

This script implements a tool for migrating URLs from a source system to a target system.
It helps identify and match broken URLs (404) with their correct counterparts in the target system.

Key Features:
- Loads and processes Excel files containing source and target URLs
- Verifies 404 URLs to ensure they are actually broken
- Implements multiple matching algorithms to find the best URL matches
- Supports parallel processing for improved performance
- Generates detailed output reports of matched URLs

The tool works in the following steps:
1. Loads input data from Excel files (source URLs with 404s and target live URLs)
2. Verifies the status of 404 URLs to ensure they are actually broken
3. Applies matching algorithms to find the best matches between broken and live URLs
4. Generates output reports with the matching results

Dependencies:
- pandas: For data manipulation and Excel file handling
- httpx: For URL verification
- joblib: For parallel processing
- config: Custom configuration module
- Various custom modules (input_handler, url_handler, matching_algorithms, output_handler)

Usage:
    mig_tool = MigTool(language='en', n_jobs=-1)
    mig_tool.run()
"""
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path
from config.config import ALGORITHM_CONFIG, REQUIRED_COLUMNS
from src.logger import setup_logger, log_error, log_info
from src.input_handler import InputHandler
from src.url_handler import URLHandler
from src.matching_algorithms import MatchingAlgorithms
from src.output_handler import OutputHandler

# classe principale per lo strumento di migrazione URL
class MigTool:
    """Main class for URL migration tool."""
    # __init__ serve a inizializzare la classe
    # inizializzare una classe serve a creare un oggetto di quella classe
    def __init__(self, language: str, n_jobs: int = -1):
        """
        Initialize MigTool.
        
        Args:
            language (str): Selected language ('it' or 'en')
            n_jobs (int): Number of parallel jobs (-1 for all cores)
        """
        # inizializzazione delle variabili
        self.logger = setup_logger()
        self.input_handler = InputHandler(self.logger)
        self.url_handler = URLHandler(self.logger)
        self.matching_algorithms = MatchingAlgorithms(self.logger, language, n_jobs=n_jobs)
        self.output_handler = OutputHandler(self.logger)
        self.algorithm_config = ALGORITHM_CONFIG

    # load_input_data serve a caricare i dati di input
    # caricare file excel, selezionare i file, caricare i dati, selezionare le colonne 
    def load_input_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load and validate input data.
        
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: (404_df, live_df)
        """
        try:
            # Get Excel files
            excel_files = self.input_handler.get_excel_files()
            if not excel_files:
                raise ValueError("No Excel files found in input directory")
            
            # Display files and get user selection
            self.input_handler.display_excel_files(excel_files)
            
            # Load 404 URLs
            print("\nSelect file with 404 URLs:")
            while True:
                try:
                    selection = int(input(f"Select file (1-{len(excel_files)}): "))
                    if 1 <= selection <= len(excel_files):
                        file_404 = excel_files[selection - 1]
                        break
                    print(f"Please enter a number between 1 and {len(excel_files)}")
                except ValueError:
                    print("Please enter a valid number")
            
            sheet_404 = self.input_handler.get_sheet_selection(file_404)
            if not sheet_404:
                raise ValueError("No sheet selected for 404 URLs")
            
            df_404 = self.input_handler.load_dataframe(file_404, sheet_404)
            if df_404 is None:
                raise ValueError("Failed to load 404 URLs")
            
            # Get column selection for 404 URLs
            col_404 = self.input_handler.get_column_selection(df_404, "Select column with 404 URLs")
            if not col_404:
                raise ValueError("No column selected for 404 URLs")
            
            # Prepare 404 DataFrame
            df_404 = self.input_handler.prepare_dataframe(df_404, col_404)
            
            # Load live URLs
            print("\nSelect file with live URLs:")
            self.input_handler.display_excel_files(excel_files)
            while True:
                try:
                    selection = int(input(f"Select file (1-{len(excel_files)}): "))
                    if 1 <= selection <= len(excel_files):
                        file_live = excel_files[selection - 1]
                        break
                    print(f"Please enter a number between 1 and {len(excel_files)}")
                except ValueError:
                    print("Please enter a valid number")
            
            sheet_live = self.input_handler.get_sheet_selection(file_live)
            if not sheet_live:
                raise ValueError("No sheet selected for live URLs")
            
            df_live = self.input_handler.load_dataframe(file_live, sheet_live)
            if df_live is None:
                raise ValueError("Failed to load live URLs")
            
            # Get column selection for live URLs
            col_live = self.input_handler.get_column_selection(df_live, "Select column with live URLs")
            if not col_live:
                raise ValueError("No column selected for live URLs")
            
            # Prepare live DataFrame
            df_live = self.input_handler.prepare_dataframe(df_live, col_live)
            
            return df_404, df_live
            
        except Exception as e:
            log_error(e, self.logger, "Error loading input data")
            return None, None
    
    # verify_404_urls serve a verificare gli URL 404
    def verify_404_urls(self, df_404: pd.DataFrame) -> pd.DataFrame:
        """
        Verify 404 URLs.
        
        Args:
            df_404 (pd.DataFrame): DataFrame with 404 URLs
            
        Returns:
            pd.DataFrame: Updated DataFrame with status codes
        """
        try:
            if not self.algorithm_config['use_404check']:
                log_info("Skipping 404 verification", self.logger)
                return df_404
            
            urls = df_404['URL'].tolist()
            results = self.url_handler.verify_urls(urls)
            
            # Update DataFrame with status codes
            status_dict = dict(results)
            df_404['status_code'] = df_404['URL'].map(status_dict)
            
            # Filter problematic URLs
            problematic_codes = [403, 430, 500, 501, 502, 503, 504, 'timeout', 'error', 'invalid']
            problematic_urls = self.url_handler.filter_problematic_urls(results, problematic_codes)
            
            if problematic_urls:
                log_info(f"Retrying {len(problematic_urls)} problematic URLs", self.logger)
                updated_results = self.url_handler.retry_problematic_urls(problematic_urls)
                
                # Update status codes for retried URLs
                for url, status in updated_results:
                    df_404.loc[df_404['URL'] == url, 'status_code'] = status
            
            # Keep only 404 URLs and URLs with "httpx error"
            df_404 = df_404[(df_404['status_code'] == 404) | (df_404['status_code'] == "httpx error")]
            
            return df_404
            
        except Exception as e:
            log_error(e, self.logger, "Error verifying 404 URLs")
            raise
    
    # run serve a eseguire lo strumento di migrazione URL
    def run(self):
        """
        Run the URL migration tool.
        
        Returns:
            Optional[Path]: Path to the created Excel file
        """
        try:
            # Load input data
            df_404, df_live = self.load_input_data()
            if df_404 is None or df_live is None:
                raise ValueError("Failed to load input data")
            
            # Verify 404 URLs
            df_404 = self.verify_404_urls(df_404)
            
            # Find matches
            matches = self.matching_algorithms.find_matches(df_404, df_live)
            
            # Generate output
            excel_path = self.output_handler.generate_output(matches)
            
            # Return the path to the created Excel file
            return excel_path
            
        except Exception as e:
            log_error(e, self.logger, "Error running URL migration tool")
            raise 