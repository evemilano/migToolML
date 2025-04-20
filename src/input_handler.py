"""
Input handling and validation for the URL Migration Tool.

This module is responsible for all input-related operations in the URL Migration Tool:

1. File Operations:
   - Discovers Excel files in the input directory
   - Loads and validates Excel files
   - Handles different Excel formats (.xlsx, .xls)

2. User Interaction:
   - Displays available Excel files and sheets
   - Manages user selection of files, sheets, and columns
   - Provides interactive prompts for user choices

3. Data Validation:
   - Validates DataFrame structure and content
   - Ensures required columns are present
   - Handles missing or invalid data

4. Data Preparation:
   - Cleans and standardizes input data
   - Removes duplicates and null values
   - Converts data types as needed
   - Prepares URLs for processing

The InputHandler class centralizes all these operations, making the codebase more
maintainable and ensuring consistent input handling across the application.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from config.config import INPUT_DIR, REQUIRED_COLUMNS
from src.logger import log_error, log_warning, log_info

class InputHandler:
    """Handles input file loading and validation."""
    
    def __init__(self, logger):
        """
        Initialize InputHandler.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.input_dir = INPUT_DIR
        
    def get_excel_files(self) -> List[Path]:
        """
        Get list of Excel files in input directory.
        
        Returns:
            List[Path]: List of Excel file paths
        """
        excel_files = list(self.input_dir.glob("*.xlsx")) + list(self.input_dir.glob("*.xls"))
        if not excel_files:
            log_warning("No Excel files found in input directory", self.logger)
        return excel_files
    
    def display_excel_files(self, files: List[Path]) -> None:
        """
        Display available Excel files.
        
        Args:
            files (List[Path]): List of Excel file paths
        """
        if not files:
            print("No Excel files found in the input directory.")
            return
            
        print("\nAvailable Excel files:")
        for idx, file in enumerate(files, 1):
            print(f"{idx}. {file.name}")
    
    def get_sheet_selection(self, file_path: Path) -> Optional[str]:
        """
        Get user selection for Excel sheet.
        
        Args:
            file_path (Path): Path to Excel file
            
        Returns:
            Optional[str]: Selected sheet name
        """
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names
            
            if not sheets:
                log_error("No sheets found in Excel file", self.logger)
                return None
            
            print("\nAvailable sheets:")
            for idx, sheet in enumerate(sheets, 1):
                print(f"{idx}. {sheet}")
            
            while True:
                try:
                    selection = int(input("\nSelect sheet (1-{}): ".format(len(sheets))))
                    if 1 <= selection <= len(sheets):
                        selected_sheet = sheets[selection - 1]
                        log_info(f"Successfully loaded sheet '{selected_sheet}' from {file_path.name}", self.logger)
                        return selected_sheet
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
                    
        except Exception as e:
            log_error(e, self.logger, f"Error reading Excel file: {file_path}")
            return None
    
    def load_dataframe(self, file_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from Excel file.
        
        Args:
            file_path (Path): Path to Excel file
            sheet_name (str): Name of sheet to load
            
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if df.empty:
                log_error("Empty DataFrame loaded", self.logger)
                return None
            return df
        except Exception as e:
            log_error(e, self.logger, f"Error loading DataFrame from {file_path}")
            return None
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate DataFrame has required columns.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (List[str]): List of required column names
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Convert column names to lowercase for case-insensitive comparison
        df_columns = [col.lower() for col in df.columns]
        required_columns = [col.lower() for col in required_columns]
        
        missing_columns = [col for col in required_columns if col not in df_columns]
        if missing_columns:
            log_warning(f"Missing required columns: {missing_columns}", self.logger)
            print("\nAvailable columns:")
            for idx, col in enumerate(df.columns, 1):
                print(f"{idx}. {col}")
            return False
        return True
    
    def get_column_selection(self, df: pd.DataFrame, prompt: str) -> Optional[str]:
        """
        Get user selection for DataFrame column.
        
        Args:
            df (pd.DataFrame): DataFrame to select from
            prompt (str): Prompt message
            
        Returns:
            Optional[str]: Selected column name
        """
        print(f"\n{prompt}")
        print("\nAvailable columns:")
        for idx, col in enumerate(df.columns, 1):
            print(f"{idx}. {col}")
        
        while True:
            try:
                selection = int(input("\nSelect column (1-{}): ".format(len(df.columns))))
                if 1 <= selection <= len(df.columns):
                    selected_column = df.columns[selection - 1]
                    log_info(f"Selected column: {selected_column}", self.logger)
                    return selected_column
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                log_error(e, self.logger, "Error getting column selection")
                return None
    
    def prepare_dataframe(self, df: pd.DataFrame, url_column: str) -> pd.DataFrame:
        """
        Prepare DataFrame for processing.
        
        Args:
            df (pd.DataFrame): DataFrame to prepare
            url_column (str): Name of URL column
            
        Returns:
            pd.DataFrame: Prepared DataFrame
        """
        try:
            # Keep only URL column
            df = df[[url_column]].copy()
            
            # Rename column to 'URL'
            df = df.rename(columns={url_column: 'URL'})
            
            # Remove rows with missing values
            df = df.dropna()
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Convert URLs to string type
            df['URL'] = df['URL'].astype(str)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            return df
            
        except Exception as e:
            log_error(e, self.logger, "Error preparing DataFrame")
            return df 