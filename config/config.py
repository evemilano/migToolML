"""
Configuration settings for the URL Migration Tool.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# Algorithm settings
ALGORITHM_CONFIG = {
    'use_404check': True,
    'use_fuzzy': True,
    'use_levenshtein': True,
    'use_jaccard': True,
    'use_hamming': True,
    'use_ratcliff': True,
    'use_tversky': True,
    'use_spacy': True,
    'use_vector': True,
    'use_jaro_winkler': True,
    'use_bertopic': True,
    'use_ml': True,  # Enable ML matching algorithm
}

# HTTP settings
HTTP_CONFIG = {
    'timeout': 30,
    'pause': 1,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'parallel_threads': 1,  # Default to 1 thread
    'batch_size': 10  # Default batch size
}

# Similarity thresholds
SIMILARITY_CONFIG = {
    'min_similarity_threshold': 0.7,
    'fuzzy_threshold': 50,
    'jaccard_threshold': 0.5,
    'hamming_threshold': 0.8,
    'ratcliff_threshold': 0.6,
    'tversky_alpha': 0.5,
    'tversky_beta': 0.5,
    'spacy_threshold': 0.4,  # Previously 0.7, then 0.5
    'vector_threshold': 0.6,
    'jaro_winkler_threshold': 0.8,
    'bertopic_threshold': 0.4,  # Previously 0.7, then 0.5
    'ml_threshold': 0.7  # Threshold for ML algorithm
}

# Logging configuration
LOG_CONFIG = {
    'log_file': LOG_DIR / 'migTool.log',
    'max_bytes': 1024 * 1024,  # 1MB
    'backup_count': 5,
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Output configuration
OUTPUT_CONFIG = {
    'auto_save_interval': 300,  # 5 minutes
    'excel_sheets': {
        'mapping': 'Mapping',
        'redirects': 'Redirects'
    }
}

# Required columns for input files
REQUIRED_COLUMNS = {
    '404_file': ['URL'],
    'live_file': ['URL']
} 