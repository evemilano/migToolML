# URL Migration Tool

A Python tool for URL migration, utilizing various matching algorithms to find the best redirects between 404 URLs and active URLs.

A comprehensive Python-based solution designed for website migration projects and SEO maintenance. This advanced tool employs a combination of sophisticated string matching algorithms, machine learning models, and natural language processing techniques to intelligently map broken URLs (404s) to their most appropriate destinations on a new or restructured website.

The migration tool analyzes URL patterns, content similarities, and structural relationships to generate high-confidence redirect recommendations, dramatically reducing the manual effort required when handling large-scale website reorganizations. By leveraging parallel processing capabilities and supporting multiple languages (English and Italian), it efficiently processes thousands of URLs while maintaining high accuracy through a multi-algorithm consensus approach.

Built for SEO professionals, web administrators, and digital migration teams, this tool helps preserve website authority, maintain user experience, and prevent traffic loss during site migrations by creating intelligent 301 redirect mappings based on both semantic understanding and URL structure analysis.

## Features

- Automatic verification of 404 URLs
- Cleaning and normalization of URLs
- Multiple matching strategies:
  - Fuzzy matching
  - Levenshtein distance
  - Jaccard similarity
  - Hamming distance
  - Ratcliff/Obershelp algorithm
  - Tversky index
  - spaCy similarity
  - Vector similarity (TF-IDF)
  - Jaro-Winkler similarity
  - BERTopic similarity
- **Machine learning capabilities:**
  - ML-based match prediction and recommendation
  - Automated training on historical match data
  - NLP-based content similarity analysis
  - Pre-trained language models for semantic understanding
- Quality metrics calculation
- Export results to Excel
- Detailed logging
- Robust error handling
- Parallel processing options
- Language selection
- Playwright integration for advanced web scraping

## Requirements

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd migTool
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required spaCy models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
```

## Project Structure

```
migTool/
├── config/
│   └── config.py         # Configurations
├── src/
│   ├── __init__.py
│   ├── main.py          # Entry point
│   ├── mig_tool.py      # Main class
│   ├── logger.py        # Logging management
│   ├── input_handler.py # Input management
│   ├── url_handler.py   # URL management
│   ├── matching_algorithms.py  # Matching algorithms
│   ├── ml.py            # Machine learning models
│   ├── excel_formatter.py # Excel formatting
│   └── output_handler.py # Output management
├── data/               # Machine learning models and embeddings
├── embeddings_cache/   # Cache for embeddings
├── tests/              # Unit tests
├── logs/               # Log files
├── input/              # Input files
├── output/             # Output files
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## Usage

1. Prepare the input Excel files:
   - A file with 404 URLs
   - A file with active URLs
   - Each file must contain a column with the URLs

2. Run the tool:
```bash
python src/main.py
```

3. Follow the interactive instructions to:
   - Select the input files
   - Select the Excel sheets
   - Select the columns with the URLs
   - Choose the language for processing
   - Enable or disable parallel processing

4. The tool will process the data and generate:
   - An Excel file with the results
   - Quality metrics
   - Detailed logs

## Configuration

Configurations can be modified in the `config/config.py` file:

- Matching algorithms to use
- Similarity thresholds
- HTTP parameters
- Logging configurations
- Output configurations
- Language selection
- Parallel processing options

## Machine Learning

The tool includes advanced machine learning models for similarity calculations. These models are stored in the `data/` directory and include pre-trained embeddings for enhanced accuracy.

## Output

The tool generates an Excel file with three sheets:

1. **Mapping**: Contains all detailed results
2. **Redirects**: Contains only the recommended redirects
3. **Metrics**: Contains quality metrics

## Logging

Logs are saved in the `logs/` directory with automatic rotation.

## Advanced Options

- **Playwright Integration**: Use Playwright for advanced web scraping and data collection.
- **Embeddings Cache**: The `embeddings_cache/` directory is used to store cached embeddings for faster processing.

## Troubleshooting

- Ensure all dependencies are installed correctly.
- Verify the input files are formatted as required.
- Check the logs in the `logs/` directory for detailed error messages.
- If using Playwright, ensure it is installed and configured properly.

## Contributing

1. Fork the repository
2. Create a branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see the LICENSE file for details.