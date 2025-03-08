# NLP Document Query System
This repository contains two implementations of an NLP document query system that processes PDF documents and ranks them based on relevance to user queries.

It also requires previously generated document from https://github.com/shefreenkaur/Web-Scraping-and-Word-Frequencies. 
However, for this part of the assignment I used different pdf files.


## Program Versions

1. **Basic Version (`nlp_query.py`)**: 
   - Implements Naive Bayes ranking for document retrieval
   - Simple and efficient implementation

2. **Advanced Version (`nlp_advanced.py`)**:
   - Implements multiple ranking methods including TF-IDF and PPMI as specified in Chapter 6
   - Provides comparison between different ranking approaches
   - More comprehensive analysis of document relevance

## Environment

- **OS Platform**: Cross-platform (Windows, macOS, Linux)
- **Python Version**: 3.8+ recommended
- **Folder Structure**:
  - `nlp_query.py`: Basic document query program
  - `nlp_advanced.py`: Enhanced program with TF-IDF and PPMI
  - `word_frequencies.csv`: CSV file containing word frequencies (generated from Assignment 1)
  - `/data`: Subfolder containing PDF files to be analyzed.

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install pandas numpy
```

Note: The original word frequency program from Assignment 1 requires additional dependencies
Please see attached link for separate repository:
```bash
pip install easyocr PyMuPDF
```

## Usage

1. Make sure you have the `word_frequencies.csv` file:
   - Place PDF files in the `data` subfolder
   - Run the word frequency program to generate the CSV

2. Run either version of the query program:
```bash
python nlp_query.py     # Basic version
python nlp_advanced.py  # Advanced version with TF-IDF and PPMI
```

3. Enter your query when prompted. The program will return the most relevant document(s) based on your topic words.

## NLP Strategy

### Document Parsing
- Both programs utilize the word frequency data generated from Assignment 1
- PDF parsing was performed using EasyOCR and PyMuPDF libraries
- Text was extracted from each page and combined for frequency analysis

### Vocabulary Management
- **Prioritized Words**: 
  - Words are ranked by their total frequency across all documents
  - Single character words are removed (typically not meaningful)
  - Special characters and numbers are filtered out
  
- **Stop Words Removal**:
  - Common English stop words are removed from queries
  - This improves query relevance by focusing on content-bearing words
  - The stop word list includes common articles, prepositions, and auxiliary verbs

- **Curation Strategy**:
  - Case normalization (all lowercase) to prevent duplication
  - Single-character words are removed as they typically don't add meaning
  - Special characters and numbers are filtered to focus on alphabetic words
  - Queries are processed using the same cleaning strategy for consistency

### Document Ranking Methods

#### Basic Version (nlp_query.py)
- **Naive Bayes**:
  - Calculates the conditional probability of a document given the query words
  - Uses smoothing to handle words not present in documents
  - Implements log probabilities to prevent numerical underflow with multiple terms

#### Advanced Version (nlp_advanced.py)
Implements three advanced document ranking methods:

1. **Naive Bayes** (same as basic version)

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   - Term Frequency: How frequently a word appears in a document (normalized by document length)
   - Inverse Document Frequency: How rare a word is across all documents
   - TF-IDF score measures how important a word is to a specific document in the corpus
   - Higher scores indicate words that are frequent in a document but rare across the corpus

3. **PPMI (Positive Pointwise Mutual Information)**:
   - Measures the strength of association between a word and a document
   - Calculated as log(P(word,doc) / (P(word) * P(doc)))
   - Positive values indicate words that appear more often in a document than would be expected by chance
   - Negative values are set to zero (hence "Positive" PMI)

4. **Combined Method**:
   - A weighted combination of the three methods above
   - Provides a balanced ranking that leverages the strengths of each method

### User Interaction
- Both programs accept free text queries from the user
- Queries are automatically cleaned and tokenized
- The basic version shows the most relevant document(s) and a normalized ranking
- The advanced version additionally shows rankings from each individual method for comparison

## Implementation Details

Both programs are organized into a reusable `DocumentQuery` class that:
- Loads word frequencies from the CSV file
- Processes user queries
- Calculates document relevance scores
- Returns ranked results

All paths in the code are relative to the parent project folder, making it easy to run in any environment.
