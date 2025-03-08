import os
import pandas as pd
import numpy as np
from collections import Counter
import re

class DocumentQuery:
    def __init__(self, csv_path='word_frequencies.csv'):
        """
        Initialize the document query system.
        
        Args:
            csv_path: Path to the CSV file containing word frequencies
        """
        self.csv_path = csv_path
        self.word_frequencies = None
        self.documents = None
        self.stop_words = self.load_stop_words()
        self.load_word_frequencies()
    
    def load_stop_words(self):
        """
        Load a list of common English stop words to filter out from queries.
        """
        #common English stop words
        stop_words = {
            'the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 
            'from', 'up', 'about', 'into', 'over', 'after', 'beneath', 'under', 'above',
            'this', 'that', 'these', 'those', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'than', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'can', 'will', 'just', 'should', 'now'
        }
        return stop_words
    
    def load_word_frequencies(self):
        """
        Load word frequencies from the CSV file.
        """
        try:
            #load CSV file
            self.word_frequencies = pd.read_csv(self.csv_path, index_col=0)
            
            #remove the "TOTAL" row if it exists
            if 'TOTAL' in self.word_frequencies.index:
                self.word_frequencies = self.word_frequencies.drop('TOTAL')
            
            #get document names (column names excluding 'Total')
            self.documents = [col for col in self.word_frequencies.columns if col != 'Total']
            
            print(f"Loaded word frequencies for {len(self.documents)} documents.")
            print(f"Vocabulary size: {len(self.word_frequencies)}")
            
            #fill NaN values with 0
            self.word_frequencies = self.word_frequencies.fillna(0)
            
            #convert to integer counts
            for col in self.word_frequencies.columns:
                self.word_frequencies[col] = self.word_frequencies[col].astype(int)
            
        except FileNotFoundError:
            print(f"Error: Could not find the word frequencies file at {self.csv_path}")
            print("Please run your Assignment 1 program first to generate the CSV file.")
            exit(1)
        except Exception as e:
            print(f"Error loading word frequencies: {str(e)}")
            exit(1)
    
    def clean_query(self, query):
        """
        Clean and tokenize the user's query.
        
        Args:
            query: User's input query string
            
        Returns:
            List of cleaned query words
        """
        #convert to lowercase
        query = query.lower()
        
        #remove special characters and numbers
        query = re.sub(r'[^a-z\s]', ' ', query)
        
        #split into words
        words = query.split()
        
        #remove stop words and keep words with length > 1
        words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        return words
    
    def calculate_document_probabilities(self, query_words):
        """
        Calculate the probability of each document matching the query words.
        Using Naive Bayes approach from Week 4.
        
        Args:
            query_words: List of query words
            
        Returns:
            Dictionary mapping document names to probabilities
        """
        #initialize probabilities (using log probabilities to avoid underflow)
        log_probabilities = {doc: 0 for doc in self.documents}
        
        #for each query word, calculate its contribution to document probabilities
        for word in query_words:
            if word in self.word_frequencies.index:
                for doc in self.documents:
                    #get word count in this document
                    word_count = self.word_frequencies.loc[word, doc]
                    
                    #calculate document's total word count
                    doc_total_words = self.word_frequencies[doc].sum()
                    
                    #calculate probability of word given document (with smoothing)
                    p_word_given_doc = (word_count + 1) / (doc_total_words + len(self.word_frequencies))
                    
                    #add log probability to avoid underflow issues
                    log_probabilities[doc] += np.log(p_word_given_doc)
            else:
                #word not in vocabulary, apply smoothing
                for doc in self.documents:
                    doc_total_words = self.word_frequencies[doc].sum()
                    p_word_given_doc = 1 / (doc_total_words + len(self.word_frequencies))
                    log_probabilities[doc] += np.log(p_word_given_doc)
        
        #convert from log probabilities back to regular probabilities
        probabilities = {doc: np.exp(log_prob) for doc, log_prob in log_probabilities.items()}
        
        return probabilities
    
    def rank_documents(self, query):
        """
        Rank documents based on their relevance to the query.
        
        Args:
            query: User's input query string
            
        Returns:
            List of tuples (document_name, probability) sorted by probability
        """
        #clean and tokenize query
        query_words = self.clean_query(query)
        
        if not query_words:
            print("Warning: Query contains only stop words or invalid characters.")
            return []
        
        print(f"Processed query words: {query_words}")
        
        #calculate document probabilities
        probabilities = self.calculate_document_probabilities(query_words)
        
        #sort documents by probability
        ranked_docs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_docs
    
    def get_top_documents(self, query):
        """
        Get the top-ranked documents for a query.
        
        Args:
            query: User's input query string
            
        Returns:
            List of document names with highest probability
        """
        ranked_docs = self.rank_documents(query)
        
        if not ranked_docs:
            return []
        
        #get the maximum probability
        max_prob = ranked_docs[0][1]
        
        #get all documents with the maximum probability
        top_docs = [doc for doc, prob in ranked_docs if abs(prob - max_prob) < 1e-10]
        
        return top_docs

def main():
    """
    Main function to run the document query system.
    """
    print("\n===== NLP Document Query System =====")
    print("This program will find the most relevant document(s) for your query.")
    
    #initialize the query system
    query_system = DocumentQuery()
    
    while True:
        #get user query
        query = input("\nEnter your query (or 'quit' to exit): ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        #get top documents
        top_docs = query_system.get_top_documents(query)
        
        if top_docs:
            if len(top_docs) == 1:
                print(f"\nThe most relevant document is: {top_docs[0]}")
            else:
                print(f"\nThe most relevant documents are:")
                for doc in top_docs:
                    print(f"- {doc}")
        else:
            print("\nNo relevant documents found for your query.")
        
        #show all rankings for detailed view
        ranked_docs = query_system.rank_documents(query)
        if ranked_docs:
            print("\nDocument rankings (normalized):")
            #normalize probabilities for readability
            max_prob = ranked_docs[0][1]
            for doc, prob in ranked_docs:
                normalized_score = prob / max_prob
                print(f"- {doc}: {normalized_score:.4f}")

if __name__ == "__main__":
    main()