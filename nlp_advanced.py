import os
import pandas as pd
import numpy as np
from collections import Counter
import re
import math

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
        self.tfidf_matrix = None
        self.ppmi_matrix = None
        self.load_word_frequencies()
        self.calculate_tfidf()
        self.calculate_ppmi()
    
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
    
    def calculate_tfidf(self):
        """
        Calculate TF-IDF (Term Frequency-Inverse Document Frequency) matrix.
        
        TF-IDF measures the importance of a word to a document in a corpus:
        - TF (term frequency): How frequently a word appears in a document
        - IDF (inverse document frequency): How rare a word is across all documents
        """
        #create a copy of the word frequencies DataFrame
        self.tfidf_matrix = pd.DataFrame(index=self.word_frequencies.index, columns=self.documents)
        
        #calculate IDF for each term
        N = len(self.documents)  # Number of documents
        idf = {}
        
        for word in self.word_frequencies.index:
            #count documents containing this word
            doc_count = sum(1 for doc in self.documents if self.word_frequencies.loc[word, doc] > 0)
            #calculate IDF with smoothing
            idf[word] = math.log((N + 1) / (doc_count + 1)) + 1
        
        #calculate TF-IDF
        for doc in self.documents:
            #get total words in document for TF normalization
            doc_total = self.word_frequencies[doc].sum()
            
            for word in self.word_frequencies.index:
                #term frequency (normalized by document length)
                tf = self.word_frequencies.loc[word, doc] / doc_total if doc_total > 0 else 0
                #TF-IDF score
                self.tfidf_matrix.loc[word, doc] = tf * idf[word]
    
    def calculate_ppmi(self):
        """
        Calculate PPMI (Positive Pointwise Mutual Information) matrix.
        
        PPMI measures the association between a word and document:
        PMI(w,d) = log(P(w,d) / (P(w) * P(d)))
        PPMI = max(0, PMI)
        """
        #create a new DataFrame for PPMI scores
        self.ppmi_matrix = pd.DataFrame(index=self.word_frequencies.index, columns=self.documents)
        
        #calculate marginal probabilities
        total_words = self.word_frequencies.values.sum()
        
        #P(w) - probability of word across all documents
        p_word = {}
        for word in self.word_frequencies.index:
            p_word[word] = self.word_frequencies.loc[word].sum() / total_words
        
        #P(d) - probability of document (proportion of corpus)
        p_doc = {}
        for doc in self.documents:
            p_doc[doc] = self.word_frequencies[doc].sum() / total_words
        
        #calculate joint probabilities and PPMI
        for word in self.word_frequencies.index:
            for doc in self.documents:
                #P(w,d) - joint probability
                p_word_doc = self.word_frequencies.loc[word, doc] / total_words
                
                #PMI calculation with smoothing
                if p_word_doc > 0:
                    pmi = math.log2((p_word_doc + 1e-10) / ((p_word[word] + 1e-10) * (p_doc[doc] + 1e-10)))
                    #PPMI - take only positive values
                    ppmi = max(0, pmi)
                else:
                    ppmi = 0
                
                self.ppmi_matrix.loc[word, doc] = ppmi
    
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
    
    def rank_documents_naive_bayes(self, query_words):
        """
        Rank documents using Naive Bayes approach.
        
        Args:
            query_words: List of query words
            
        Returns:
            Dictionary of document probabilities
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
    
    def rank_documents_tfidf(self, query_words):
        """
        Rank documents using TF-IDF scores.
        
        Args:
            query_words: List of query words
            
        Returns:
            Dictionary of document relevance scores based on TF-IDF
        """
        #initialize scores
        scores = {doc: 0 for doc in self.documents}
        
        #for each query word, add its TF-IDF score to the document's total
        for word in query_words:
            if word in self.tfidf_matrix.index:
                for doc in self.documents:
                    scores[doc] += self.tfidf_matrix.loc[word, doc]
        
        return scores
    
    def rank_documents_ppmi(self, query_words):
        """
        Rank documents using PPMI scores.
        
        Args:
            query_words: List of query words
            
        Returns:
            Dictionary of document relevance scores based on PPMI
        """
        #initialize scores
        scores = {doc: 0 for doc in self.documents}
        
        #for each query word, add its PPMI score to the document's total
        for word in query_words:
            if word in self.ppmi_matrix.index:
                for doc in self.documents:
                    scores[doc] += self.ppmi_matrix.loc[word, doc]
        
        return scores
    
    def rank_documents(self, query, method='combined'):
        """
        Rank documents based on their relevance to the query using the specified method.
        
        Args:
            query: User's input query string
            method: Ranking method - 'naive_bayes', 'tfidf', 'ppmi', or 'combined'
            
        Returns:
            List of tuples (document_name, score) sorted by score
        """
        #clean and tokenize query
        query_words = self.clean_query(query)
        
        if not query_words:
            print("Warning: Query contains only stop words or invalid characters.")
            return []
        
        print(f"Processed query words: {query_words}")
        
        #calculate document scores based on the specified method
        if method == 'naive_bayes':
            scores = self.rank_documents_naive_bayes(query_words)
        elif method == 'tfidf':
            scores = self.rank_documents_tfidf(query_words)
        elif method == 'ppmi':
            scores = self.rank_documents_ppmi(query_words)
        else:  # 'combined' - use all methods and normalize
            nb_scores = self.rank_documents_naive_bayes(query_words)
            tfidf_scores = self.rank_documents_tfidf(query_words)
            ppmi_scores = self.rank_documents_ppmi(query_words)
            
            #normalize each set of scores
            def normalize_scores(scores):
                max_score = max(scores.values()) if scores and max(scores.values()) > 0 else 1
                return {doc: (score / max_score if max_score > 0 else 0) for doc, score in scores.items()}
            
            nb_norm = normalize_scores(nb_scores)
            tfidf_norm = normalize_scores(tfidf_scores)
            ppmi_norm = normalize_scores(ppmi_scores)
            
            #combine normalized scores (weighted average)
            scores = {}
            for doc in self.documents:
                scores[doc] = (
                    0.2 * nb_norm[doc] + 
                    0.4 * tfidf_norm[doc] + 
                    0.4 * ppmi_norm[doc]
                )
        
        #sort documents by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_docs
    
    def get_top_documents(self, query, method='combined'):
        """
        Get the top-ranked documents for a query.
        
        Args:
            query: User's input query string
            method: Ranking method - 'naive_bayes', 'tfidf', 'ppmi', or 'combined'
            
        Returns:
            List of document names with highest scores
        """
        ranked_docs = self.rank_documents(query, method)
        
        if not ranked_docs:
            return []
        
        #get the maximum score
        max_score = ranked_docs[0][1]
        
        #get all documents with the maximum score (allowing for ties)
        top_docs = [doc for doc, score in ranked_docs if abs(score - max_score) < 1e-10]
        
        return top_docs

def main():
    """
    Main function to run the document query system.
    """
    print("\n===== NLP Document Query System =====")
    print("This program will find the most relevant document(s) for your query.")
    print("Using TF-IDF and PPMI for enhanced ranking as specified in Chapter 6.")
    
    #initialize the query system
    query_system = DocumentQuery()
    
    while True:
        #get user query
        query = input("\nEnter your query (or 'quit' to exit): ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        #get top documents using the combined method
        top_docs = query_system.get_top_documents(query, method='combined')
        
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
        ranked_docs = query_system.rank_documents(query, method='combined')
        if ranked_docs:
            print("\nDocument rankings (normalized, using combined TF-IDF and PPMI):")
            #normalize scores for readability
            max_score = ranked_docs[0][1]
            for doc, score in ranked_docs:
                normalized_score = score / max_score
                print(f"- {doc}: {normalized_score:.4f}")
            
            #show individual method rankings for comparison
            print("\nComparison of ranking methods:")
            
            methods = ['naive_bayes', 'tfidf', 'ppmi']
            for method in methods:
                ranked_by_method = query_system.rank_documents(query, method)
                if ranked_by_method:
                    print(f"\n{method.upper()} ranking:")
                    max_method_score = ranked_by_method[0][1] if ranked_by_method[0][1] > 0 else 1
                    for doc, score in ranked_by_method:
                        norm_score = score / max_method_score if max_method_score > 0 else 0
                        print(f"- {doc}: {norm_score:.4f}")

if __name__ == "__main__":
    main()