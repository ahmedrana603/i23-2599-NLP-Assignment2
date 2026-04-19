import numpy as np
import math

class TFIDF:
    """
    Implementation of TF-IDF (Term Frequency-Inverse Document Frequency) from scratch.
    """
    def __init__(self, word2idx=None):
        self.word2idx = word2idx if word2idx is not None else {}
        self.idx2word = {i: w for w, i in self.word2idx.items()} if self.word2idx else {}
        self.idf = None
        self.vocab_size = len(self.word2idx)

    def fit_transform(self, corpus):
        """
        Fits to the corpus and returns the TF-IDF matrix.
        """
        self.fit(corpus)
        return self.transform(corpus)

    def fit(self, corpus):
        """
        Calculate IDF for each word in the corpus.
        Args:
            corpus: List of list of strings (tokenized documents)
        """
        N = len(corpus)
        if not self.word2idx:
            unique_words = sorted(list(set([word for doc in corpus for word in doc])))
            self.word2idx = {word: i for i, word in enumerate(unique_words)}
            self.idx2word = {i: word for word, i in self.word2idx.items()}
            self.vocab_size = len(unique_words)
        
        # Calculate Document Frequency (DF)
        df_counts = np.zeros(self.vocab_size)
        for doc in corpus:
            doc_unique_words = set(doc)
            for word in doc_unique_words:
                if word in self.word2idx:
                    df_counts[self.word2idx[word]] += 1
        
        # Calculate IDF: log(N / (1 + df)) as per user request
        self.idf = np.log(N / (1 + df_counts))

    def transform(self, corpus):
        """
        Transform a corpus into TF-IDF vectors.
        Args:
            corpus: List of list of strings (tokenized documents)
        Returns:
            tfidf_matrix: np.ndarray of shape (num_docs, vocab_size)
        """
        num_docs = len(corpus)
        tfidf_matrix = np.zeros((num_docs, self.vocab_size))
        
        for i, doc in enumerate(corpus):
            # Calculate Term Frequency (TF) - Raw count
            tf = np.zeros(self.vocab_size)
            for word in doc:
                if word in self.word2idx:
                    tf[self.word2idx[word]] += 1
            
            # TF-IDF = TF * IDF
            tfidf_matrix[i] = tf * self.idf
            
        return tfidf_matrix