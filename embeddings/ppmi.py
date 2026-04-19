import numpy as np

class PPMI:
    """
    Implementation of Positive Pointwise Mutual Information (PPMI) from scratch.
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.co_occurrence_matrix = np.zeros((vocab_size, vocab_size))
        self.ppmi_matrix = None

    def build_co_occurrence(self, corpus, window_size=5):
        """
        Build the co-occurrence matrix from a tokenized corpus (list of indices).
        Args:
            corpus: List of word indices.
            window_size: Number of context words on each side.
        """
        for i, center_idx in enumerate(corpus):
            # Context window boundaries
            start = max(0, i - window_size)
            end = min(len(corpus), i + window_size + 1)
            
            for j in range(start, end):
                if i == j:
                    continue
                context_idx = corpus[j]
                self.co_occurrence_matrix[center_idx][context_idx] += 1

    def calculate_ppmi(self, epsilon=1e-9):
        """
        Convert co-occurrence matrix to PPMI matrix.
        PPMI = max(0, log2(P(w1,w2)/(P(w1)*P(w2))))
        """
        # Sum of all co-occurrences
        total_sum = np.sum(self.co_occurrence_matrix)
        
        # Marginal sums P(w1), P(w2)
        row_sums = np.sum(self.co_occurrence_matrix, axis=1)
        col_sums = np.sum(self.co_occurrence_matrix, axis=0)
        
        # Avoid division by zero
        row_sums[row_sums == 0] = epsilon
        col_sums[col_sums == 0] = epsilon
        
        # PPMI calculation: total_sum * C(w1,w2) / (C(w1)*C(w2))
        # Note: Using outer product for vectorized denominator
        denominator = np.outer(row_sums, col_sums)
        
        # Calculate PMI
        pmi = np.log2((self.co_occurrence_matrix * total_sum) / denominator + epsilon)
        
        # PPMI = max(0, PMI)
        self.ppmi_matrix = np.maximum(0, pmi)
        return self.ppmi_matrix

def cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors from scratch.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)