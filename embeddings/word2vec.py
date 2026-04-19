import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    """
    Skip-gram Word2Vec model with Negative Sampling.
    """
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Center word embeddings
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        # Context word embeddings
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialization
        initrange = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        """
        input_labels: [batch_size]
        pos_labels: [batch_size]
        neg_labels: [batch_size, num_neg_samples]
        """
        # [batch_size, embed_dim]
        input_vectors = self.in_embed(input_labels)
        # [batch_size, embed_dim]
        pos_vectors = self.out_embed(pos_labels)
        # [batch_size, num_neg_samples, embed_dim]
        neg_vectors = self.out_embed(neg_labels)
        
        # Positive score: dot product of input and positive context
        # [batch_size, 1]
        pos_score = torch.sum(input_vectors * pos_vectors, dim=1).unsqueeze(1)
        
        # Negative scores: dot product of input and negative contexts
        # [batch_size, num_neg_samples]
        neg_score = torch.bmm(neg_vectors, input_vectors.unsqueeze(2)).squeeze(2)
        
        return pos_score, neg_score

    def get_embeddings(self):
        """
        Return the trained center word embeddings.
        """
        return self.in_embed.weight.data.cpu().numpy()