import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMEncoder(nn.Module):
    """
    Core 2-layer BiLSTM encoder.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           bidirectional=True, batch_first=True, dropout=0.2)

    def forward(self, x):
        embedded = self.embedding(x)
        # out shape: [batch, seq_len, hidden_dim * 2]
        out, _ = self.lstm(embedded)
        return out

class POSModel(nn.Module):
    """
    POS Tagger: BiLSTM + Linear + Softmax.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, target_size):
        super(POSModel, self).__init__()
        self.encoder = BiLSTMEncoder(vocab_size, embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim * 2, target_size)

    def forward(self, x):
        feats = self.encoder(x)
        tag_space = self.hidden2tag(feats)
        return F.log_softmax(tag_space, dim=2)

class CRF(nn.Module):
    """
    Manual CRF layer implementation for NER.
    """
    def __init__(self, target_size):
        super(CRF, self).__init__()
        self.target_size = target_size
        # Transition matrix: transitions[i][j] is score from j to i
        self.transitions = nn.Parameter(torch.randn(target_size, target_size))

    def _forward_alg(self, feats):
        # Calculate the partition function (sum of scores of all possible paths)
        # Using log-sum-exp trick for stability
        # feats: [seq_len, target_size]
        init_alphas = torch.full((1, self.target_size), -10000.)
        init_alphas[0][0] = 0. # Start state
        
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.target_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.target_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        
        terminal_var = forward_var
        return torch.logsumexp(terminal_var, dim=1)[0]

    def _viterbi_decode(self, feats):
        # Implementation of Viterbi to find most likely path
        backpointers = []
        init_vvars = torch.full((1, self.target_size), -10000.)
        init_vvars[0][0] = 0.
        
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.target_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                vvars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(vvars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
            
        best_tag_id = torch.argmax(forward_var)
        best_path = [best_tag_id.item()]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())
        
        return best_path[::-1][1:] # Remove start state effects

class NERModel(nn.Module):
    """
    NER Tagger: BiLSTM + Linear + Manual CRF.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, target_size):
        super(NERModel, self).__init__()
        self.encoder = BiLSTMEncoder(vocab_size, embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim * 2, target_size)
        self.crf = CRF(target_size)

    def neg_log_likelihood(self, sentence, tags):
        # Sum of gold scores over batch
        feats = self.hidden2tag(self.encoder(sentence)).squeeze(0)
        forward_score = self.crf._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def _score_sentence(self, feats, tags):
        # Calculate the score of the provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([0], dtype=torch.long), tags]) # Start
        for i, feat in enumerate(feats):
            score = score + self.crf.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        return score

    def forward(self, sentence):
        # Only used during inference
        feats = self.hidden2tag(self.encoder(sentence)).squeeze(0)
        return self.crf._viterbi_decode(feats)