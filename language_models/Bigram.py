import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int): 
        super().__init__()
         # Represent each character as a embedding vector, which is unnecessary
        # but this is for learning purposes.
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, encoded_word_index: torch.tensor, targets=None):
        logits = self.embedding_table(encoded_word_index)
        if targets is None:
            loss = None             
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, character_idx_sequence: torch.tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, loss = self(character_idx_sequence)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            character_idx_to_decode = torch.multinomial(probs, num_samples=1)
            character_idx_sequence = torch.cat((character_idx_sequence, character_idx_to_decode), dim=1)
        return character_idx_sequence