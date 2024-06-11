import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def generate_causal_mask(sequence_length):
    # Create an upper triangular matrix of shape (sequence_length, sequence_length)
    mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)
    
    # Convert the mask to a boolean tensor
    mask = mask.bool()
    
    return mask


class DecoderBlock(nn.Module):
    def __init__(self, d_model, context_length, num_heads, dropout=0.1):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.mask = generate_causal_mask(context_length)
        
    def forward(self, x):
        # x: input tensor of shape (batch_size, sequence_length, d_model)
        # mask: tensor of shape (batch_size, sequence_length, sequence_length)
        
        # Self-attention with causal masking
        self_attention_output, _ = self.self_attention(x, x, x, attn_mask=self.mask)  # Q, K, V are all x
        self_attention_output = self.layer_norm1(x + self_attention_output)  # Residual connection and layer normalization
        
        # Feed-forward
        feed_forward_output = self.feed_forward(self_attention_output)
        feed_forward_output = self.dropout(feed_forward_output)  # Apply dropout
        decoder_output = self.layer_norm2(self_attention_output + feed_forward_output)  # Residual connection and layer normalization
        
        return decoder_output
    
class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        context_length,
        num_heads,
        num_layers,
        dropout = 0.1):
        super().__init__()

        decoders = [DecoderBlock(d_model, context_length, num_heads, dropout=dropout) for _ in range(num_layers) ]

        self.decoders = nn.Sequential(*decoders)

    def forward(
        self,
        x):
        """
        :param x: (batch_size, sequence_length, d_model)
        """

        return self.decoders(x)



class Parrot(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        context_size,
        n_layers = 2):
        super().__init__()


        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Create positional embeddings
        self.positional_embeddings = positionalencoding1d(embedding_dim, context_size).to(self.device)

        # Decoder Layers
        self.decoder = Decoder(embedding_dim, context_size, 8, n_layers)

        # Create a dense layer mapping to vocab_size outputs
        # Each neuron will correspond to the probability
        # of a character being the next character in the sequence
        self.dense = nn.Linear(embedding_dim, vocab_size)

        # Store the parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

    def forward(
        self,
        tokens,
        targets = None):
        """
        :param tokens: (b, n, d)
        :param targets: (b, n, v)
        """

        # Generate embeddings
        embeddings = self.embedding(tokens)  # (batch_size, sequence_length, embedding_dim)

        # Add position to the embeddings
        embeddings_with_positions = embeddings + self.positional_embeddings  # (batch_size, sequence_length, embedding_dim)

        # Run the embeddings through the decoder blocks
        decoded = self.decoder(embeddings_with_positions)

        # Run the output of the decoder blocks through a linear layer
        # to map to vocab_size outputs
        logits = self.dense(decoded)  # (batch_size, sequence_length, vocab_size)

        # Normalize the output into a probability distribution
        # over the vocabulary
        logits = F.softmax(logits, -1)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape

            loss_logits = logits.view(B*T, C)
            loss_targets = targets.view(B*T)

            loss = F.cross_entropy(loss_logits, loss_targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        number_of_tokens: int,
        temperature: int = 0.8,
        top_k: int = None):
        """
        Takes a (b, t) tensor of tokens and generates number_of_tokens tokens
        """
        for _ in range(number_of_tokens):
            tokens_trimmed = tokens if tokens.size(-1) <= self.context_size else tokens[:, -self.context_size:]

            # Put the tokens through the model to get the logits
            logits, _ = self.forward(tokens_trimmed)

            # Take the last token from the logits and apply the temperature
            logits = logits[:, -1, :] / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Reapply a softmax
            probs = F.softmax(logits, -1)

            # Sample from the distribution
            token_next = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat(tokens, token_next, dim=-1)

        return tokens


# String -> Embeddings -> Positional Embeddings -> Transformer -> Dense -> Softmax -> Next Char
#
#