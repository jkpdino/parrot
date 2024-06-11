import datasets
from tokenizer import CharacterDataset
from model import Parrot

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_text(file):
    with open(file) as f:
        text = f.read()

        ratio = 0.1

        split = int(len(text) * ratio)

        return text[:split]


# Total weights: d * ((2v + 1) + L * (12d + 13))
    # embeddings: v * d
    # decoder: 12d^2+13d
    #   attention: 4*(d^2+d)
    #   layer norm 1: 2 * d
    #   feed forward 1: 4d^2+4d
    #   feed forward 2: 4d^2+d
    #   layer norm 2: 2 * d
    # dense: v * d + d

def compute_loss(cal, net, dataloader):
    """Computer average loss over a dataset."""
    net.eval()
    all_losses = []
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        _, loss = net(X_batch, y_batch)

        all_losses.append(loss.item())

    return np.mean(all_losses)

def get_context_window(tokens: list[int], context_window_size: int, padding_ch=0):
    if len(tokens) > context_window_size:
        return tokens[-context_window_size:]
    else:
        return tokens + [padding_ch] * (context_window_size - len(tokens))

def generate_text(prompt: str, model: Parrot, nchars: int, dataset: CharacterDataset, context_window_size: int, seed=None) -> str:
    tokens = [dataset.ch2ix[c] for c in prompt]

    tokens_tensor = torch.tensor([tokens], device=device, requires_grad=False)

    model.eval()
    output_tokens = model.generate(tokens_tensor, nchars, 0.8, 5)
    output_text = [dataset.ix2ch[i] for i in output_tokens]

    return ''.join(output_text)

def train_model():
    # Hyperparameters
    embedding_dim = 16
    context_window_size = 64
    n_layers = 2

    # Training config
    n_epochs = 15
    train_split = 0.8
    batch_size = 64
    seed = 42

    torch.manual_seed(seed)

    # Load the dataset
    text = get_text('shakespeare.txt')
    dataset = CharacterDataset(text, context_window_size)
    vocabulary_size = dataset.vocab_size

    n_samples = len(dataset)
    split_idx = int(n_samples * train_split)

    train_indices, validate_indices = np.arange(split_idx), np.arange(split_idx, n_samples)

    train_dataloader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices), batch_size=batch_size)
    validate_dataloader = DataLoader(dataset, sampler=SubsetRandomSampler(validate_indices), batch_size=batch_size)


    model = Parrot(vocabulary_size,
                   embedding_dim,
                   context_window_size,
                   n_layers=n_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        model.train()

        for input, target in tqdm(train_dataloader):
            input  = input.to(device)
            target = target.to(device)

            # Set all gradients to zero
            optimizer.zero_grad()

            # Run the model
            _, loss = model(input, target)

            # Backpropagate
            loss.backward()

            # Update the weights of the model
            optimizer.step()

        train_loss = compute_loss(model, train_dataloader)
        validate_loss = compute_loss(model, validate_dataloader)

        print(f"Epoch: {epoch}, {train_loss=:.3f}, {validate_loss=:.3f}")

        # Generate a sentence
        prompt = "I hope it works "
        completion = generate_text(prompt, model, 100, dataset, context_window_size, seed=seed)

        print(completion)


train_model()