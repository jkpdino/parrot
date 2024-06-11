import torch
import bigram
import tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_text(file):
    with open(file) as f:
        text = f.read()

        ratio = 0.1

        split = int(len(text) * ratio)

        return text[:split]
    

context_length = 8
    
text = get_text("tinyshakespeare.txt")
dataset = tokenizer.CharacterDataset(text, window_size=context_length)
    
m = bigram.BigramLanguageModel(dataset.vocab_size)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


# Train here
batch_size = 32
epochs = 25
dataloader = DataLoader(dataset, batch_size)

for i in range(epochs):
    m.train()
    for xs, ys in tqdm(dataloader):
        logits, loss = m(xs, ys)

        m.zero_grad()
        loss.backward()
        optimizer.step()

    print(m.generate())