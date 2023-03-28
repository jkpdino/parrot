import datasets
from tokenizer import createWindows

def train():
    dataset = datasets.load_dataset('empathetic_dialogues')

    for example in dataset['train']:
        utterance = example['utterance']
        windows = createWindows(utterance, 64)

        for window in windows:
            print(window.tokens, window.target)

train()
