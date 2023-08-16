from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset

# Give credit

class CharacterDataset(Dataset):
    """Custom dataset.

    Parameters
    ----------
    text : str
        Input text that will be used to create the entire database.

    window_size : int
        Number of characters to use as input features.

    vocab_size : int
        Number of characters in the vocabulary. Note that the last character
        is always reserved for a special "~" out-of-vocabulary character.

    Attributes
    ----------
    ch2ix : defaultdict
        Mapping from the character to the position of that character in the
        vocabulary. Note that all characters that are not in the vocabulary
        will get mapped into the index `vocab_size - 1`.

    ix2ch : dict
        Mapping from the character position in the vocabulary to the actual
        character.

    vocabulary : list
        List of all characters. `len(vocabulary) == vocab_size`.
    """
    def __init__(self, text, window_size=1):
        vocab_size = len(sorted(list(set(text))))

        self.vocab_size = vocab_size
        self.text = " ".join(text.split())
        self.window_size = window_size
        self.ch2ix = defaultdict(lambda: vocab_size - 1)

        most_common_ch2ix = {
            x[0]: i
            for i, x in enumerate(Counter(self.text).most_common()[: (vocab_size - 1)])
        }
        self.ch2ix.update(most_common_ch2ix)
        self.ch2ix["~"] = vocab_size - 1

        self.ix2ch = {v: k for k, v in self.ch2ix.items()}
        self.vocabulary = [self.ix2ch[i] for i in range(vocab_size)]

    def __len__(self):
        return len(self.text) - self.window_size

    def __getitem__(self, ix):
        X = torch.LongTensor(
            [self.ch2ix[c] for c in self.text[ix : ix + self.window_size]]
        )
        Y = torch.LongTensor(
            [self.ch2ix[c] for c in self.text[ix + 1 : ix + self.window_size + 1]]
        )

        return X, Y