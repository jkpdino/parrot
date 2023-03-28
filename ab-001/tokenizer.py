class Token:
    def __init__(self, n: int):
        self.num = n

    def __str__(self):
        return str(self.num)

def tokenFromChar(c: str) -> Token:
    # 1 - 26 for a - z
    if ord('a') <= ord(c) <= ord('z'):
        return Token(1 + (ord(c) - ord('a')))
    # 27 - 52 for A - Z
    elif ord('A') <= ord(c) <= ord('Z'):
        return Token(27 + (ord(c) - ord('A')))
    # 53 - 62 for 0 - 9
    elif ord('0') <= ord(c) <= ord('9'):
        return Token(53 + (ord(c) - ord('0')))
    elif c == ' ':
        return Token(63)
    elif c in [ '.', '!', '?', ',', ':', ';' ]:
        return Token(64)
    elif c in [ '&', '$', '#' ]:
        return Token(65)
    elif c in [ '(', '[', '{' ]:
        return Token(66)
    elif c in [ ')', ']', '}' ]:
        return Token(67)
    
    return Token(0)

class Window:
    def __init__(self, tokens: list[Token], target: Token):
        self.tokens = tokens
        self.target = target

    def __str__(self):
        return f'{map(lambda t: str(t), self.tokens)} -> {self.target}'

def tokenize(s: str) -> list[Token]:
    return [tokenFromChar(c) for c in s] + [ Token(0) ]

def createWindows(s: str, window_size: int) -> list[Window]:
    tokens = tokenize(s)
    windows = []

    for window_end in range(len(tokens)):
        window_start = max(0, window_end - window_size)
        window = Window(tokens[window_start:window_end], tokens[window_end])

        windows.append(window)

    return windows