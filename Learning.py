class Tokenizer():
    def __init__(self,text):
        self.x = sorted(list(set(text)))
        self.vocab = len(self.x)
        self.y = {ch: i for i, ch in enumerate(self.x)}
        self.z = {i: ch for ch, i in self.y.items()}

    def encode(self,a):
        return [self.y[c] for c in a]
    
    def decode(self,b):
        return ''.join([self.z[c] for c in b])


