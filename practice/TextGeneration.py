import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        # This is very english language specific
        # We will ingest only these characters:
        self.whitelist = [chr(i) for i in range(32, 127)]

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                line = ''.join([c for c in line if c in self.whitelist])
                words = line.split() + ['<eos>']
                tokens += len(words)  # Why do we need this variable? TODO
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        # TODO: Can't I achieve the same thing by iterating the dictionary? Better?
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = ''.join([c for c in line if c in self.whitelist])
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class RNNModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(RNNModel, self).__init__()

        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop1(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop2(output)
        # tuple returned by gru (sequence length, batch, directions * hidden size)
        # tuple (0,1,2) indices
        # therefore, to linearize input to linear = (sequence length * batch_size) , hidden size
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    if torch.cuda.is_available():
        data = data.cuda()
    return data

corpus = Corpus('./data/shakespear')
dummy_data = "Once upon a time there was a good king and a queen"
dummy_data_idx = [corpus.dictionary.word2idx[w] for w in dummy_data.split()]
dummy_tensor = torch.LongTensor(dummy_data_idx)
op = batchify(dummy_tensor, 2)
for row in op:
    print("%10s %10s" %  (corpus.dictionary.idx2word[row[0]], corpus.dictionary.idx2word[row[1]]))

bs_train = 20       # batch size for training set
bs_valid = 10       # batch size for validation set
bptt_size = 35      # number of times to unroll the graph for back propagation through time
clip = 0.25         # gradient clipping to check exploding gradient

embed_size = 200    # size of the embedding vector
hidden_size = 200   # size of the hidden state in the RNN
num_layers = 2      # number of RNN layres to use
dropout_pct = 0.5   # %age of neurons to drop out for regularization

train_data = batchify(corpus.train, bs_train)
val_data = batchify(corpus.valid, bs_valid)

vocab_size = len(corpus.dictionary)
model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout_pct)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss

def get_batch(source, i, evaluation=False):
    seq_len = min(bptt_size, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()
    return data, target


data, target = get_batch(train_data, 1)


def train(data_source, lr):
    # Turn on training mode which enables dropout.

    model.train()
    total_loss = 0
    hidden = model.init_hidden(bs_train)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt_size)):

        data, targets = get_batch(data_source, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = Variable(hidden.data)

        if torch.cuda.is_available():
            hidden = hidden.cuda()

        # model.zero_grad()
        optimizer.zero_grad()

        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()
        total_loss += len(data) * loss.data

    return total_loss[0] / len(data_source)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(bs_valid)

    for i in range(0, data_source.size(0) - 1, bptt_size):
        data, targets = get_batch(data_source, i, evaluation=True)

        if torch.cuda.is_available():
            hidden = hidden.cuda()

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, vocab_size)

        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = Variable(hidden.data)

    return total_loss[0] / len(data_source)


# Loop over epochs.
best_val_loss = None


def run(epochs, lr):
    global best_val_loss

    for epoch in range(0, epochs):
        train_loss = train(train_data, lr)
        val_loss = evaluate(val_data)
        print("Train Loss: ", train_loss, "Valid Loss: ", val_loss)

        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./4.model.pth")

run(10, 0.001)

num_words = 200
temperature = 1

model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout_pct)
model.load_state_dict(torch.load("./4.model.pth"))

if torch.cuda.is_available():
    model.cuda()

model.eval()
# https://nlp.stanford.edu/blog/maximum-likelihood-decoding-with-rnns-the-good-the-bad-and-the-ugly/
# Which sample is better? It depends on your personal taste. The high temperature
# sample displays greater linguistic variety, but the low temperature sample is
# more grammatically correct. Such is the world of temperature sampling - lowering
# the temperature allows you to focus on higher probability output sequences and
# smooth over deficiencies of the model.

# If we set a high temperature, we can get more entropic (*noisier*) probabilities
# Often we want to sample with low temperatures to produce sharp probabilities
temperature = 0.8

hidden = model.init_hidden(1)
idx = corpus.dictionary.word2idx['I']
input = Variable(torch.LongTensor([[idx]]).long(), volatile=True)

if torch.cuda.is_available():
    input.data = input.data.cuda()

print(corpus.dictionary.idx2word[idx], '', end='')

for i in range(num_words):
    output, hidden = model(input, hidden)
    word_weights = output.squeeze().data.div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input.data.fill_(word_idx)
    word = corpus.dictionary.idx2word[word_idx]

    if word == '<eos>':
        print('')
    else:
        print(word + ' ', end='')


