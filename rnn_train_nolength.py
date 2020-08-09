import sys
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, TabularDataset, Iterator
from konlpy.tag import Mecab
from tqdm import tqdm

tokenizer = Mecab()
tokenize_func = lambda x: ['/'.join((tkn.lower(), pos.lower())) for (tkn, pos) in tokenizer.pos(x)]

# None for flexible sequence lengths
g_max_seq_len = 64
# None for no maximum
g_vocab_max_size = None
g_vocab_min_freq = 5
g_stop_words = ['./sf']
print("* stopwords: ", g_stop_words)

TEXT = Field(
		sequential = True,
		use_vocab = True,
		tokenize = tokenize_func,
		lower = True,
		fix_length = g_max_seq_len,
		include_lengths = False,
		batch_first = True,
		stop_words = g_stop_words)

LABEL = Field(
		sequential = False,
		use_vocab = False,
		preprocessing = lambda x: int(x),
		include_lengths = False,
		batch_first = True,
		is_target = True)

train_data, test_data = TabularDataset.splits(
		path='./data/nsmc',
		train="ratings_train.txt",
		test="ratings_test.txt",
		format='tsv',
		fields=[(None, None), ('text', TEXT), ('label', LABEL)],
		skip_header=True)

SEED = 30
valid_data, test_data = test_data.split(split_ratio = 0.8, random_state = random.seed(SEED))

# data.examples, data.fields
# data.examples -> {text, label}
print("\n*** Data ***")
print("# train: ", len(train_data.examples), len(train_data))
print("# valid: ", len(valid_data.examples))
print("# test: ", len(test_data.examples))
print("train example[0]: ", vars(train_data.examples[0]))

TEXT.build_vocab(train_data, min_freq = g_vocab_min_freq, max_size = g_vocab_max_size, vectors = None)
print("\n*** Vocabulary ***")
print("# vocab: ", len(TEXT.vocab))
print(TEXT.vocab.freqs.most_common(10))
print(TEXT.vocab.itos[:10])
#print('<pad> index:', TEXT.vocab.stoi['<pad>'])

g_batch = 64
g_epochs = 5
g_log_steps = 200

g_pad_idx = TEXT.vocab.stoi.get('<pad>', None)
print("<pad> index:", g_pad_idx)
#g_pad_idx = None
g_vocab_size = len(TEXT.vocab)
g_embed_size = 100
g_hidden_size = 100
g_output_size = 2
# num_layers must be larger than or equal to 2, to use dropout in rnn
g_num_layers = 3
g_dropout_p = 0.3
g_batch_first = True
g_bidirectional = True
g_batchnorm = False
g_learning_rate = 0.001
g_max_grad_norm = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print("\n*** device: ", device)

train_loader = Iterator(dataset = train_data, batch_size = g_batch, device = device, shuffle = True)
valid_loader = Iterator(dataset = valid_data, batch_size = g_batch, device = device, shuffle = False)
test_loader = Iterator(dataset = test_data, batch_size = g_batch, device = device, shuffle = False)


class LSTMClassifier(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers = 3,
		 batch_first = True, bidirectional = True, dropout_p = 0.0, pad_idx = None, batchnorm = True):
		super(LSTMClassifier, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.dropout_p = dropout_p
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		if bidirectional:
			self.num_directions = 2
		else:
			self.num_directions = 1

		self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = pad_idx)
		self.rnn = nn.LSTM(input_size = embed_size,
							 hidden_size = hidden_size,
							 num_layers = num_layers,
							 bias = True,
							 batch_first = batch_first,
							 dropout = dropout_p,
							 bidirectional = bidirectional)

		# fully connected network
		self.linear = nn.Linear(self.num_directions * hidden_size, self.num_directions * hidden_size)
		self.relu = nn.ReLU()
		if dropout_p > 0.0:
			self.dropout = nn.Dropout(p = dropout_p)
		if batchnorm:
			self.batchnorm = nn.BatchNorm1d(self.num_directions * hidden_size)
		else:
			self.batchnorm = None
		self.classifier = nn.Linear(self.num_directions * hidden_size, output_size)

	def forward(self, x):
		embeded = self.embedding(x)
		hidden, cell = self.init_hiddens(x.size(0), device = x.device)
		# output = (batch_size, seq_len, hidden_size * bidirection)
		# hidden = (num_layers * bidirection, batch_size, hidden_size)
		rnn_output, (hidden, cell) = self.rnn(embeded, (hidden, cell))
		#last_hidden = torch.cat([rnn_output[:, -1, : -self.hidden_size], rnn_output[ :, 0, -self.hidden_size : ]], dim = 1)
		last_hidden = torch.cat([h for h in hidden[-self.num_directions : ]], dim = 1)

		if self.dropout_p > 0.0:
			last_hidden = self.dropout(last_hidden)
		if self.batchnorm:
			last_hidden = self.batchnorm(last_hidden)

		# fully connected network
		last_hidden = self.relu(self.linear(last_hidden))

		# batch_size, output_size
		output = self.classifier(last_hidden)
		#return output.view(-1)
		return output
	
	def init_hiddens(self, batch_size, device):
		hidden = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size)
		cell = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size)
		return hidden.to(device), cell.to(device)


model = LSTMClassifier(g_vocab_size, g_embed_size, g_hidden_size, g_output_size, g_num_layers,
		g_batch_first, g_bidirectional, g_dropout_p, g_pad_idx, g_batchnorm).to(device)
#print(model)

num_params = 0
for params in model.parameters():
    num_params += params.view(-1).size(0)
print("\n*** # of parameters: {}".format(num_params))


loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = g_learning_rate)

def calc_accuracy(X, Y, use_sum = False):
	_, pred = torch.max(X, 1)
	if use_sum:
		accuracy = (pred.view(Y.size()) == Y).sum()
	else:
		accuracy = (pred.view(Y.size()) == Y).sum()/torch.tensor(Y.size()[0], dtype=torch.float64)
	return accuracy

for e in range(g_epochs):
	model.train()
	train_size = len(train_loader.dataset)
	train_acc = 0.0
	for b, batch in enumerate(train_loader):
		text, label = batch.text.to(device), batch.label.to(device)
		#print(text.size())
		#print(text[0])
		optimizer.zero_grad()
		output = model(text)
		loss = loss_func(output, label)
		loss.backward()
		if g_max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), g_max_grad_norm)
		optimizer.step()
		train_acc += calc_accuracy(output, label)
		if b % g_log_steps == 0:
			print("train epoch {} ({:05.2f}%), loss: {:.7f}, accurcy: {:7f}".format(e + 1, (100.* b * train_loader.batch_size)/train_size, loss.item(), train_acc/(b + 1)))


	model.eval()
	valid_size = len(valid_loader.dataset)
	valid_correct = 0.0
	for batch in valid_loader:
		text, label = batch.text.to(device), batch.label.to(device)
		output = model(text)
		valid_correct += calc_accuracy(output, label, use_sum = True)

	valid_acc = valid_correct/valid_size
	print("* valid epoch {}, accuracy: {:.9f}".format(e + 1, valid_acc))


