import torch
import torch.nn as nn

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

	def forward(self, x, xlen = None):
		embeded = self.embedding(x)
		hidden, cell = self.init_hiddens(x.size(0), device = x.device)
		# output = (batch_size, seq_len, hidden_size * bidirection)
		# hidden = (num_layers * bidirection, batch_size, hidden_size)
		rnn_output, (hidden, cell) = self.rnn(embeded, (hidden, cell))
		#last_hidden = torch.cat([rnn_output[:, -1, : -self.hidden_size], rnn_output[ :, 0, -self.hidden_size : ]], dim = 1)
		if xlen is not None:
			xlen = xlen - 1
			if self.num_directions == 2:
				last_hidden = torch.cat([rnn_output[list(range(len(rnn_output))), xlen, : -self.hidden_size], rnn_output[ :, 0, -self.hidden_size : ]], dim = 1)
			else:
				last_hidden = [rnn_output[list(range(len(rnn_output))), xlen, : -self.hidden_size]]
		else:
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

