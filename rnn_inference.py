import torch
import numpy as np
from torchtext.data import Field, TabularDataset, Iterator
from konlpy.tag import Mecab
from LSTMClassifier import LSTMClassifier

tokenizer = Mecab()
tokenize_func = lambda x: ['/'.join((tkn.lower(), pos.lower())) for (tkn, pos) in tokenizer.pos(x)]

device = "cpu"
model_path = "./model-nsmc.pt"

g_max_seq_len = 64
g_stop_words = ['./sf']

TEXT = Field(
		sequential = True,
		use_vocab = True,
		tokenize = tokenize_func,
		lower = True,
		fix_length = g_max_seq_len,
		include_lengths = True,
		batch_first = True,
		stop_words = g_stop_words)

checkpoint = torch.load(model_path)
TEXT.vocab = checkpoint['vocab']
g_vocab_size = len(TEXT.vocab)
print("vocab size:", g_vocab_size)
#print(TEXT.vocab.freqs.most_common(10))
#print(TEXT.vocab.itos[:10])

g_pad_idx = TEXT.vocab.stoi.get('<pad>', None)
g_embed_size = 100
g_hidden_size = 100
g_output_size = 2
g_num_layers = 3
g_dropout_p = 0.0
g_batch_first = True
g_bidirectional = True
g_batchnorm = False

model = LSTMClassifier(g_vocab_size, g_embed_size, g_hidden_size, g_output_size, g_num_layers,
		g_batch_first, g_bidirectional, g_dropout_p, g_pad_idx, g_batchnorm).to(device)
model.load_state_dict(checkpoint['model_state_dic'])
#print(model)
print(checkpoint['best_acc'])

model.eval()
sample = "wow very good"
processed = TEXT.process([tokenize_func(sample)])
x, xlen = processed[0].to(device), processed[1].to(device)
output = model(x, xlen)
_, pred = torch.max(output, 1)
#print(output.softmax(-1))
print("{}\t{}".format(sample, pred.item()))

