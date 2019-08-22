import torch
from torch import nn
import torchvision
from torchvision.models import resnet101


# comment out assertion if you do not have cuda/gpu
assert torch.cuda.is_available()
device = torch.device('cuda') 


class Encoder(nn.Module):
	"""
	ResNet 101 for feature extraction.
	"""

	def __init__(self, enc_img_size = 14, fine_tune = False):
		self.enc_img_size = enc_img_size
		# use resnet101 trained on ImageNet and remove final linear and pool
		net = resnet101(pretrained=True)
		self.net = nn.Sequential(*list(net.children())[:-2])

		self.a_pool = nn.AdaptiveAvgPool2d((enc_img_size, enc_img_size))

	def fine_tune(self, fine_tune)
		# fine tuning block 2-4
		for p in self.net.parameters():
			p.requires_grad = False
		for child in list(self.net.children())[5:]:
			for p in child.parameters():
				p.requires_grad = fine_tune

	def forward(self, imgs):      # (batch_s, 3,         img_s,     img_s)         
		x = self.net(imgs)        # (batch_s, 2048,      img_s/32,  img_s/32) 
		x = self.a_pool(x)        # (batch_s, 2048,      enc_img_s, enc_img_s)
		x = x.permute(0, 2, 3, 1) # (batch_s, enc_img_s, enc_img_s, 2048)


class Attention(nn.Module):
	"""
	Attention network for focusing on import areas of features.
	"""

	def __init__(self, enc_dim, dec_dim, att_dim):
		self.enc_att = nn.Linear(enc_dim, att_dim) # encoded image -> attention size
		self.dec_att = nn.Linear(dec_dim, att_dim) # decoder output -> attention size
		self.att = nn.Linear(att_dim, 1)           # linear layer for softmax
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, enc_out, dec_out):
		# transform encoder + decoder output (hidden state) then apply softmax
		att_enc = self.enc_att(enc_out) # (batch_s, num_pix, att_s)
		att_dec = self.dec_att(dec_out) # (batch_s, att_s)
		att = self.relu(att_enc + att_dec.unsqueeze(1)) # (batch_s, num_pix, att_s)
		att = self.att(att) # (batch_s, num_pix, 1)
		att = att.squeeze(2) # (batch_s, num_pix)
		alpha = self.softmax(att) # (batch_s, num_pix)
		weighted_enc = (enc_out * alpha.unsqueeze(2)).sum(dim=1)

		return weighted_enc, alpha


class Decoder(nn.Module):
	"""
	Decoder with attention for guessing caption.
	"""

	def __init__(self, vocab_size, embed_dim, dec_dim, att_dim, enc_dim=2048):
		# similar to a normal RNN model but:
		# 1. LSTMCell takes in both encoded image and embeddings
		# 2. There is an attention layer making weighted encodings
		# 3. Need to create initial cell state, hidden state, sigmoid gate

		# initialize hyperparameters
		self.enc_dim = enc_dim
		self.att_dim = att_dim
		self.embed_dim = embed_dim
		self.dec_dim = dec_dim
		self.vocab_size = vocab_size
		# attention network/block
		self.attention = Attention(enc_dim, dec_dim, att_dim)
		# cell state and hidden state
		self.init_hidden = nn.Linear(enc_dim, dec_dim)
		self.init_cell = nn.Linear(enc_dim, dec_dim)
		# generic RNN model components
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.dropout = nn.Dropout(p=0.5) # you can set this as an arguement
		self.decode = nn.LSTMCell(embed_dim+enc_dim, dec_dim, bias=True)
		self.gate = nn.Linear(dec_dim, enc_dim) # weight of weighted encoding t LSTM
		self.sigmoid = nn.Sigmoid()
		self.fc = nn.Linear(dec_dim, vocab_size)
		# initialize weights (weights all (-0.1, 1) and bias all 0)
		self.embedding.weight.data.uniform_(-0.1, 0.1)
		self.fc.bias.data.fill_(0)
		self.fc.weight.data.uniform_(-0.1, 0.1)

	def forward(self, enc_out, enc_caps, len_caps):
		batch_size = enc_out.size(0)
		enc_dim = enc_out.size(-1) # or size(4)
		vocab_size = self.vocab_size

		# flatten image in pixels for dim(1)
		enc_out = enc_out.view(batch_size, -1, enc_dim)
		num_pix = enc_out.size(1)

		# sort by captions decreasing length and match the image features
		len_caps, idxs = len_caps.squeeze(1).sort(dim=0, descending=True)
		enc_caps, enc_out = enc_caps[idx], enc_out[idx]

		# set input for embeddings
		embeddings = self.embedding(enc_caps)

		# initialize hidden and cell states of LSTM
		mean_enc_out = enc_out.mean(dim=1) # mean pixel val -> (batch_size, decoder_dim)
		hidden = self.init_hidden(mean_enc_out)
		cell = self.init_cell(mean_enc_out)

		# get rid of <end>
		len_decs = (len_caps-1).toList()

		# allocate memory for tensors to store scores and alphas
		preds = torch.zeros(batch_size, max(len_decs), vocab_size).to(device)
		alphas = torch.zeros(batch_size, max(len_decs), num_pix).to(device)

		# used for look due to the extra attention operations
		for t in range(max(len_decs)):
			# sorted cap lengths due to this
			cbs = sum([l > t for l in len_decs]) # cbs = current batch size
			weighted_enc, alpha = self.attention(enc_out[:cbs], hidden[:cbs])
			# weight the attention weighted encoding
			gate = self.sigmoid(self.f_beta(hidden[:cbs]))
			weighted_enc *= gate
			# decode (cat emb with enc as input and also pass in the previous state)
			hidden, cell = self.decode(torch.cat([embeddings[:cbs, t, :], weighted_enc], dim=1), (hidden[:cbs], cell[:cbs]))
			pred = self.fc(self.dropout(h))
			preds[:cbs, t, :] = pred # 3D tensors output
			alphas[:cbs, t, :] = alpha

		# stored alphas for training
		return enc_caps, len_decs, preds, alphas, idx









