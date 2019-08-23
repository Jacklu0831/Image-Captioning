# generic
import numpy as np 
import json
import argparse

# image
import skimage.transform
from scipy.misc import imread, imresize
from PIL import image
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as F 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def caption_image(enc, dec, img, word_map, b=3):
    """
    Captions an image with beam search. "b" means beam search.
    """

	vocab_size = len(word_map)

	# read image, make sure B&W have 3 channels, resize, shift axis, normalize
	img = imread(img)
	if len(img.shape) == 2:
		img = img[:, :, np.newaxis]
		img = np.concatenate([img, img, img], axis=2)
	img = imresize(img, (256, 256)).transpose(2, 0, 1) / 255.
	img = torch.FloatTensor(img).to(device)

	# normalize for encoder
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    img = transform(img) # (3, 256, 256)

    # encode image, extract params and flatten
    enc_out = encoder(img.unsqueeze(0)) # (1, enc_s, enc_s, enc_dim)
    enc_size = enc_out.size(1)
    enc_dim = enc_out.size(-1)
    enc_out = enc_out.view(1, -1, enc_dim) # (1, num_pix, enc_dim)
    num_pix = enc_out.size(1)
    enc_out = enc_out.expand(b, num_pix, enc_dim) # (b, num_pix, enc_dim)

    prev_words = torch.LongTensor([[word_map['<start>']]] * b).to(device)
    seqs = prev_words
    top_scores = torch.zeros(b, 1).to(device)

    alphas = torch.ones(b, 1, enc_size, enc_size).to(device)

    curr_seqs = []
    curr_alphas = []
    curr_scores = []

    t = 1
    hidden, cell = decoder.init_state(enc_out)

    while True:
    	# procedure similar to evaluation
    	emb = decoder.embedding(prev_words).squeeze(1) # (t, emb_dim)
    	att_out, alpha = decoder.attention(enc_out, hidden) # (t, enc_dim), (t, num_pix)
    	alpha = alpha.view(-1, enc_size, enc_size) # (t, enc_s, enc_s)

    	gate = decoder.sigmoid(decoder.f_beta(hidden)) # (t, enc_dim)
    	att_out *= gate # (t, enc_dim)

        # (t, dec_dim), (t, dec_dim)
    	hidden, cell = decoder.decode(torch.cat([emb, att_out], dim=1), (hidden, cell))

    	scores = decoder.fc(hidden) # (t, vocab_s)
    	scores = F.log_softmax(scores, dim=1) # (t, vocab_s)
    	scores = top_scores.expand_as(scores) + scores # (t, vocab_s)

        if t == 1:
            # top values and indices
            top_scores, top_words = scores[0].topk(b, 0, largest=True, sorted=True) # (s)
        else:
            top_scores, top_words = scores.view(-1).topk(b, 0, largest=True, sorted=True) # (s)
        
        prev_word_idxs = top_words / vocab_size # (t)
        next_word_idxs = top_words % vocab_size # (t)

        # update seqs combinations and alphas
        seqs = torch.cat([seqs[prev_word_idxs], next_word_idxs.unsqueeze(1)], dim=1) # (t, time+1)
        # (t, time+1, enc_s, enc_s)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)], dim=1)
        
        # find the indices that are not end
        going_idxs = [idx for idx, next_word in enumerate(next_word_idxs) if next_word is not word_map['end']]
        ended_idxs = [set(range(len(next_word_idxs)) - set(going_idxs))]
        
        # add completed captions
        if ended_idxs:
            curr_seqs.extend(seqs[ended_idxs].tolist())
            curr_scores.extend(top_scores[ended_idxs])
            curr_alphas.extend(seqs_alpha[ended_idxs].tolist())
            
        # update beam size
        b -= len(ended_idxs)
        if b == 0:
            break
            
        # filter out all completed sentences
        seqs = seqs[going_idxs]
        seqs_alpha = seqs_alpha[ended_idxs]
        hidden = hidden[prev_word_idxs[going_idxs]]
        cell = cell[prev_word_idxs[going_idxs]]
        enc_out = enc_out[prev_word_idxs[going_idxs]]
        top_scores = top_scores[going_idxs].unsqueeze(1)
        prev_words = next_word_idxs[going_idxs].unsqueeze(1)
        
        # break if caption too long
        if step > 100:
            break
        step += 1

    # get best of all
    i = curr_scores.index(max(curr_scores))

    return curr_seq[i], curr_alphas[i]


def visualize(img, seq, alphas, reverse_word_map, smooth=True):
    # open and resize image
	img = Image.open(img).resize([14 * 24, 14 * 24], Image.LANCZOS)
    # get caption in a list
	words = [reverse_word_map[i] for i in seq]

    # make sure the file does not crash (caption too long)
	assert t <= 100

	for t in range(len(words)):
        # refer to matplotlib documentation
		plt.subplot(np.ceil(len(words)/5.), 5, t+1)
		plt.text(0, 1, '%s' % words[t], color='black', backgroundcolor='white', fontsize=12)
		plt.imshow(img)

		curr_alpha = alphas[t, :]

		if smooth:
			alpha = skimage.transform.pyramid_expand(curr_alpha.numpy(), upscale=24, sigma=8)
		else:
			alpha = skimage.transform.resize(curr_alpha.numpy(), [14*24, 14*24])

        # overlap alpha on image
		if t == 0:
			plt.imshow(alpha, alpha=0)
		else:
			plt.imshow(alpha, alpha=0.8)

		plt.set_cmap(plt.cm.Greys_r)
		plt.axis('off')

	plt.show()


# command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--img', help='image path')
ap.add_argument('-o', '--output', default=None, help='output path')
ap.add_argument('-m', '--model', help='model path')
ap.add_argument('-w', '--wordmap', help='word map path')
ap.add_argument('-b', '--beam_size', help='beam size')
ap.add_argument('-s', '--smooth', help='smooth alpha overlay')
args = ap.parse_args()

# load model
trained_model = torch.load(args.model)
decoder = trained_model['decoder']
encoder = trained_model['encoder']
decoder = decoder.to(device)
encoder = encoder.to(device)
decoder.eval()
encoder.eval()

# load word map
with open(args.word_map, 'r') as f:
	word_map = json.laod(f)
reverse_word_map = {v:k for k, v in word_map.items()}

# get caption and alpha values (to tensor)
seq, alphas = caption_image(encoder, decoder, args.img, word_map, args.beam_size)
alphas = torch.FloatTensor(alphas)

# visualize caption and attention
visualize(args.img, seq, alphas, reverse_word_map, args.smooth)




