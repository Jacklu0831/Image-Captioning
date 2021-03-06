import json
import os
import h5py
import argparse
import numpy as np
from collections import Counter
from random import choice, sample

from scipy.misc import imread, imresize

# I set these default values for my convenience, feel free to change them
ap = argparse.ArgumentParser()
ap.add_argument('-j', '--json', default='input/dataset_flickr30k.json', help="path to json file")
ap.add_argument('-i', '--img_dir', default='input/flickr30k_images/', help="directory to images")
ap.add_argument('-o', '--out_dir', default='processed_input', help='directory to store files')
ap.add_argument('-mf', '--min_freq', default=5, help="minimum word frequency")
ap.add_argument('-l', '--max_len', default=5000, help="max length of caption")
ap.add_argument('-cpi', '--captions_per_image', default=5, help="captions per image of dataset")
args = vars(ap.parse_args())

json_path = args['json']
img_dir = args['img_dir']
out_dir = args['out_dir']
min_freq = args['min_freq']
max_len = args['max_len']
cpi = args['captions_per_image']

# load data from Karpathy
with open(json_path, 'r') as f:
	data = json.load(f)

print(list(data.keys()))         			           # ['images', 'dataset']
print(list(data['images'][0].keys()))                  # ['sentids', 'imgid', 'sentences', 'split', 'filename']
print(list(data['images'][0]['sentences'][0].keys()))  # ['tokens', 'raw', 'imgid', 'sentid']
print(data['images'][0]['split'])                      # train

train_img = []
train_cap = []
val_img = []
val_cap = []
test_img = []
test_cap = []
freq = Counter()

# populate the above containers
for img in data['images']:
	caps = []
	# update word frequency and get captions
	for s in img['sentences']:
		freq.update(s['tokens'])
		if len(s["tokens"]) <= 100:
			caps.append(s["tokens"])
	if not caps:
		continue
	# populate our lists
	path = os.path.join(img_dir, img['filename'])
	if img['split'] == 'train' or img['split'] == 'restval':
		train_img.append(path)
		train_cap.append(caps)
	elif img['split'] == 'val':
		val_img.append(path)
		val_cap.append(caps)
	elif img['split'] == 'test':
		test_img.append(path)
		test_cap.append(caps)

print(len(train_img)) # 29000
print(len(val_img))   # 1014
print(len(test_img))  # 1000

# create word map
vocab = [w for w in freq.keys() if freq[w] > min_freq] # filter out lesser used words
word_map = {k: v+1 for v, k in enumerate(vocab)}  # word2idx from 1
word_map['<pad>'] = 0
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<unk>'] = len(word_map) + 1

# store word map
with open(os.path.join(out_dir, 'wordmap.json'), 'w') as f:
	json.dump(word_map, f)

# store values
# 'train' will take a few minutes but the others are within seconds
for paths, caps, split in [(train_img, train_cap, 'train'),
						  (val_img, val_cap, 'val'),
						  (test_img, test_cap, 'test')]:

	with h5py.File(os.path.join(out_dir, split + '_imgs.hdf5'), 'a') as f:
		f.attrs['captions_per_image'] = cpi
		images = f.create_dataset('images', (len(paths), 3, 256, 256), dtype='uint8')

		# encoded captions and length of captions
		enc_caps = []
		len_caps = []

		for i, path in enumerate(paths):
			# randomly sample 5 captions
			if len(caps[i]) >= cpi:
				cap_selected = sample(caps[i], k=cpi)
			else:
				cap_selected = caps[i] + [choice(caps[i]) for _ in range(cpi-len(caps[i]))]

			# process image (take care of B&W) to 3x256x256 then store in h5py to not crash RAM
			img = imread(paths[i])
			if len(img.shape) == 2:
				img = img[:,:,np.newaxis]
				img = np.concatenate([img, img, img], axis=2)
			img = imresize(img, (256, 256))
			img = img.transpose(2, 0, 1)
			images[i] = img

			for j, c in enumerate(cap_selected):
				# concat start, end, pads (encoded caption)
				enc_cap = [word_map['<start>']] + [word_map.get(w, word_map['<unk>']) for w in c] \
						  + [word_map['<end>']] + (max_len - len(c)) * [word_map['<pad>']]
				enc_caps.append(enc_cap)
				len_caps.append(len(c)+2) # ignore pads

		# sanity check
		assert images.shape[0] * captions_per_image == len(enc_caps) == len(len_caps)

		# store captions and their lengths
		with open(os.path.join(out_dir, split + '_enccaps.json'), 'w') as f:
			json.dump(enc_caps, f)
		with open(os.path.join(out_dir, split + '_lencaps.json'), 'w') as f:
			json.dump(len_caps, f)
