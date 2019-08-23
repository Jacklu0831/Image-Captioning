import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CapData(Dataset):
	'''
	Dataset class to be used in DataLoader class for creating batches.
	'''

	def __init__(self, input_dir, split, transform=None):
		# get split/section of data
		assert split in {'train', 'val', 'test'}
		self.split = split

		# load file data and attribute
		self.f = h5py.File(os.path.join(input_dir, self.split + '_imgs.hdf5'), 'r')
		self.cpi = self.f.attrs['captions_per_image']
		self.imgs = self.f['images']

		# obtain captions and caption lengths
		with open(os.path.join(input_dir, self.split + '_enccaps.json'), 'r') as f:
			self.caps = json.load(f)
		with open(os.path.join(input_dir, self.split + '_lencaps.json'), 'r') as f:
			self.len_caps = json.load(f)

		self.size = len(self.caps)
		self.transform = transform

	def __getitem__(self, cap_i):
		"""
		Get the corresponding image and caption with given index
		"""
		img_i = cap_i // self.cpi  # each image corresponds to cpi of captions
		img = torch.FloatTensor(self.imgs[img_i] / 255.)
		cap = torch.LongTensor(self.caps[cap_i])
		cap_l = torch.LongTensor(self.len_caps[cap_i])

		if self.transform:
			img = self.transform(img)

		if self.split is 'train':
			return img, cap, cap_l
		else:
			# return all captions of the image for calculating BLEU score (eval and test)
			all_cap = torch.LongTensor(self.caps[self.cpi*img_i: self.cpi*(img_i+1)])
			return img, cap, cap_l, all_cap

	def __len__(self):
		return self.size
