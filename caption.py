import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CapData(Dataset):
	'''
	Caption class inherits PyTorch dataset for the PyTorch DataLoader
	'''

	def __init__(self, in_dir, data_type, split, transform=None):
		# get parameters, images, encoded captions, caption lengths
		if split not in {'train', 'val', 'test'}:
			raise Exception("split value is not valid")

		self.split = split
		self.f = h5py.File(os.path.join(in_dir, self.split + '_imgs.hdf5'), 'r')
		self.cpi = self.f.attrs['captions_per_image']
		self.imgs = self.f['images']

		with open(os.path.join(in_dir, self.split + '_enccaps.json'), 'r') as f:
			self.caps = json.load(f)
		with open(os.path.join(in_dir, self.split + '_lencaps.json'), 'r') as f:
			self.len_caps = json.load(f)

		self.size = len(caps)
		self.transform = transform

	def __getitem__(self, cap_i):
		img_i = cap_i // self.cpi  # each image corresponds to cpi of captions
		img = torch.FloatTensor(self.imgs[img_i]/255.)
		cap = torch.LongTensor(self.caps[cap_i])
		cap_l = torch.LongTensor(self.len_caps[cap_i])

		if self.transform:
			img = self.transform(img)

		if self.split is 'train':
			return img, cap, cap_l
		else:
			# if not training, need to return all captions for calculating BLEU score
			all_cap = torch.LongTensor(self.caps[self.cpi*img_i : self.cpi*(img_i+1)])
			return img, cap, cap_l, all_cap

	def __len__(self):
		return self.size
