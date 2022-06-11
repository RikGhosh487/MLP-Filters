#!/usr/bin/env python

from torch.utils.data import Dataset, DataLoader
from csv import reader
import torch


class GAIAData(Dataset):
	def __init__(self, dataset_path:str) -> None:
		self.datapath = dataset_path
		self.u_s = None
		self.g_s = None
		self.r_s = None
		self.i_s = None
		self.z_s = None
		self.g_g = None
		self.bp_g = None
		self.rp_g = None

		with open('./%s/mags.csv' % self.datapath, newline='') as csv_file:
			dataset = reader(csv_file)
			# headers in the first row
			for idx, row in enumerate(dataset):
				if idx == 0:
					if 'u' in row:
						self.u_s = list()
					if 'g' in row:
						self.g_s = list()
					if 'r' in row:
						self.r_s = list()
					if 'i' in row:
						self.i_s = list()
					if 'z' in row:
						self.z_s = list()
					if 'gaia_g' in row:
						self.g_g = list()
					if 'bp' in row:
						self.bp_g = list()
					if 'rp' in row:
						self.rp_g = list()

				# populate fields
				else:
					if self.u_s is not None:
						self.u_s.append(float(row[0]))
					else:
						raise ValueError('u is not in the csv')
					if self.g_s is not None:
						self.g_s.append(float(row[1]))
					else:
						raise ValueError('g is not in the csv')
					if self.r_s is not None:
						self.r_s.append(float(row[2]))
					else:
						raise ValueError('r is not in the csv')
					if self.i_s is not None:
						self.i_s.append(float(row[3]))
					else:
						raise ValueError('i is not in the csv')
					if self.z_s is not None:
						self.z_s.append(float(row[4]))
					else:
						raise ValueError('z is not in the csv')
					if self.g_g is not None:
						self.g_g.append(float(row[5]))
					else:
						raise ValueError('gaia_g is not in the csv')
					if self.bp_g is not None:
						self.bp_g.append(float(row[6]))
					else:
						raise ValueError('bp is not in the csv')
					if self.rp_g is not None:
						self.rp_g.append(float(row[7]))
					else:
						raise ValueError('rp is not in the csv')

	def __len__(self) -> int:
		return len(self.g_g)

	def __getitem__ (self, idx:int) -> tuple:
		u, g, r, i, z = self.u_s[idx], self.g_s[idx], self.r_s[idx], self.i_s[idx], self.z_s[idx]
		gg, bp, rp = self.g_g[idx], self.bp_g[idx], self.rp_g[idx]
		x = torch.tensor((gg, bp, rp))
		y = torch.tensor((u, g, r, i, z))

		return x, y			# input 3 -> output 5


class SDSSData(GAIAData):
	def __getitem__(self, idx:int) -> tuple:
		x, y = super().__getitem__(idx)
		return y, x 		# input 5 -> output 3


# accuracy index - Gaussian Convergence
def accuracy(pred:torch.Tensor, truth:torch.Tensor) -> float:
	# error tensor
	err = pred - truth
	scaled = -2 * torch.pow(err, 2)
	return torch.mean(torch.exp(scaled), dim=1)


def load_data(dataset_path:str, model:str, num_workers:int=0, batch_size:int=128) -> DataLoader:
	if model == 'gaia':
		dataset = SDSSData(dataset_path)
	elif model == 'sdss':
		dataset = GAIAData(dataset_path)
	else:
		raise ValueError('%s is not a valid model type' % model)
	
	return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

