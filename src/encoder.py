import numpy as np
import random
from tqdm import tqdm
from utils import sigmoid


class Encoder(object):
	"""
	The Encoder module is responsible for the training of the node 
	embeddings. Each node embedding consists of four different features:
	outward/inward latent features and outward/inward context tendency
	features.

	It uses the positive and negative sample multisets generated
	by the Preprocessor module to optimize the embedding 
	features for each node.


	Attributes:
		d_graph (dict): a dictionary that keeps the context-label information
		lf_dim (integer): inward/outward latent feature (L) dimension
		ct_dim (integer): inward/outward context tendency feature (CT) dimension
		nodes (list): list of nodes in the context-labeled trust network
		num_noise (integer): number of noise sample for each positive sample
	"""
	def __init__(self, d_graph, lf_dim, ct_dim, nodes, num_noise):
		self.d_graph = d_graph
		self.lf_dim = lf_dim
		self.ct_dim = ct_dim
		self.num_nodes = len(nodes)
		self.nodes = nodes
		self.num_noise = num_noise

		# Randomly initializes the outward latent features
		self.out_lf = np.random.rand(self.num_nodes, lf_dim)
		# Randomly initializes the inward latent features
		self.in_lf = np.random.rand(self.num_nodes, lf_dim)

		# Initializes the outward context tendency features to zeros
		self.out_ct = np.zeros((self.num_nodes, ct_dim))
		# Initializes the inward context tendency features to zeros
		self.in_ct = np.zeros((self.num_nodes, ct_dim))

	def fit(self, epoch, lr, samples):
		"""
		Optimizes the node embeddings using Stochastic Gradient Ascent.

		Parameters:
			epoch (integer): number of epochs
			lr (integer): learning rate
			samples (tuple): positive and negative sample multisets
		"""
		pos_samples, neg_samples = samples
		pbar = tqdm(total=epoch * self.num_nodes, desc='Optimizing', ncols=100)

		for _ in range(epoch):
			random.shuffle(self.nodes)
			for u in self.nodes:
				pbar.update(1)
				for i, v in enumerate(pos_samples[u]):

					# If v is a neighbour of u, updates the ct features
					if v in self.d_graph[u]:
						edge_label = self.d_graph[u][v].T
						f_uv = self.out_lf[u] @ self.in_lf[v] + (self.out_ct[u] + self.in_ct[v]) @ edge_label 
						sigf_uv = sigmoid(f_uv)
						self.out_ct[u] += lr * ((1 - sigf_uv) * edge_label)
						self.in_ct[v] += lr * ((1 - sigf_uv) * edge_label)
					else:
						f_uv = self.out_lf[u] @ self.in_lf[v]
						sigf_uv = sigmoid(f_uv)

					out_u_g = 0
					out_u_g += (1 - sigf_uv) * self.in_lf[v]
					self.in_lf[v] += lr * ((1 - sigf_uv) * self.out_lf[u])

					noises = neg_samples[u][i * self.num_noise : (i + 1) * self.num_noise]

					for noise in noises:
						f_unoise = self.out_lf[u] @ self.in_lf[noise]
						sigf_unoise = sigmoid(-f_unoise)
						out_u_g -= (1 - sigf_unoise) * self.in_lf[noise]
						self.in_lf[noise] += lr * (-(1 - sigf_unoise) * self.out_lf[u])
					
					self.out_lf[u] += lr * out_u_g

		pbar.close()

	def get_embeddings(self):
		"""
		Fills the embedding lookup table.

		Returns:
			outward embedding table (W_out),
			inward embedding table (W_in)
		"""
		W_out = np.zeros((self.num_nodes, self.lf_dim + self.ct_dim))
		W_in = np.zeros((self.num_nodes, self.lf_dim + self.ct_dim))
		for i in self.nodes:
			W_out[i, : self.lf_dim] = self.out_lf[i]
			W_out[i, self.lf_dim:] = self.out_ct[i]
			W_in[i, : self.lf_dim] = self.in_lf[i]
			W_in[i, self.lf_dim:] = self.in_ct[i]

		return W_out, W_in