import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from utils import parallel_generate_walks
import random
from tqdm import tqdm


class Preprocessor(object):
	"""
	The Preprocessor is responsible for the generation of the positive and
	negative sample multisets for each node. 

	Attributes:
		G (networkx graph): context-labeled trust network
		d_graph (dict): dictionary that keeps the context-label information
		num_walks (integer): number of walks starting from each node
		walk_len (integer): maximum walk length
		window_size (integer): length of the context window
		workers (integer): number of workers for the parallel random walk generation
		sim_metric (string): similarity measure (e.g., cosine similarity, euclidean similarity etc.)
		num_noise (integer): number of noise sample for each positive sample
		neg_test_edges (list)
	"""
	def __init__(self, G, d_graph, num_walks, walk_len, window_size, workers, sim_metric, num_noise, neg_test_edges):
		self.G = G
		self.d_graph = d_graph
		self.num_walks = num_walks
		self.walk_len = walk_len
		self.window_size = window_size
		self.workers = workers
		self.sim_metric = sim_metric
		self.num_noise = num_noise
		self.neg_test_edges = neg_test_edges
		self.walks = self._generate_walks()


	def _generate_walks(self):
		"""
		Runs the random walk generator parallelly.

		Returns:
			list of random walks
		"""
		flatten = lambda l: [item for sublist in l for item in sublist]
		num_walks_lists = np.array_split(range(self.num_walks), self.workers)

		walk_results = Parallel(n_jobs= self.workers)(
			delayed(parallel_generate_walks)(self.d_graph,
											 self.walk_len,
											 len(num_walks),
											 idx, self.sim_metric, ) for
			idx, num_walks in enumerate(num_walks_lists))

		return flatten(walk_results)


	def generate_samples(self):
		"""
		Generates positive and negative sample multisets for each node. Positive
		samples are selected from the nodes inside the context window, negative
		samples are selected randomly from the non-neighbour node pairs.

		Returns:
			positive and negative sample multisets
		"""
		negative_samples = {}
		positive_samples = {}
		nodes = list(self.d_graph.keys())
		pbar = tqdm(total=len(nodes), desc='Negative Sampling', ncols=100)

		for node in nodes:
			positive_samples[node] = []
			negative_samples[node] = []

		for walk in self.walks:
			walk_len = len(walk)
			for start in range(walk_len - 1):
				u = walk[start]
				context_window = walk[start + 1: min(start + self.window_size + 1, walk_len)]
				positive_samples[u] += context_window

		for node in nodes:
			pbar.update(1)
			test_set = set([ t for s, t in self.neg_test_edges if s == node ])
			option_list = set(nx.non_neighbors(self.G, node)).difference(set(positive_samples[node]) | test_set)
			negative_samples[node] = random.choices(list(option_list), k=self.num_noise * len(positive_samples[node]))

		pbar.close()

		return positive_samples, negative_samples
