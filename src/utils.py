import argparse
from tqdm import tqdm

import numpy as np
from numpy.linalg import norm
import networkx as nx
import random
from sklearn.model_selection import KFold



def parameter_parser():
	"""
	Parses the command-line arguments
	"""
	parser = argparse.ArgumentParser(description="Run CS-Trust2Vec")
	parser.add_argument("--graph-path",
						nargs="?",
						default="./data/networkx_graph.pkl",
						help="Path to the context-labeled trust network.")
	parser.add_argument("--ldim",
						type=int,
						default=48,
						help="Dimension of the latent feature. Default is 48.")
	parser.add_argument("--ctdim",
						type=int,
						default=28,
						help="Dimension of the context tendency feature. Default is 28.")
	parser.add_argument("--n",
						type=int,
						default=3,
						help="Number of noise samples. Default is 3.")
	parser.add_argument("--window-size",
						type=int,
						default=5,
						help="Context window size. Default is 5.")
	parser.add_argument("--num-walks",
						type=int,
						default=20,
						help="Walks per node. Default is 20.")
	parser.add_argument("--walk-len",
						type=int,
						default=10,
						help="Length per walk. Default is 10.")
	parser.add_argument("--workers",
						type=int,
						default=4,
						help="Number of threads for random walking. Default is 4.")
	parser.add_argument("--epoch",
						type=int,
						default=5,
						help="Number of epochs. Default is 5.")
	parser.add_argument("--sim-metric",
						nargs="?",
						default="cosine",
						help="Similarity metric. Options: cosine, euclidean, jaccard, uniform. Default is cosine")
	parser.add_argument("--learning-rate",
						type=float,
						default=0.02,
						help="Learning rate. Default is 0.02.")
	parser.add_argument("--split-seed",
						type=int,
						default=2,
						help="Random seed for splitting dataset. Default is 2.")

	return parser.parse_args()


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def cosine_similarity(v1, v2):
	norm_mult = norm(v1) * norm(v2)
	return 0 if not norm_mult else (v1 @ v2.T) / norm_mult

def euclidean_similarity(v1, v2):
	distance = norm(v1 - v2)
	return np.reciprocal(distance + 1)

def jaccard_similarity(v1, v2):
	v_union = np.sum(np.logical_or(v1, v2))
	return 0 if not v_union else np.sum(np.logical_and(v1, v2)) / v_union


def choose_next_node(previous_edge, walk_dict, succs, sim_metric):
	"""
	Selects the next node of the walk randomly from a probability distribution.

	Parameters:
		previous_edge (ndarray): context label of the previous edge.
		walk_dict (dict): dictionary that contains the context labels
		of the outgoing edges.
		succs (list): list of the successor nodes.
		sim_metric (string): similarity measure used to determine the
		weights of the probability distribution.

	Returns:
		id of the selected node
	"""

	if(sim_metric == "cosine"):
		weight_list = [cosine_similarity(previous_edge, walk_dict[v]) for v in succs]
	elif(sim_metric == "euclidean"):
		weight_list = [euclidean_similarity(previous_edge, walk_dict[v]) for v in succs]
	elif(sim_metric == "jaccard"):
		weight_list = [jaccard_similarity(previous_edge, walk_dict[v]) for v in succs]
	elif(sim_metric == "uniform"):
		return succs[random.randint(0, len(succs)-1)]


	# If all weights are zero, randomly choose from the uniform distribution
	if all([w == 0 for w in weight_list]):
		index = random.randint(0, len(succs)-1)
	else:
		# normalize the weights
		weight_list /= np.sum(weight_list)
		index = np.random.choice(list(range(len(succs))), p=weight_list)

	return succs[index]


def parallel_generate_walks(d_graph, walk_len, num_walks, cpu_num, sim_metric):
	"""
	Generates random walks.

	Parameters:
		d_graph (dict)
		walk_len (integer): maximum length of the random walk
		num_walks (integer): number of random walks per node
		cpu_num (integer): thread id
		sim_metric (string): similarity measure

	Returns:
		list of random walks
	"""

	walks = list()
	pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

	for n_walk in range(num_walks):
		pbar.update(1)

		# Shuffle the nodes
		shuffled_nodes = list(d_graph.keys())
		random.shuffle(shuffled_nodes)

		for source in shuffled_nodes:

			walk = [source]
			previous_edge = []

			while len(walk) < walk_len:

				walk_dict = d_graph[walk[-1]]
				succs = list(walk_dict.keys())

				# Skip dead end nodes
				if not walk_dict:
					break

				if previous_edge == []:
					walk_to = succs[random.randint(0, len(succs)-1)]
				else:
					walk_to = choose_next_node(previous_edge, walk_dict, succs, sim_metric)

				previous_edge = walk_dict[walk_to]
				walk.append(walk_to)

			if len(walk) > 2:
				walks.append(walk)

	pbar.close()
	
	return walks


def sample_edge_list(edge_iter, indices):
	index_it = iter(sorted(indices))
	cur_index = next(index_it)
	edge_list = []

	for i, item in enumerate(edge_iter):
		if i == cur_index:
			edge_list.append(item)
			try:
				cur_index = next(index_it)
			except StopIteration:
				break

	random.shuffle(edge_list)
	return edge_list


def test_split(G, n, seed=None, shuffle=True):
	"""
	Splits the edges of G into k-folds. Each fold consists of 
	positive train set, positive test set, negative train set and 
	negative test set.

	Returns:
		list of folds
	"""

	splitter = KFold(n_splits=n, random_state=seed, shuffle=shuffle)

	node_num = len(G.nodes())
	edge_num = len(G.edges())
	non_edge_num = node_num * (node_num - 1) - edge_num
	non_edge_indices = random.sample(range(non_edge_num), edge_num)

	pos_edges = np.array(G.edges())
	neg_edges = np.array(sample_edge_list(nx.non_edges(G), non_edge_indices))

	splits = []
	pos_idx = list(splitter.split(pos_edges))
	neg_idx = list(splitter.split(neg_edges))
	for i in range(n):
		p_train, p_test = pos_idx[i]
		n_train, n_test = neg_idx[i]
		splits.append((list(map(tuple, pos_edges[p_train])), 
			list(map(tuple, pos_edges[p_test])), 
			list(map(tuple, neg_edges[n_train])), 
			list(map(tuple, neg_edges[n_test]))))

	return splits



