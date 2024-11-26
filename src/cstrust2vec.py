import os
import networkx as nx
import numpy as np
from collections import defaultdict
from utils import parameter_parser, test_split
import pickle
import copy

from preprocessor import Preprocessor
from encoder import Encoder
from predictor import Predictor


class CSTrust2Vec(object):
	"""
	The main module that uses the Preprocessor, Encoder and Predictor modules to 
	generate node embeddings and test the quality of the embeddings.


	Attributes:
		args (object): command-line arguments
		G (networkx graph): context-labeled trust network
		split (tuple): a fold containing positive and negative training-test sets
	"""
	def __init__(self, args, G, split):
		self.args = args
		self.G = copy.deepcopy(G)
		self.pos_train_edges, self.pos_test_edges, self.neg_train_edges, self.neg_test_edges = split
		self.G.remove_edges_from(self.pos_test_edges)

		self.d_graph = defaultdict(dict)

		for node in self.G.nodes():
			self.d_graph[node] = {}
			for _, target_node_id, attrs in self.G.out_edges(node, data=True):
				self.d_graph[node][target_node_id] = attrs['channels']

		self.preprocessor = Preprocessor(self.G, self.d_graph, args.num_walks, args.walk_len, args.window_size,
			args.workers, args.sim_metric, args.n, self.neg_test_edges)

		self.encoder = Encoder(self.d_graph, args.ldim, args.ctdim, list(self.G.nodes()), args.n)


	def train(self):
		"""
		Trains the node embeddings and obtains the optimized
		embedding lookup table.
		"""
		samples = self.preprocessor.generate_samples()
		self.encoder.fit(self.args.epoch, self.args.learning_rate, samples)
		self.W_out, self.W_in = self.encoder.get_embeddings()


	def evaluate(self):
		"""
		Evaluates the performance of the model on the link 
		prediction task.

		Returns:
			macro-micro F1 and AUC scores
		"""
		train_edges = (self.pos_train_edges, self.neg_train_edges)
		test_edges = (self.pos_test_edges, self.neg_test_edges)
		embeddings = (self.W_out, self.W_in)

		link_predictor = Predictor(train_edges, test_edges, embeddings)
		return link_predictor.test()


	def save_emb(self, out_path, in_path):
		"""
		Saves the generated node embeddings.

		Parameters:
			out_path (string): path for the outward embedding
			table.
			in_path (string): path for the inward embedding
			table.
		"""
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		os.makedirs(os.path.dirname(in_path), exist_ok=True)

		np.save(out_path, self.W_out)
		np.save(in_path, self.W_in)


def run():
	"""
	Loads the context-labeled trust network and
	splits the edges of the network into k-folds.

	For each fold, trains the model, saves the generated node embeddings,
	and evaluates the performance of the model.
	"""
	with open(args.graph_path, 'rb') as f:
		G = pickle.load(f, encoding='utf-8')

	try:
		with open("./data/train_test_split.pkl", 'rb') as f:
			train_test_split = pickle.load(f)
	except:
		train_test_split = test_split(G, n=5, seed=args.split_seed, shuffle=True)
		with open("./data/train_test_split.pkl", 'wb') as f:
			pickle.dump(train_test_split , f, pickle.HIGHEST_PROTOCOL)

	for i, fold in enumerate(train_test_split):
		model = CSTrust2Vec(args, G, fold)
		model.train()
		
		out_path = "./embeddings/CSTRUST2VEC_outward_" + str(i)
		in_path = "./embeddings/CSTRUST2VEC_inward_" + str(i)
		model.save_emb(out_path, in_path)

		f1_macro, f1_micro, auc  = model.evaluate()
		print('Fold %i : F1_macro %f, F1_micro %f, AUC %f' % (i, f1_macro, f1_micro, auc))


if __name__ == '__main__':
	args = parameter_parser()
	run()
