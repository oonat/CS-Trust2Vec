import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


class Predictor(object):
	"""
	The Predictor is used to evaluate the performance of the model. The module 
	contains a logistic regression model. After the training of the logistic
	regression model with the provided training set, the Predictor makes link
	predictions for the node pairs given in the test set. 

	The module uses the F1 and AUC metrics to evaluate the performance of the
	model on the link prediction task.

	Attributes:
		train_edges (tuple): positive and negative training samples
		test_edges (tuple): positive and negative test samples
		emb (tuple): tuple that contains outward embedding table (W_out)
		and inward embedding table (W_in)
	"""
	def __init__(self, train_edges, test_edges, emb):
		self.pos_train_edges, self.neg_train_edges = train_edges
		self.pos_test_edges, self.neg_test_edges = test_edges
		self.emb = emb
		self.train_x, self.train_y, self.test_x, self.test_y = self._process()


	def _process(self):
		"""
		Prepares the encodings for the edges in the training and test sets.

		Returns:
			training set edge encodings train_x,
			training set ground-truth train_y,
			test set edge encodings test_x,
			test set ground-truth test_y
		"""
		out_emb, in_emb = self.emb
		out_dim = out_emb.shape[1]
		in_dim = in_emb.shape[1]

		train_set = self.pos_train_edges + self.neg_train_edges
		train_x = np.zeros((len(train_set), (out_dim + in_dim) * 2))
		train_y = np.concatenate((np.ones(len(self.pos_train_edges)), np.zeros(len(self.neg_train_edges))))
		for i, edge in enumerate(train_set):
			u = edge[0]
			v = edge[1]
			u_emb = np.concatenate((out_emb[u], in_emb[u]))
			v_emb = np.concatenate((out_emb[v], in_emb[v]))
			train_x[i, : ] = np.concatenate((u_emb , v_emb))

		test_set = self.pos_test_edges + self.neg_test_edges
		test_x = np.zeros((len(test_set), (out_dim + in_dim) * 2))
		test_y = np.concatenate((np.ones(len(self.pos_test_edges)), np.zeros(len(self.neg_test_edges))))
		for i, edge in enumerate(test_set):
			u = edge[0]
			v = edge[1]
			u_emb = np.concatenate((out_emb[u], in_emb[u]))
			v_emb = np.concatenate((out_emb[v], in_emb[v]))
			test_x[i, : ] = np.concatenate((u_emb , v_emb))

		return train_x, train_y, test_x, test_y


	def test(self):
		"""
		Trains the logistic regression model and makes link predictions for the samples
		in the test set.

		Returns:
			macro-micro F1 and AUC scores
		"""
		ss = StandardScaler()
		self.train_x = ss.fit_transform(self.train_x)
		self.test_x = ss.fit_transform(self.test_x)

		lr = LogisticRegression(solver='lbfgs')
		lr.fit(self.train_x, self.train_y)
		test_y_score = lr.predict_proba(self.test_x)[:, 1]
		test_y_pred = lr.predict(self.test_x)

		print(confusion_matrix(self.test_y, test_y_pred))
		macro_f1_score = f1_score(self.test_y, test_y_pred, average='macro')
		micro_f1_score = f1_score(self.test_y, test_y_pred, average='micro')
		auc_score = roc_auc_score(self.test_y, test_y_score, average='macro')

		return macro_f1_score, micro_f1_score, auc_score
