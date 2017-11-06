import numpy as np
from scipy.special import digamma, gamma, gammaln
import math
from scipy.sparse import coo_matrix
import time
from document import Document
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from predictive import Predictive
import csv

def normalize(X, axis):
	if axis == 0:
		# Column nomalize
		X1 = 1. * X / np.sum(X, axis=0) 
	elif axis == 1:
		# Row normalize
		X1 = 1.* X / np.sum(X, axis=1).reshape(X.shape[0], 1)
	else:
		return X
	return X1

class OnlineLDAVB:
	def __init__(self, alpha=0.1, beta=None, tau0=None, kappa=None, \
				K=None, V=None, predictive_ratio=.8, \
				var_converged=1e-6, var_max_iter=50, batch_size=100, t=0):
		self.K = K # Number of topics
		self.V = V # Dictionary size
		self.alpha = alpha # Dirichlet parameters of topics distribution 
		# Topic - term probability
		if (beta):
			self.beta = beta 
		else:
			self._init_beta()
		self.kappa = kappa # Control the rate old values of beta are forgotte
		self.tau0 = tau0 # Slow down the early stop iterations of the algorithm
		self.predictive_ratio = predictive_ratio # Predictive observed - held-out ratio
		self.var_converged = var_converged 
		self.var_max_iter = var_max_iter
		self.batch_size = batch_size # Batch size
		self.t = t

	# Get parameters for this estimator.
	def get_params(self):
		return self.K, self.V, self.alpha, self.tau0, self.kappa, \
				self.predictive_ratio, self.var_converged, self.var_max_iter, \
				self.batch_size, self.t

	# Init beta	
	def _init_beta(self):
		# Multinomial parameter beta: KxV
		self.beta = normalize(np.random.gamma(100, 1./100, (self.K, self.V)), axis=1)

	# Init beta corpus
	def _init_beta_corpus(self, W, D):
		self.beta = np.zeros((self.K, self.V))
		num_doc_per_topic = 5

		for i in range(num_doc_per_topic):
		    rand_index = np.random.permutation(D)
		    for k in range(self.K):
		        d = rand_index[k]
		        doc = W[d]
		        for n in range(doc.num_terms):
		            self.beta[k][doc.terms[n]] += doc.counts[n]
		self.beta += 1
		self.beta = normalize(self.beta, axis=1)

	def log_info(self, log):
		log.write('---------------------------------\n')
		log.write('Online LDA:\n')
		log.write('Number of topics: %d\n' % self.K)
		log.write('Number of terms: %d\n' % self.V)
		log.write('Batch size: %d\n' % self.batch_size)
		log.write('alpha=%f\n' % self.alpha)
		log.write('tau0=%f\n' % self.tau0)
		log.write('kappa=%f\n' % self.kappa)
		log.write('var_converged=%f\n' % self.var_converged)
		log.write('var_max_iter=%d\n' % self.var_max_iter)
		log.write('----------------------------------\n')

	# Fit data
	def fit_batch(self, W):
		"""
			W: list of documents
		"""	
		# Run EM
		self._em(W)

	# Fit minibatch
	def fit(self, W, batch_ids):
		self._em_minibatch(W, batch_ids)

	def _em_minibatch(self, W, batch_ids):
		# Estimation for minibatch
		suff_stat = self._estimate(W, batch_ids)
		# Update beta
		beta_star = self._maximize(suff_stat) # intermediate
		ro_t = (self.tau0 + self.t) ** (-self.kappa) # update weight
		self.beta = (1 - ro_t) * self.beta + ro_t * beta_star	
		# Time
		self.t += 1
		
	# EM with all documents
	def _em(self, W):
		D = len(W)
		# Permutation
		random_ids = np.random.permutation(D)
		# For minibatch
		batchs = range(int(math.ceil(D/self.batch_size)))
		for i in batchs:
			# Batch documents id
			batch_ids = random_ids[i * self.batch_size: (i + 1) * self.batch_size]
			# EM minibatch
			self._em_minibatch(W, batch_ids)

	# Init variational parameters for each document
	def _doc_init_params(self, W_d):
		phi_d = np.ones((W_d.num_words, self.K)) / self.K
		gamma_d = (self.alpha + 1. * W_d.num_words / self.K) * np.ones(self.K)	
		return phi_d, gamma_d	

	# Estimate batch
	def _estimate(self, W, batch_ids):
		# Init sufficiency statistic for minibatch
		suff_stat = np.zeros(self.beta.shape)
		# For document in batch
		for d in batch_ids:
			# Estimate doc
			phi_d, gamma_d, W_d = self._estimate_doc(W, d)
			# Update sufficiency statistic
			for j in range(W[d].num_words):
				for k in range(self.K):
					suff_stat[k][W_d[j]] += phi_d[j][k]
		return suff_stat

	def _estimate_doc(self, W, d):
		# Document flatten
		W_d = W[d].to_vector()	
		# Init variational parameters
		phi_d, gamma_d = self._doc_init_params(W[d])

		# Coordinate ascent
		old_gamma_d = gamma_d
		for i in range(self.var_max_iter):
			# Update phi
			phi_d = normalize(self.beta.T[W_d, :] * np.exp(digamma(gamma_d)), axis=1)
			# Update gamma
			gamma_d = self.alpha + np.sum(phi_d, axis=0)

			# Check convergence
			meanchange = np.mean(np.fabs(old_gamma_d - gamma_d))
			if meanchange < self.var_converged:
				break
			old_gamma_d = gamma_d
		return phi_d, gamma_d, W_d

	# Update global parameter
	def _maximize(self, suff_stat):
		return normalize(suff_stat, axis=1) + 1e-100

	# Get top words of each topics
	def get_top_words_indexes(self):
		top_idxs = []
		# For each topic
		for t in self.beta:
			desc_idx = np.argsort(t)[::-1]
			top_idx = desc_idx[:20]
			top_idxs.append(top_idx)
		return np.array(top_idxs)	

	# Inference new docs
	def infer(self, W, D):
		phi = []
		var_gamma = []
		for d in range(D):
			phi_d, gamma_d, W_d = self._estimate_doc(W, d)
			phi.append(phi_d)
			var_gamma.append(gamma_d)
		return phi, var_gamma	
		
	# Save model
	def save(self, filename):
		with open(filename, 'w') as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerow(self.get_params())
			writer.writerows(self.beta)

	# Load model
	def load(filename):
		with open(filename, 'r') as f:
			# Read alpha, kappa, tau0
			arr = np.array(f.readline().split(',')).astype(np.float)
			K = arr[0]
			V = arr[1]
			alpha = arr[2]
			tau0 = arr[3]
			kappa = arr[4]
			predictive_ratio = arr[5]
			var_converged = arr[6]
			var_max_iter = arr[7]
			batch_size = arr[8]
			t = arr[9]
			# Read beta
			arr = f.readlines()
			for line in arr:
				beta.append(line.split(','))
			beta = np.array(beta).astype(np.float)
			# Model
			model = OnlineLDAVB(alpha=alpha, beta=beta, tau0=tau0, kappa=kappa, \
					K=K, V=V, predictive_ratio=predictive_ratio,
					var_converged=var_converged, var_max_iter=var_max_iter, batch_size=batch_size, t=t)
		return model

