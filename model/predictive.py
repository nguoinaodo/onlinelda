from document import Document
import numpy as np

class Predictive:
	def __init__(self, ldaModel, ratio=None):
		self.ldaModel = ldaModel
		if ratio:
			self.ratio = ratio
		else:
			self.ratio = ldaModel.predictive_ratio

	# Split observed - held-out
	def split_observed_heldout(self, W):
		D = len(W)
		W_obs = []
		W_he = []
		# Split document to observed and held-out
		for d in range(D):
			W_d = W[d].to_vector()
			N_d = W[d].num_words
			i = 0
			count_obs = 0
			while i < W[d].num_terms and 1. * (count_obs + W[d].counts[i]) / N_d < self.ratio:
				count_obs += W[d].counts[i]
				i += 1
			W_d_obs = Document(i, count_obs, W[d].terms[: i], W[d].counts[: i])
			W_d_he = Document(W[d].num_terms - i, N_d - count_obs, W[d].terms[i:], \
					W[d].counts[i:])
			if W_d_obs.num_words > 0 and W_d_he.num_words > 0:
				W_obs.append(W_d_obs)
				W_he.append(W_d_he)
		return W_obs, W_he

	# Predictive distribution
	def _predictive(self, W_obs, W_he):
		sum_log_prob = 0 # Sum of log of P(w_new|w_obs, W)
		num_new_words = 0 # Number of new words
		# Infer
		phi, var_gamma = self.ldaModel.infer(W_obs, len(W_obs))
		# Per-word log probability
		sum_log_prob = 0
		D = len(W_he)
		for d in range(D):
			doc_log_prob = 0
			num_new_words = W_he[d].num_words
			for i in range(W_he[d].num_terms):
				doc_log_prob += W_he[d].counts[i] * \
						np.log(1. * var_gamma[d].dot(self.ldaModel.beta[:, W_he[d].terms[i]]) / \
						np.sum(var_gamma[d]))
			sum_log_prob += 1. * doc_log_prob / num_new_words
		result = 1. * sum_log_prob / D
		return result	

	def predictive(self, W):
		# Split 
		W_obs, W_he = self.split_observed_heldout(W)
		return self._predictive(W_obs, W_he)

	def predictive_splitted(self, W_obs, W_he):
		return self._predictive(W_obs, W_he)