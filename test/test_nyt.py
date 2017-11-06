# Data
import sys
import numpy as np
sys.path.append('../')
from model.online_lda import OnlineLDAVB
from model.predictive import Predictive
from preprocessing.read import read, read_n_docs
from preprocessing.dictionary import read_inverse_dic
from test import train_model, evaluate_model, make_result_dir, save_model, save_top_words
import math

def main(train=True, evaluate=True, model_file=None):
	# Params
	k = 100
	alpha = 0.1
	kappa = 0.5
	tau0 = 64
	var_i = 50
	size = 2000
	V = read_count('../../dataset/nyt/nytimes_voca_count.txt')
	if V == None:
		return
	# Model
	if model_file:
		ldaModel = OnlineLDAVB.load(model_file)
	else:
		ldaModel = OnlineLDAVB(alpha=alpha, K=k, V=V, kappa=kappa, tau0=tau0,\
				batch_size=size, var_max_iter=var_i)
	# Result directory
	result_dir = make_result_dir(ldaModel, '../result/nyt/')
	# Train
	if train:
		# Number of lines each time
		n = 2000
		# Number of documents
		D = read_count('../../dataset/nyt/document_counts.txt')
		if D == None:
			return
		# File
		corpus_file = open('../../dataset/nyt/nytimes_row_200k.txt')
		for i in xrange(int(math.ceil(D/n))):
			# Read n docs
			W = read_n_docs(corpus_file, n)
			# Train
			train_model(ldaModel, W, result_dir)
		# Save model
		save_model(ldaModel, result_dir)	
		# Save stop words
		save_top_words(ldaModel, read_inverse_dic('../../dataset/nyt/vocab.nytimes.txt'), \
				result_dir)
		# Close file
		corpus_file.close()

	# Evaluate
	if evaluate:
		# Read test set
		W_tests = []
		for i in range(1, 11):
			W_obs = read('../../dataset/nyt/data_test_%d_part_1' % i)
			W_he = read('../../dataset/nyt/data_test_%d_part_2' % i)
			ids = [j for j in range(len(W_obs)) if W_obs[j].num_words > 0 \
					and W_he[j].num_words > 0]
			W_obs = np.array(W_obs)[ids]
			W_he = np.array(W_he)[ids]
			W_tests.append((W_obs, W_he))
		# Evaluate and save result
		evaluator = Predictive(ldaModel)
		evaluate_model(evaluator, W_tests, result_dir)

def read_count(filename):
	with open(filename, 'r') as f:
		arr = f.read().strip().split()
		for x in arr:
			if (is_int(x)):
				return int(x)
		return None

def is_int(x):
	try:
		x = int(x)
		x += 1
		return True
	except (TypeError, ValueError) as e:
		return False

main(evaluate=False)