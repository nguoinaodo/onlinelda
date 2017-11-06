# Data
import sys
sys.path.append('../')
from model.online_lda import OnlineLDAVB
from model.predictive import Predictive
from preprocessing.read import read
from preprocessing.dictionary import read_vocab
from test import test

def main():
	# Init 
	W = read('../../dataset/gro/grolier-train.txt')
	V, dic, inv_dic = read_vocab('../../dataset/gro/grolier-voca.txt')
	# W_obs_1 = read('../../dataset/gro/data_test_1_part_1.txt')
	# W_he_1 = read('../../dataset/gro/data_test_1_part_2.txt')
	W_test_gro = read('../../dataset/gro/grolier_test_24k.txt')
	V = len(dic) # number of terms
	dirname = '../result/gro/'
	W_tests = []
	# Test
	for var_i in [100]:
		for size in [1000]:
			for k in [100]:
				for alpha in [.1]:
					for kappa in [.5]:
						for tau0 in [64]:
							# Model
							lda = OnlineLDAVB(alpha=alpha, K=k, V=V, kappa=kappa, tau0=tau0,\
									batch_size=size, var_max_iter=var_i)
							evaluator = Predictive(lda) # per word log predictive evaluator
							W_tests = [evaluator.split_observed_heldout(W_test_gro)]
							test(lda, evaluator, k, V, W, W_tests, dirname, dic, inv_dic, \
									predictive_per_minibatch=False)

def predictive(model_file, test_file):
	W_test = read(test_file)
	ldaModel = OnlineLDAVB.load(model_file)
	evaluator = Predictive(ldaModel)
	return evaluator.predictive(W_tests)

main()