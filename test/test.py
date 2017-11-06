import numpy as np 
import os
import time
import datetime
import matplotlib.pyplot as plt
import math

def train_model(ldaModel, W, result_dir):
	# Log
	log_file = get_log_file(result_dir)
	if (os.path.exists(log_file)):
		log = open(log_file, 'a')
	else:
		log = open(log_file, 'w')
		ldaModel.log_info(log)	
	# Fit
	D = len(W)
	random_ids = np.random.permutation(D)
	batch_size = ldaModel.batch_size
	batchs = range(int(math.ceil(D/batch_size)))	
	log_print(log, 'Start: %s' % datetime.datetime.now())
	for t in batchs:
		log_print(log, "Minibatch %d" % ldaModel.t)
		# Batch documents id
		batch_ids = random_ids[t * batch_size: (t + 1) * batch_size]
		mbstart = time.time() # Start minibatch
		ldaModel.fit(W, batch_ids)
		# End minibatch
		log_print(log, 'Minibatch time: %d' % (time.time() - mbstart))
	# Close file
	log.close()
	

def save_model(ldaModel, result_dir):
	# Save model
	model_file = get_model_file(result_dir)
	ldaModel.save(model_file)

def log_print(log, content):
	print(content)
	log.write('%s\n' % content)

def save_top_words(ldaModel, inv_dic, result_dir):
	top_words_file = get_top_words_file(result_dir)
	top_idxs = ldaModel.get_top_words_indexes()
	with open(top_words_file, 'w') as f:
		for i in range(len(top_idxs)):
			s = '\nTopic %d:' % i 
			for idx in top_idxs[i]:
				s += ' %s' % inv_dic[idx]
			f.write(s)

def evaluate_model(evaluator, W_tests, result_dir):
	evstart = time.time() # Evaluation start
	# Calculate predictive
	predictive = calc_predictive(evaluator, W_tests)
	log_print(log, 'Evaluate time: %d' % (time.time() - evstart))
	log_print(log, 'Predictive: %f' % predictive)
	# Save predictive
	save_predictive(predictive, result_dir)

def save_predictive(predictive, result_dir):
	predictive_file = get_predictive_file(result_dir)
	with open(predictive_file, 'w') as f:
		f.write('%f' % predictive)

def calc_predictive(evaluator, W_tests):
	predictives = []
	for item in W_tests:
		if len(item) == 2:
			W_obs = item[0]
			W_he = item[1]
			predictive = evaluator.predictive_splitted(W_obs, W_he)
		else:
			predictive = evaluator.predictive(item)
		predictives.append(predictive)
	return np.average(predictives)

def make_result_dir(ldaModel, dirname):
	info = 'var%d-batchsize%d-topics%d-alpha%.2f-kappa%.2f-tau0%d' \
			% (ldaModel.var_max_iter, ldaModel.batch_size, ldaModel.K, ldaModel.alpha, \
			ldaModel.kappa, ldaModel.tau0)
	result_dir = dirname + info
	make_dir(result_dir)
	return result_dir

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_log_file(result_dir):
	return '%s/%s' % (result_dir, 'log.txt')

def get_model_file(result_dir):
	return '%s/%s' % (result_dir, 'model.csv')

def get_top_words_file(result_dir):
	return '%s/%s' % (result_dir, 'top_words.txt')

def get_plot_file(result_dir):
	return '%s/%s' % (result_dir, 'plot.png')

def get_predictive_file(result_dir):
	return '%s/%s' % (result_dir, 'predictive.png')









def files(result_dir):
	log_file = '%s/%s' % (result_dir, 'log.txt')
	model_file = '%s/%s' % (result_dir, 'model.csv')
	top_words_file = '%s/%s' % (result_dir, 'top_words.txt')
	plot_file = '%s/%s' % (result_dir, 'plot.png')
	predictive_file = '%s/%s' % (result_dir, 'predictive.txt')
	return log_file, model_file, top_words_file, plot_file, predictive_file

def test(ldaModel, evaluator, K, V, W, W_tests, dirname, dic, inv_dic,\
		predictive_per_minibatch=True):
	# Model info
	batch_size = ldaModel.batch_size
	info = 'var%d-batchsize%d-topics%d-alpha%.2f-kappa%.2f-tau0%d' \
			% (ldaModel.var_max_iter, ldaModel.batch_size, K, ldaModel.alpha, \
			ldaModel.kappa, ldaModel.tau0)
	# Make dir
	result_dir = dirname + info
	make_dir(result_dir)
	# File name
	log_file, model_file, top_words_file, plot_file, predictive_file = \
			files(result_dir)

	# Log
	if (os.path.exists(log_file)):
		log = open(log_file, 'a')
	else:
		log = open(log_file, 'w')
		ldaModel.log_info(log)		

	# Fit and evaluate
	D = len(W)
	random_ids = np.random.permutation(D)
	batchs = range(int(math.ceil(D/batch_size)))
	if W_tests:
		predictives = read_predictives(predictive_file) # past predictive
	log_print(log, 'Start: %s' % datetime.datetime.now())
	start = time.time() # Start
	for t in batchs:
		log_print(log, "Minibatch %d" % t)
		# Batch documents id
		batch_ids = random_ids[t * batch_size: (t + 1) * batch_size]
		mbstart = time.time() # Start minibatch
		ldaModel.fit(W, batch_ids)
		# End minibatch
		log_print(log, 'Minibatch time: %d' % (time.time() - mbstart))
		
		# Predictive per minibatch
		if predictive_per_minibatch and W_tests:
			evstart = time.time() # Evaluation start
			predictive = calc_predictive(evaluator, W_tests)
			log_print(log, 'Evaluate time: %d' % (time.time() - evstart))
			log_print(log, 'Predictive: %f' % predictive)
			predictives.append(predictive)	
	# Final predictive
	if predictive_per_minibatch == False and W_tests:
		evstart = time.time() # Evaluation start
		predictive = calc_predictive(evaluator, W_tests)
		log_print(log, 'Evaluate time: %d' % (time.time() - evstart))
		log_print(log, 'Final predictive: %f' % predictive)
		predictives = [predictive]	
	# End
	log_print(log, 'Runtime: %d' % (time.time() - start))
	log.close()	
	# Save model
	ldaModel.save(model_file)		
	# Top words
	top_idxs = ldaModel.get_top_words_indexes()
	save_top_words(inv_dic, top_idxs, top_words_file)
	# Save predictives
	if W_tests:
		save_predictives(predictives, predictive_file)
	# Plot
	plot_2d(x=range(ldaModel.t), y=predictives, xlabel='Minibatch', \
			ylabel='Per-words log predictive', plotfile=plot_file)
				






def save_predictives(predictives, predictive_file):
	with open(predictive_file, 'w') as f:
		for p in predictives:
			f.write('%s\n' % str(p))

# Plot
def plot_2d(x, y, xlabel=None, ylabel=None, plotfile=None):
	plt.plot(x, y, 'b')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if plotfile:
		plt.savefig(plotfile)

# Read past predictive
def read_predictives(predictive_file):
	if os.path.exists(predictive_file):
		with open(predictive_file, 'r') as f:
			lines = f.readlines()
			predictives = np.array(lines).astype(np.float)
			return predictives
	else:
		return []

# Append predictives
def append_predictives(predictives, predictive_file):
	with open(predictive_file, 'a') as f:
		for p in predictives:
			f.write('%s\n' % str(p))

