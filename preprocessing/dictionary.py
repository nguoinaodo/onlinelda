"""
Read dictionary:
word1
word2
...
"""
def read_vocab(filename):
	f = open(filename, 'r')
	# Read lines
	lines = f.readlines()
	f.close()
	V = len(lines)
	# Dictionary
	dictionary = {}
	inverse_dictionary = {}
	terms = []
	for i in range(V):
		t = lines[i].strip()
		terms.append(t)
		dictionary[t] = i
		inverse_dictionary[i] = t
	return V, dictionary, inverse_dictionary

def read_inverse_dic(filename):
	f = open(filename, 'r')
	# Read lines
	lines = f.readlines()
	f.close()
	V = len(lines)
	# Dictionary
	inverse_dictionary = {}
	for i in range(V):
		t = lines[i].strip()
		inverse_dictionary[i] = t
	return inverse_dictionary
