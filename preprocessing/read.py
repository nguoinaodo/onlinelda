import numpy as np
from model.document import Document
from dictionary import read_vocab
from scipy.sparse import coo_matrix

# Sparse document to type Document
def to_documents(lines):
	documents = []
	for l in lines:
		a = l.strip().split(' ')
		num_terms = int(a[0]) # number of unique terms
		terms = []
		counts = []
		num_words = 0
		# Add word to doc
		for t in a[1:]:
			b = t.split(':')
			w = int(b[0]) # term
			n_w = int(b[1]) # number of occurrence
			terms.append(w)
			counts.append(n_w)
			num_words += n_w
		# Add doc to corpus
		doc = Document(num_terms, num_words, terms, counts)
		documents.append(doc)
	return documents

# To documents: sparse matrix
def to_sparse_matrix(lines, V):
	D = len(lines)
	row = []
	col = []
	data = []
	for d in range(D):
		a = lines[d].strip().split(' ')
		for t in a[1:]:
			b = t.split(':')
			term = int(b[0])
			count = int(b[1])
			row.append(d)
			col.append(term)
			data.append(count)

	return coo_matrix((data, (row, col)), shape=(D, V))

# Read documents
def read(filename):
	f = open(filename, 'r')
	# Read lines
	lines = f.readlines()
	f.close()
	docs = to_documents(lines)

	return docs

# Read documents and split train test
def read_split(filename):
	f = open(filename, 'r')
	# Read lines
	lines = f.readlines()
	D = len(lines)
	D_train = 3 * D /4
	D_test = D - D_train
	train = lines[: D_train]
	test = lines[D_train:]
	# To documents
	docs = to_documents(lines)
	docs_train = to_documents(train)
	docs_test = to_documents(test)

	return docs, docs_train, docs_test

def read_n_docs(file, n):
	lines = []
	for i in xrange(n):
		l = file.readline()
		if l != None:
			lines.append(l.strip())
		else: 
			break
	docs = to_documents(lines)
	return docs
