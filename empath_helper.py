from empath import Empath
lexicon = Empath()

from tqdm import tqdm
from collections import defaultdict


def get_lexical_categ_counts(text, normalize=False, sort=False):
	"""
	Analyzes the input text across pre-built lexical categories using Empath.

	Parameters
	----------
	text : string
		The text to be analized. 

	normalize : bool, optional
		If true, normalizes the raw counts by the number of words in the text.

	sort : bool, optional
		If true, sorts the resulting dictionary in decreasing order. 

	Returns
	-------
	Dictionary with pre-built lexical categories as the keys and number of matched words to each category as the values.
	"""
	analysis = lexicon.analyze(text, normalize=normalize)
	if sort:
		analysis = {lex_categ: count for lex_categ, count in sorted(analysis.items(), key=lambda item: -item[1])}

	return analysis




def map_docs_to_lexical_categs(docs):
	"""
	Maps each document to one (or several) pre-built lexical categories in Empath. A document is mapped to a category 
	if the document contains at least one word that is matched to that category using Empath.

	Parameters
	----------
	docs : list
		List of strings.

	Returns
	-------
	Dictionary with pre-built lexical categories as the keys and list of mapped documents as the values.
	"""
	mapping = defaultdict(list)
	for doc in tqdm(docs):
		analysis = lexicon.analyze(doc)
		matched_categs = [k for k,v in analysis.items() if v>0]
		for lex_categ in matched_categs:
			mapping[lex_categ].append(doc)

	return mapping




def count_non_empty_results(docs):
	"""
	Counts the number of documents that are matched to at least one of the pre-built lexical categories in Empath.

	Parameters
	----------
	docs : list
		List of strings.

	Returns
	-------
	An integer giving the number of documents matched to at least one of the pre-built lexical categories.
	"""
	count = 0
	for doc in tqdm(docs):
		if len(doc)>0:
			analysis = lexicon.analyze(doc)
			matched_categs = [k for k,v in analysis.items() if v>0]
			if len(matched_categs)>0:
				count+=1

	return count

	


