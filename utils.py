import datetime
from datetime import date
from tqdm import tqdm

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_processing import is_english, tokenize, remove_stopwords


def get_year_quarters(year):
    """ 
    Gives the start and end dates of each quarter for the input year.
    
    Parameters
    ----------
    year : int or str
        Year

    Returns
    -------
    List of 4 pair dates, one for each quarter.
    """
    quarters = []
    m = ['01','04','07','10','01']
    for i in range(len(m)-1):
        start = datetime.datetime.strptime(str(year) + '-' + m[i] + '-01', '%Y-%m-%d').date()
        if i==3: year+=1
        end = datetime.datetime.strptime(str(year) + '-' + m[i+1] + '-01', '%Y-%m-%d').date()
        quarters.append((start,end))
        
    return quarters




def get_human_readable_quarter_name(date_str):
    """ 
    Produces human readable name for quarters.
    
    Parameters
    ----------
    date_str : str
        Start date of a quarter which is expected to be of format 'dd-mm-yyyy'.

    Returns
    -------
    String of human readable quarter name.  
    """
    mapping = {'01':'1st','04':'2nd','07':'3rd','10':'4th'}
    return date_str[6:] + ',' + mapping[date_str[3:5]] + ' qrt.'




def top_words_per_quarter(time_stamped_data, docs, years, n_words=3):
    """ 
    Computes top-n words/phrases with corresponding frequences per each quarter for the given list of years. 
    
    Parameters
    ----------
    time_stamped_data : pandas.DataFrame or pandas.Series
        Pandas DataFrame or Series with column name 'date' containing one-to-one correspondance time stamp ('dd-mm-yyyy') for each document.

    docs : list
        List of tokenized strings.

    years : list
        List of years to consider for computing top-n words/phrases.

    n_words : int, optional
        Number of top words to consider.

    Returns
    -------
    Dictionary where keys are the quarter start dates and values are the corresponding list of top-n words/phrases with their frequences. 
    """
    top_words = defaultdict(list)

    for y in years: 
        qrt = get_year_quarters(y)

        for i in range(4):
            key = qrt[i][0].strftime('%d-%m-%Y')

            indices = sorted(time_stamped_data[(qrt[i][0]<=time_stamped_data.date) & (time_stamped_data.date<qrt[i][1])].index)
            words = [w for e in indices for w in docs[e]]

            if len(words)>0:
                c = Counter(words)
                c_sort = {k: v for k, v in sorted(c.items(), key=lambda item: -item[1])}
                top_words[key] = list(c_sort.items())[:n_words] 

    return top_words
    



def top_words_freq_history(time_stamped_data, docs, years, n_words=5):
    """ 
    Computes the frequencies of each of the top-n words/phrases per quarter for the given list of years. 
    
    Parameters
    ----------
    time_stamped_data : pandas.DataFrame or pandas.Series
        Pandas DataFrame or Series with column name 'date' containing one-to-one correspondance time stamp ('dd-mm-yyyy') for each document.

    docs : list
        List of tokenized strings.

    years : list
        List of years to consider for computing top-n words'/phrases' frequences per quarter.

    n_words : int, optional
        Number of top words to consider.

    Returns
    -------
    Dictionary where keys are the words/phrases and values are dictionaries with quarter-start-date and frequency as key-value pairs.
    """
    top_n_words_freq = defaultdict(dict)

    words = [word for doc in docs for word in doc]
    count = Counter(words)
    count_srt = {k: v for k, v in sorted(count.items(), key=lambda item: -item[1])}

    top_n_words = list(count_srt.keys())[:n_words]

    for y in years:
        qrt = get_year_quarters(y)

        for i in range(4):
            key = qrt[i][0].strftime('%d-%m-%Y')

            indices = sorted(time_stamped_data[(qrt[i][0]<=time_stamped_data.date) & (time_stamped_data.date<qrt[i][1])].index)
            words = [w for e in indices for w in docs[e]]

            if len(words)>0:
                c = Counter(words)

                for w in top_n_words:
                    if w in c:
                        top_n_words_freq[w][key] = c[w]
                    else:
                        top_n_words_freq[w][key] = 0

    return top_n_words_freq




def read_categories(FILE_PATH):
    """ 
    Reads and returns predefined categories with their corresponding context words from the input file.
    Input file is expected to be txt and of format {category}/{context word} on each line.
    
    Parameters
    ----------
    FILE_PATH : str
        Path of the txt file containing categories and the corresponding context words. 

    Returns
    -------
    Dictionary with category names as the keys and list of the corresponding context words as the values. 
    """
    categories_dict = defaultdict(list)

    with open(FILE_PATH, 'r') as f:
        for line in f.readlines():
            ctg, cntx_word = line.split('\\')
            categories_dict[ctg].append(cntx_word.rstrip())

    return categories_dict




def doc2vec(doc_tok, word2vec_model, vec_dim=300):
    """
    Mapping a tokenized document to a single vector through averaging. 

    Parameters
    ----------
    doc_tok : list
        Tokenized string.

    word2vec_model : gensim.Word2VecKeyedVectors
        Word2vec model.

    vec_dim: int, optional
        Dimension of word vectors. Must be the same as in the word2vec_model. 

    Returns
    -------
    A numpy array of dimension vec_dim representing the input document. 
    """
    doc_2_vec = np.zeros(vec_dim)

    i = 0
    for w in doc_tok:
        if w in word2vec_model.vocab.keys():
            doc_2_vec += word2vec_model.get_vector(w)
            i+=1
    if i>0:
        return doc_2_vec/i
    return np.array([])




def sim_with_category(doc_vec, category, word2vec_model, categories_dict):
    """
    Computes the cosine similarity of the input document with the given category (e.g. Technology, Entertainment).
    The cosine similarity is computed between the input document vector and each of the context words corresponding to the input category and averaged at the end.

    Parameters
    ----------
    doc_vec : numpy.ndarray
        Vector representing the document.

    category : str
        Category name from the following list: ['Technology', 'Entertainment', 'Education', 'Health', 'Business', 'Law & Politics', 'Living', 'Finance']

    word2vec_model : gensim.Word2VecKeyedVectors
        Word2vec model.

    categories_dict : dict or collections.defaultdict
        Dictionary with category names as the keys and list of the corresponding context words as the values. 

    Returns
    -------
    An integer representing how similar the input document is to the input category with respect to cosine similarity.
    """
    sim_score = 0

    n = len(categories_dict[category])
    for context_word in categories_dict[category]:
        sim_score+=cosine_similarity([doc_vec,word2vec_model.get_vector(context_word)])[0][1]
    
    return sim_score/n




def sim_with_category_v2(doc_vec, category, word2vec_model, categories_dict):
    """
    Computes the cosine similarity of the input document with the given category (e.g. Technology, Entertainment).
    The cosine similarity is computed between the input document vector and the average vector of the context words corresponding to the input category.

    Parameters
    ----------
    doc_vec : numpy.ndarray
        Vector representing the document.

    category : str
        Category name from the following list: ['Technology', 'Entertainment', 'Education', 'Health', 'Business', 'Law & Politics', 'Living', 'Finance']

    word2vec_model : gensim.Word2VecKeyedVectors
        Word2vec model.

    categories_dict : dict or collections.defaultdict
        Dictionary with category names as the keys and list of the corresponding context words as the values. 

    Returns
    -------
    An integer representing how similar the input document is to the input category with respect to cosine similarity.
    """
    dim = len(word2vec_model.get_vector(categories_dict[category][0]))
    shape = (len(categories_dict[category]),dim)
    context_words_vectors = np.zeros(shape)

    i = 0
    for context_word in categories_dict[category]:
        context_words_vectors[i] = word2vec_model.get_vector(context_word)

    sim_score = cosine_similarity([doc_vec,context_words_vectors.mean(axis=0)])[0][1]
    return sim_score




def doc_categ_matching(docs, word2vec_model, categories_dict, sim_method, thr=0.2):
    """
    Maps each document in the given list of documents to the best matching (closest) category.

    Parameters
    ----------
    docs : list
        List of strings.

    word2vec_model : gensim.Word2VecKeyedVectors
        Word2vec model.

    categories_dict : dict or collections.defaultdict
        Dictionary with category names as the keys and list of the corresponding context words as the values. 

    method : function
        Specifies the method for finding the best matching category.

    thr : int, optional
        A document having its highest similarity score less than the value of thr is ignored and is not considered close to any of the predefined categories.

    Returns
    -------
    Dictionary with categories as the keys and matched documents with their corresponding similarity scores as the values. 
    """

    docs_eng = list(map(lambda q: q if is_english(q) else '', docs)) # keep only English queries
    docs_tok = tokenize(docs_eng)
    docs_tok = remove_stopwords(docs_tok)

    doc2categ_mapping = defaultdict(list)

    for j, doc in tqdm(enumerate(docs_tok)):
        doc_vec = doc2vec(doc, word2vec_model)

        if len(doc_vec)>0:
            cos_sim = -2
            for categ in categories_dict.keys():
                sim_score = sim_method(doc_vec, categ, word2vec_model, categories_dict)
                if sim_score>cos_sim:
                    best_categ = categ
                    cos_sim = sim_score

            if cos_sim>thr:
                doc2categ_mapping[best_categ].append((docs[j],cos_sim))
    
    return doc2categ_mapping




