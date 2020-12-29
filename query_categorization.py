from tqdm import tqdm

from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_processing import is_english, tokenize, remove_stopwords, doc2vec


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
    dim = word2vec_model.vector_size
    shape = (len(categories_dict[category]),dim)
    context_words_vectors = np.zeros(shape)

    i = 0
    for context_word in categories_dict[category]:
        context_words_vectors[i] = word2vec_model.get_vector(context_word)

    sim_score = cosine_similarity([doc_vec,context_words_vectors.mean(axis=0)])[0][1]
    return sim_score




def doc_categ_matching(docs, word2vec_model, categories_dict, sim_method, thr=0.2):
    """
    Maps each document in the given list of documents to the best matching(closest) category. 

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




def doc_categ_matching_windowed(docs_df, word2vec_model, categories_dict, sim_method, window=60, thr=0.2):
    """
    Maps each document in the given list of documents to the best matching(closest) category. Implements the same algorithm
    as doc_categ_matching(), in addition taking into account documents' timestamps - documents having timestamps close to 
    each other are grouped and then matched to the same category based on majority voting principle.

    Parameters
    ----------
    docs_df : pandas.DataFrame
        Timestamped documents. The dataframe is expected to contain column names 'time_stamp' and 'query'
        with the latter containing the list of documents and the former - the corresponding timestamps. 

    word2vec_model : gensim.Word2VecKeyedVectors
        Word2vec model.

    categories_dict : dict or collections.defaultdict
        Dictionary with category names as the keys and list of the corresponding context words as the values. 

    method : function
        Specifies the method for finding the best matching category.

    window : int, optional
        Two documents having timestamps difference (in seconds) less than window are put into the same group. 

    thr : int, optional
        A document having its highest similarity score less than the value of thr is ignored and is not considered close to any of the predefined categories.

    Returns
    -------
    Dictionary with categories as the keys and matched documents as the values. 
    """
    docs_raw = docs_df['query'].tolist()
    docs_df_copy = docs_df.copy()

    docs_df_copy['query'] = docs_df_copy['query'].apply(lambda q: q if is_english(q) else '') # keep only English queries
    docs_df_copy['query'] = list(tokenize(docs_df_copy['query'].tolist()))
    docs_df_copy['query'] = remove_stopwords(docs_df_copy['query'].tolist())

    doc2categ_mapping = defaultdict(list)

    pbar = tqdm(total=docs_df_copy.shape[0])

    i = 0
    while i < docs_df_copy.shape[0]-1:
        group = []
        k = 0

        if i==docs_df_copy.shape[0]-2:
            group.append(docs_df_copy.iloc[i].query)
            k+=1
            i+=1

        else:
            while True:
                k+=1
                tmp_cur = docs_df_copy.iloc[i].time_stamp
                tmp_prev = docs_df_copy.iloc[i+1].time_stamp

                diff = round((tmp_cur - tmp_prev).total_seconds())
                if diff<window:
                    group.append(docs_df_copy.iloc[i].query)
                    i+=1
                    if i==docs_df_copy.shape[0]-2:
                        group.append(docs_df_copy.iloc[i].query)
                        break
                else:
                    group.append(docs_df_copy.iloc[i].query)
                    i+=1
                    break

        if i==docs_df_copy.shape[0]-1:
            pbar.update(k+1)
        else:
            pbar.update(k)                

        doc2categ_mapping_mini = defaultdict(list)
        for doc in group:
            doc_vec = doc2vec(doc, word2vec_model)
            if len(doc_vec)>0:
                cos_sim = -2
                for categ in categories_dict.keys():
                    sim_score = sim_method(doc_vec, categ, word2vec_model, categories_dict)
                    if sim_score>cos_sim:
                        best_categ = categ
                        cos_sim = sim_score

                if cos_sim>thr:
                    doc2categ_mapping_mini[best_categ].append((doc,cos_sim))

        if len(doc2categ_mapping_mini)>0:
            categ_sizes = np.array([len(vals) for _, vals in doc2categ_mapping_mini.items()])
            index = np.argwhere(categ_sizes==max(categ_sizes)).reshape(-1)
            keys = list(doc2categ_mapping_mini.keys())

            if len(index)==1:
                winner_categ = keys[index[0]]
            else:
                l = []
                for ind in index:
                    sm = 0
                    for each in doc2categ_mapping_mini[keys[ind]]:
                        sm+=each[1]
                    l.append(sm)
                winner_categ = keys[index[l.index(max(l))]]

            for j, doc in enumerate(group):
                doc2categ_mapping[winner_categ].append(docs_raw[j+i-len(group)])

    pbar.close()

    return doc2categ_mapping




