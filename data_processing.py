import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['use','used',
                   'take','took','taken',
                   'begin','began','begun',
                   'build','built',
                   'do','did','done',
                   'find','found',
                   'get','got','gotten',
                   'give','gave','given',
                   'make','made',
                   'vs'])

import gensim
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser

import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])


def tokenize(docs, deacc=False):
    """ 
    Generator function for tokenizing a string.
    
    Parameters
    ----------
    docs : list
        List of strings to be tokenized.
    
    deacc : bool, optional
        If true, removes accent marks from tokens.
        
    Returns
    -------
    An iterator of tokenized strings (one value at a time)
    """
    for d in docs:
        yield(gensim.utils.simple_preprocess(d, deacc=deacc))
        



def remove_stopwords(docs_tok):
    """ 
    Removing stop words from a tokenized string.
    
    Parameters
    ----------
    docs_tok : list
        List of tokenized strings. 
        
    Returns
    -------
    List of tokenized strings without stop words. 
    """
    return [[word for word in doc if word not in stop_words] for doc in docs_tok]




def lemmatization(docs_tok, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']):
    """ 
    Lemmatizing tokens.
    
    Parameters
    ----------
    docs_tok : list
        List of tokenized strings.
    
    allowed_postags: list, optional
        Part-of-speech tags to be kept in tokenized strings. Default keeps nouns, adjectives, verbs, adverbs,
        and proper nouns.
        
    Returns
    -------
    List of tokenized strings with lemmatized tokens.  
    """
    result = []
    for d in tqdm(docs_tok):
        doc = nlp(" ".join(d)) 
        result.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return result




def is_english(s):
    """ 
    Checks if the input string contains non-English alphabet characters. 
    
    Parameters
    ----------
    s : str
        Input string.
        
    Returns
    -------
    True if the input string contains only English alphabet characters, otherwise false.
    """    
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True




def add_bigrams(docs, threshold=10.0, min_count=5):
    """ 
    Form bigrams and add to the input.
    
    Parameters
    ----------
    docs : list
        List of tokenized strings.
    
    threshold : float, optional
        Represent a score threshold for forming the phrases (higher means fewer phrases).
    
    min_count : float, optional
        Ignore all words and bigrams with total collected count lower than this value.
        
    Returns
    -------
    List of tokenized strings with bigrams.
    """
    bigram = Phrases(docs, threshold=threshold, min_count=min_count)
    bigram_mod = Phraser(bigram)
    
    return [bigram_mod[doc] for doc in docs]




