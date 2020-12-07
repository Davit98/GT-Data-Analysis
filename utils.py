import datetime
from datetime import date

from collections import Counter, defaultdict


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




def top_words_per_quarter(time_stamped_data,docs,years,n_words=3):
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
    



def top_words_freq_history(time_stamped_data,docs,years,n_words=5):
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


