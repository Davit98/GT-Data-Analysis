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




# def top_words_per_quarter(time_data,docs,years,n_words=3):
#     """ 
#     Computes top-n words with corresponding frequences per each quarter for the given years. 
    
#     Parameters
#     ----------
#     years : list
#         List of years to consider for computing top-n words.

#     n_words : int, optional
#         Number of top words to consider.

#     Returns
#     -------
#     ... 
    

