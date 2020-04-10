# Vinay Bhupathiraju
# A14432633

import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.

    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# Question # 1 
# ---------------------------------------------------------------------

def median(nums):
    """
    median takes a non-empty list of numbers,
    returning the median element of the list.
    If the list has even length, it should return
    the mean of the two elements in the middle.

    :param nums: a non-empty list of numbers.
    :returns: the median of the list.
    
    :Example:
    >>> median([6, 5, 4, 3, 2]) == 4
    True
    >>> median([50, 20, 15, 40]) == 30
    True
    >>> median([1, 2, 3, 4]) == 2.5
    True
    """
    
    size = len(nums)
    nums.sort()
    
    if size % 2 == 1:
        med = nums[size // 2]
    # sets med to the mean of the 2 middle values if array has even 
    # number of elements
    elif size % 2 == 0: 
        med1 = nums[size // 2]
        med2 = nums[(size // 2) - 1]
        med = (med1 + med2) / 2
    
    return med


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance
    as integers is also i.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    elements as described above.

    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """

    size = len(ints)
    
    for i in range(0, size):
        for j in range(i + 1, size):
            diff_idx = abs(i - j)
            diff_vals = abs(ints[i] - ints[j])
            if diff_idx == diff_vals:
                return True
            
    return False


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def prefixes(s):
    """
    prefixes returns a string of every 
    consecutive prefix of the input string.

    :param s: a string.
    :returns: a string of every consecutive prefix of s.

    :Example:
    >>> prefixes('Data!')
    'DDaDatDataData!'
    >>> prefixes('Marina')
    'MMaMarMariMarinMarina'
    >>> prefixes('aaron')
    'aaaaaraaroaaron'
    """

    size = len(s)
    word = s[0]
    for i in range(1, size):
        to_add = s[0:i+1]
        word = word + to_add        
        
    return word


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def evens_reversed(N):
    """
    evens_reversed returns a string containing 
    all even integers from  1  to  N  (inclusive)
    in reversed order, separated by spaces. 
    Each integer is zero padded.

    :param N: a non-negative integer.
    :returns: a string containing all even integers 
    from 1 to N reversed, formatted as decsribed above.

    :Example:
    >>> evens_reversed(7)
    '6 4 2'
    >>> evens_reversed(10)
    '10 08 06 04 02'
    """
    
    reversed_list = ''
    size = len(str(N))
    
    even_odd = N % 2
    if even_odd == 1:
        N = N - 1
    
    for i in range(N, 0, -2):
        num = str(i)
        diff = size - len(num)
        # adds zero padding
        for i in range(diff):
            num = '0' + num
        reversed_list = reversed_list + ' ' + num
    
    return reversed_list[1:]


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of the line.

    :param fh: a file object to read from.
    :returns: a string of last characters from fh

    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """

    txt = ''
    
    with fh as fg:
        for line in fg:
            last = line[-2]
            txt = txt + last
    
    return txt


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """

    size = len(A)
    # creates array of index values
    B = np.arange(size)
    B = B**0.5
    
    # adds square root of index to initial array elements
    with_sqrt = A + B
    
    return with_sqrt


def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is divisble by 16.

    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.

    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 33]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """

    return A % 16 == 0


def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """

    B = A[0:-1]
    C = A[1:]
    
    # growth formula
    growth = (C - B) / B
    growth = np.round(growth, 2)

    return growth


def arr_4(A):
    """
    Create a function arr_4 that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: an integer of the total number of shares.

    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """

    twenty = 20
    # number of shares to be purchased each day with $20
    shares = twenty // A
    # how much is spent on stocks
    spent = shares * A
    # leftover from the $20 not spent on stocks
    leftover = twenty - spent
    # running total of leftover to see when we can buy stock on our own
    running_leftover = leftover.cumsum()
    # boolean array, True when we can purchase stock with leftover
    when_to_purchase = running_leftover > A
    # indeces where you can purchase stock with leftover
    days = np.where(when_to_purchase == True)
    # if never, then return -1
    if days[0].size == 0:
        return -1
    day_to_purchase = min(min(days))

    return day_to_purchase


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def movie_stats(movies):
    """
    movies_stats returns a series as specified in the notebook.

    :param movies: a dataframe of summaries of
    movies per year as found in `movies_by_year.csv`
    :return: a series with index specified in the notebook.

    :Example:
    >>> movie_fp = os.path.join('data', 'movies_by_year.csv')
    >>> movies = pd.read_csv(movie_fp)
    >>> out = movie_stats(movies)
    >>> isinstance(out, pd.Series)
    True
    >>> 'num_years' in out.index
    True
    >>> isinstance(out.loc['second_lowest'], str)
    True
    """

    # dictionary to add each key:value pair if it can be computed
    out_dict = {}
    
    try:
        min_year = movies['Year'].min()
        max_year = movies['Year'].max()
        num_years = max_year - min_year
        out_dict.update(num_years = num_years)
    except:
        out_dict = out_dict
        
    try: 
        tot_movies = movies['Number of Movies']
        out_dict.update(tot_movies = tot_movies)
    except:
        out_dict = out_dict
    
    try: 
        least_mov = movies[movies['Number of Movies'] == movies['Number of Movies'].min()]
        yr_fewest_movies = least_mov['Year'].min()
        out_dict.update(yr_fewest_movies = yr_fewest_movies)
    except:
        out_dict = out_dict
    
    try:
        avg_gross = movies['Total Gross'].mean()
        out_dict.update(avg_gross = avg_gross)
    except:
        out_dict = out_dict
        
    try:
        highest_gross = movies[movies['Total Gross'] == movies['Total Gross'].max()]
        highest_per_movie = highest_gross['Year'].min()
        out_dict.update(highest_per_movie = highest_per_movie)
    except:
        out_dict = out_dict

    try:
        # reset index so that I can access second lowest using index
        by_gross = movies.sort_values('Total Gross').reset_index()
        second_lowest = by_gross['#1 Movie'][1]
        out_dict.update(second_lowest = second_lowest)
    except:
        out_dict = out_dict
    
    try:
        # find movies with Harry Potter
        harry = movies[movies['#1 Movie'].str.contains('Harry Potter')]
        # get years of those movies
        harry_years = harry['Year']
        year_after_harry = harry_years + 1
        year_after_harry = year_after_harry.tolist()
        # df with only the years after Harry movies
        movies_after_harry = movies[movies['Year'].isin(year_after_harry)]
        avg_after_harry = movies_after_harry['Number of Movies'].mean()
        out_dict.update(avg_after_harry = avg_after_harry)
    except:
        out_dict = out_dict

    return pd.Series(out_dict)
    

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a 
    properly formatted dataframe (as described in 
    the question).

    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data, 
    as specificed in the question statement.

    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """

    rows = []
    with open(fp) as fl:
        cols = fl.readline().strip()
        cols = cols.split(',')
        for line in fl:
            word = line.replace('"', '') # take out " in the geo locations
            word = word.strip()
            entry = word.split(',')

            # there are some null (empty) values in the data so take them out
            if '' in entry:
                entry = ' '.join(entry).split()
                
            # put the geo location together and have it at the end of the list
            geo = entry[-2] + ',' + entry[-1]
            entry = entry[:-2]
            entry.append(geo)

            # add all entries to the row list
            rows.append(entry)

        # close file to make sure there is no memory leak
        fl.close()

        # create the data frame and set data types
        df = pd.DataFrame(rows, columns = cols)
        df['first'] = df['first'].astype(str)
        df['last'] = df['last'].astype(str)
        df['weight'] = df['weight'].astype(np.float64)
        df['height'] = df['height'].astype(np.float64)
        df['geo'] = df['geo'].astype(str)

    return df


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median'],
    'q02': ['same_diff_ints'],
    'q03': ['prefixes'],
    'q04': ['evens_reversed'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['movie_stats'],
    'q08': ['parse_malformed']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
