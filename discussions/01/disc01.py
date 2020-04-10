
import numpy as np
import os

def data2array(filepath):
    """
    data2array takes in the filepath of a 
    data file like `restaurant.csv` in 
    data directory, and returns a 1d array
    of data.

    :Example:
    >>> fp = os.path.join('data', 'restaurant.csv')
    >>> arr = data2array(fp)
    >>> isinstance(arr, np.ndarray)
    True
    >>> arr.dtype == np.dtype('float64')
    True
    >>> arr.shape[0]
    100000
    """
        
    empty = []
    file = np.genfromtxt(filepath, delimiter=',')
    for i in file:
        empty.append(i)
    empty.pop(0)
    full_arr = np.array(empty)

    return full_arr


def ends_in_9(arr):
    """
    ends_in_9 takes in an array of dollar amounts 
    and returns the proprtion of values that end 
    in 9 in the hundredths place.

    :Example:
    >>> arr = np.array([23.04, 45.00, 0.50, 0.09])
    >>> out = ends_in_9(arr)
    >>> 0 <= out <= 1
    True
    """
    
    full_len = len(arr)
    nine = 9
    multiplier = 100
    ten = 10
    
    multiplied = arr * multiplier
    rounded = np.round(multiplied, 0)
    modulus = rounded % ten
    num_nines = np.count_nonzero(modulus == nine)
    
    proportion = num_nines / full_len
    return proportion
 
