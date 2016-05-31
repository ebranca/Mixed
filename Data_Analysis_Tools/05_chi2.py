# -*- coding: utf-8 -*-
import sys
# import io

from collections import OrderedDict
from tabulate import tabulate
import decimal
from decimal import Decimal
import itertools
import numbers
import string


import numpy as np
from scipy import stats
from scipy.stats.distributions import chi2 as dist_chi2
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


###############################################################################


def factorial(x):
    #import decimal
    # calculate the factorial of a number
    if not isNumber(x):
        raise ValueError("factorial: Received invalid number.")
    if (x < 0):
        raise ValueError("factorial: Cannot calculate factorial of negative numbers!")
    if (x == 0) or (x == 1):
        return(1)
    else:
        result = 1
        for i in range(2, x + 1):
            i = decimal.Decimal(str(i))
            try:
                result = (result * i)
            except decimal.Overflow as e:
                raise ValueError("Number too big, internal error (OVERFLOW)")
            except Exception as e:
                raise ValueError("%s"% (ascii(e),))
        return(int(result))

def comb_nonrep(n, k):
    #import itertools
    # calculate the number of combinations without repetitions
    # for "n" objects diveded into groups of k" objects
    if (n < 0) or (k < 0):
        raise ValueError("received invalid negative number.")
    nom = factorial(n)
    denom = (factorial(k) * factorial(n - k))
    result = int(nom / denom)
    return(result)


def sign(number):
    #import numbers
    try:
        number = float(number)
    except Exception:
        raise ValueError("ERROR: string %r is not a valid number." % (number,))
    # check if supposed number is number
    if (isinstance(number, numbers.Real)):
        if (number > 0):
            return(1)
        elif (number == 0):
            return(0)
        elif (number < 0):
            return(-1)
        else:
            raise ValueError("ERROR: unexpected error while evaluating  %r" % (number, ))


def xlog(number, base=None, round_at=None, option=None):
    # http://stats.stackexchange.com/questions/1444/how-should-i-transform-non-negative-data-including-zeros
    # http://robjhyndman.com/hyndsight/transformations/
    #import numbers
    #from decimal import Decimal

    options = ["int", "sign", "shift:"]
    if option:
        option = str(option)

    if not number:
        raise ValueError("input is invalid, please provide a number (float or integer).")
    # check if supposed number is number
    if not isinstance(number, numbers.Real):
        raise ValueError("number '%s' is not a valid real number." % (ascii(number),))
    if option:
        if (number < 0) and not ("sign" in option):
            raise ValueError("cannot calculate logarithm of a negative number.")
    else:
        if (number < 0):
            raise ValueError("cannot calculate logarithm of a negative number.")

    if base:
        if (base == 'e'):
            pass
        else:
            if not isNumber(number):
                raise ValueError("invalid base value '%s'" % (ascii(base),))
            if (base == 1) and (number == 1):
                raise ValueError("calculation of log in base '1' of number '1' is not possible.")
            if (base == 0):
                raise ValueError("calculation of log in base 0 is not possible.")
            if (base < 0):
                raise ValueError("calculation of log with a negative base is not possible.")
            if (base == number):
                return(1)

    if option:
        if (option == "sign") and (option in options):
            nsign = sign(number)
            number = abs(Decimal(str(number)))
        elif ("shift:" in option):
            sopt= option.split(":")
            if not (len(sopt) == 2):
                raise ValueError("invalid option '%s'" % (ascii(option),))

            shift = sopt[1]
            if not isNumber(shift):
                raise ValueError("invalid shift value %s in option %s" % (ascii(shift), ascii(option)))
            shift = float(shift)
            number = number + shift
            if not (isinstance(shift, numbers.Real)):
                raise ValueError("shift %s is not a valid real number." % (ascii(shift),))
            if (shift <= 0):
                raise ValueError("shift can only be a positive integer value.")

    if base:
        if (base == "e"):
            result = Decimal(str(number)).ln()
        elif (base == 10):
            result = Decimal(str(number)).log10()
        else:
            base = float(base)
            result = (Decimal(str(number)).ln() / Decimal(str(base)).ln())
    else:
        result = Decimal(str(number)).log10()

    if (option == "sign") and (option in options):
        result = (nsign * result)

    if round_at:
        if not isNumber(round_at):
            raise ValueError("rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
        result = round(result, precision)

    if (option == "int") and (option in options):
        result = round(result, 0)
    if result.is_zero():
        result = 0

    return result


def xarccosh(number):
    # https://en.wikipedia.org/wiki/Inverse_hyperbolic_function
    # http://www.cs.washington.edu/research/projects/uns/F9/src/boost-1.39.0/libs/math/doc/sf_and_dist/html/math_toolkit/special/inv_hyper/acosh.html
    #from decimal import Decimal
    # arcsinh(185) = 5.913495700956333819331535103687269835462231550162396896624 - WOLFRAM ALPHA OK
    # calculated using Decimal -> 5.913495700956333819331535104 - OK
    if not isNumber(number):
        raise ValueError(" number '%s' is not a valid real number." % (ascii(number),))
    if (number < 1):
        raise ValueError(" cannot calculate arccosh of a number smaller than 1.")
    # calculate INVERSE ARC COS
    # arccosh = ln(x + ( sqrt( (x ** 2) - 1) ) )
    number = Decimal(str(number))
    result = Decimal(number + (xsqrt((number**2) - 1))).ln()
    return float(result)

def xarcsinh(number):
    #from decimal import Decimal
    # https://en.wikipedia.org/wiki/Inverse_hyperbolic_function
    # http://worthwhile.typepad.com/worthwhile_canadian_initi/2011/07/a-rant-on-inverse-hyperbolic-sine-transformations.html
    # arcsinh(185) = 5.913510310160134810673581720 - WOLFRAM ALPHA OK
    # calculated using Decimal -> 5.913510310160134810673581720 - OK
    if not isNumber(number):
        raise ValueError(" number '%s' is not a valid real number." % (ascii(number),))
    if (number < 1):
        raise ValueError(" cannot calculate arcsinh of a number smaller than 1.")
    # calculate INVERSE ARC SIN
    # arcsinh = ln(x + ( sqrt( (x ** 2) + 1) ) )
    number = Decimal(str(number))
    result = Decimal(number + (xsqrt((number**2) + 1))).ln()
    return float(result)

def xarctanh(number):
    # https://en.wikipedia.org/wiki/Inverse_hyperbolic_function
    # http://www.cs.washington.edu/research/projects/uns/F9/src/boost-1.39.0/libs/math/doc/sf_and_dist/html/math_toolkit/special/inv_hyper/atanh.html
    #from decimal import Decimal
    # arcsinh(185) = 0.5493061443340548456976226184 - WOLFRAM ALPHA OK
    # calculated using Decimal -> 0.5493061443340548456976226185 - OK
    if not isNumber(number):
        raise ValueError(" number '%s' is not a valid real number." % (ascii(number),))
    if (number >= 1):
        raise ValueError(" cannot calculate arcsinh of a number equal or greater then 1.")
    if (number <= -1):
        raise ValueError(" cannot calculate arcsinh of a number equal or smaller then -1.")
    # calculate INVERSE ARC TAN
    # arctan = (ln((1 + x) / (1 - x)) / 2)
    number = Decimal(str(number))
    result = (Decimal((1 + number) / (1 - number)).ln() / 2)
    return float(result)

def xcbrt(number):
    #from decimal import Decimal
    if not isNumber(number):
        raise ValueError(" number '%s' is not a valid real number." % (ascii(number),))
    # calculate CUBIC ROOT
    number = Decimal(str(number))
    result = (number ** (Decimal("1") / Decimal("3")))
    return float(result)

def xsqrt(number):
    #from decimal import Decimal
    if not isNumber(number):
        raise ValueError(" number '%s' is not a valid real number." % (ascii(number),))
    # calculate CUBIC ROOT
    number = Decimal(str(number))
    result = (number ** (Decimal("1") / Decimal("2")))
    return float(result)

def bonferroni_adj(alpha, tests):
    # Bonferroni adjustment = (significance level / number of tests)
    #from decimal import Decimal
    if not isNumber(alpha):
        raise ValueError(" number '%s' is not a valid real number." % (ascii(alpha),))
    if not isNumber(tests):
        raise ValueError(" number '%s' is not a valid real number." % (ascii(tests),))
    if (alpha < 0):
        raise ValueError("received invalid probability score (p < 0).")
    if (alpha > 1):
        raise ValueError("received invalid probability score (p > 1).")
    if (alpha < 0) or (tests < 0):
        raise ValueError("received invalid negative number.")
    if (alpha == 0) or (tests == 0):
        raise ValueError("received parameter as zero.")
    # calculate value for bonferroni adjustment
    alpha = Decimal(str(alpha))
    tests = Decimal(str(tests))
    badj = (alpha / tests)
    return badj

def isprintable(data_string):
    #import string
    printset = set(string.printable)
    isprintable = set(data_string).issubset(printset)
    return isprintable

def descriptive(nparray, selection):
    if (len(nparray) < 1):
        raise ValueError("ERROR: data array not valid.")

    if selection:
        selection = str(selection).lower()
    else:
        raise ValueError("ERROR: selection variable missing.")

    # allocate ordered dictionary
    diststats = OrderedDict()
    # get data columns
    columns = nparray.columns.values
    # create header list
    headers = list()
    for name in columns:
        headers.append(name)
    del(columns)

    if (selection == "df"):
        for name in headers:
            # get column type
            dtype = str(nparray[name].dtypes)
            if (("float" in dtype) or ("int" in dtype)):
                dtype = "numerical (%s)" % (dtype,)
            elif ("catego" in dtype):
                dtype = "categorical"
            # get total items in the columns
            dtotal = float(nparray[name].shape[0])
            # get total items of numeric type (excludes NaN, nulls, strings, etc..)
            dvalid = float(nparray[name].count())
            # get MISSING or NUL or NaN items (NaN, nul, etc..)
            dnulls = float(nparray[name].isnull().sum())
            # numeric percentage
            dvalidp = float((dvalid/dtotal) * 100)
            # missing percentage
            dnullsp = float((dnulls/dtotal) * 100)
            # prepare formatted list
            dstring = [name, dtype, dtotal, dvalid, dvalidp, dnulls, dnullsp]
            # add list to ordered dictionary
            diststats[name] = dstring

        tab_headers = ["Column", "Type", "Elements", "Valid", "Valid (%)",
                       "Missing", "Missing (%)"]

    elif (selection == "num"):
        for name in headers:
            # get column type
            dtype = str(nparray[name].dtypes)
            if (("float" in dtype) or ("int" in dtype)):
                # get total items in the columns
                dtotal = float(nparray[name].shape[0])
                dmin = float(nparray[name].min())
                dmax = float(nparray[name].max())
                drange = float(dmax - dmin)
                dmean = float(nparray[name].mean())
                dmedian = float(nparray[name].median())
                dmode = float(nparray[name].mode())
                dvar = float(nparray[name].var())
                # prepare formatted list
                dstats = [name, dtotal, dmin, dmax, drange, dmean, dmedian, dmode, dvar]
                # add list to ordered dictionary
                diststats[name] = dstats

        tab_headers = ["Variable", "count", "min", "max", "range",
                        "mean", "median", "mode", "variance"]

    elif (selection == "dist"):
        for name in headers:
            # get column type
            dtype = str(nparray[name].dtypes)
            if (("float" in dtype) or ("int" in dtype)):
                # get total items in the columns
                dtotal = float(nparray[name].shape[0])
                dstd = float(nparray[name].std())
                dmin = float(nparray[name].min())
                dq1 = float(nparray[name].quantile(0.25))
                dq2 = float(nparray[name].quantile(0.5))
                dq3 = float(nparray[name].quantile(0.75))
                dmax = float(nparray[name].max())
                #dmad = float(nparray[name].mad())
                dIQR = (dq3 - dq1)
                dskew = float(nparray[name].skew())
                # prepare formatted list
                ddistr = [name, dtotal, dstd, dmin, dq1, dq2, dq3, dIQR, dmax, dskew]
                # add list to ordered dictionary
                diststats[name] = ddistr

        tab_headers = ["Variable", "count", "std", "min", "Q1 (25%)", "Q2 (50%)",
                       "Q3 (75%)", "IQR", "max", "skew"]

    # prepare tabulation lines
    tablines = list()
    for k,v in diststats.items():
        tablines.append(v)
    # create tabulate
    strtab = str(tabulate(tablines, headers=tab_headers, tablefmt='grid'))
    return(strtab)

def isNumber(number):
    #import numbers
    try:
        number = float(number)
        if isinstance(number, numbers.Real):
            return(True)
        else:
            return(False)
    except:
        return(False)

def manualbin(nparray, minval=None, maxval=None, numbins=None, binning=False):
    # check start value is number
    if not isNumber(minval):
        minval = np.min(nparray)
    if (minval > np.min(nparray)):
        minval = np.min(nparray)

    # check end value is number
    if not isNumber(maxval):
        maxval = np.max(nparray)
    if (maxval < np.max(nparray)):
        maxval = np.max(nparray)

    if (maxval == minval):
        minval = np.min(nparray)
        maxval = np.max(nparray)

    # sort the array in ASCENDING order
    nparray = np.sort(nparray)

    # check minimum array length
    if (len(nparray) < 4):
        return 1, minval, maxval
    else:
        if (isNumber(numbins) and (numbins > 0)):
            # calculate bin size as np float
            binsize = ((maxval - minval) / numbins)
            # generate the bins of size binsize from "minval" to "maxval"
            npbins = np.arange(minval, maxval, binsize)
            #print(minval)
            #print(maxval)
            #print(numbins)
            #print(binsize)
            #print(npbins)
            #print(ascii(binning))
            if binning:
                # for each element in array, get bin number of value in i-th index
                # Output array of indices, of same shape as x.
                binned = np.digitize(nparray, npbins)
                return binsize, minval, maxval, numbins, binned
            else:
                return binsize, minval, maxval, numbins
        else:
            raise ValueError("ERROR: value for number of bins null or invalid.")

def freqency_table(nparray, numbins, minval=None, maxval=None, binning=True):
    # check start value is number
    if not isNumber(minval):
        minval = np.min(nparray)
    if (minval > np.min(nparray)):
        minval = np.min(nparray)

    # check end value is number
    if not isNumber(maxval):
        maxval = np.max(nparray)
    if (maxval < np.max(nparray)):
        maxval = np.max(nparray)

    # check range make sense
    if (maxval == minval):
        minval = np.min(nparray)
        maxval = np.max(nparray)

    # check number of bins is correct
    if not isNumber(numbins):
        numbins = int(numbins)

    # get total number of elements
    #tot_elem = data[].shape[0]
    tot_elem = len(nparray)
    # sort the array in ASCENDING order
    nparray = np.sort(nparray)
    # get the binnig
    binsize, minval, maxval, numbins, binned = manualbin(nparray, minval, maxval, numbins=numbins, binning=True)
    # generate the bins of size binsize from "minval" to "maxval"
    npbins = np.arange(minval, maxval, binsize)

    # get how many elements per interval
    unique, counts = np.unique(binned, return_counts=True)
    bincount = OrderedDict(zip(unique, counts))

    # create list for each interval range
    headbin = list()
    nbcount = 0
    for num in npbins:
        nbcount = nbcount + 1
        imin = npbins[(nbcount - 1)]
        if (nbcount < numbins):
            imax = npbins[(nbcount)]
        elif (nbcount == numbins):
            imax = maxval
        else:
            raise ValueError()
        headbin.append([nbcount, imin, imax])
    del(npbins)

    # add bin count to each list
    for pos, val in bincount.items():
        for elem in headbin:
            if (elem[0] == pos):
                elem.append(val)

    # add zero to any interval with no items
    for pos, val in bincount.items():
        for elem in headbin:
            if (len(elem) == 3):
                elem.append(0)
    del(bincount)

    ftable = list()
    tot_freq = 0
    tot_freqp = 0
    for inter in headbin:
        # set interval headers
        if (inter[0] < numbins):
            interval = "[%s <-> %s)" % (inter[1], inter[2])
        else:
            interval = "[%s <-> %s]" % (inter[1], inter[2])
        # frequency
        freq = inter[3]
        # frequency percentage
        freqp = ((freq / float(tot_elem)) * 100)
        # cumulative frequency
        tot_freq = tot_freq + freq
        # cumulative frequency percentage
        tot_freqp = ((tot_freq / float(tot_elem)) * 100)
        # set printable list
        dstring =[interval, freq, freqp, tot_freq, tot_freqp]
        ftable.append(dstring)

    freq_headers = ["Interval", "Frequency", "Frequency (%)", "Cumulative Freq.", "Cumulative Freq. (%)"]

    # create tabulate
    strtab = (str(tabulate(ftable, headers=freq_headers, tablefmt='orgtbl')) + "\n")
    return(strtab)

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def uniquevals(nparray):
    # create temp dictionary to store unique value count
    tvalcount = dict()
    # count unique values
    for item in nparray:
        # save unique diametrer values
        tvalcount[item] = tvalcount.get(item, 0) + 1
    # sort temp dictionary by key (that is numeric) and store in ordered dict
    valcount = OrderedDict(sorted(tvalcount.items()))
    # delete temp dictionary
    del(tvalcount)
    # return counts
    return valcount

def splitdict(dictionary):
    # splits a dictionary into 2 nparray, one for K, one for V
    listA = list()
    listB = list()
    for key, value in dictionary.items():
        listA.append(float(key))
        listB.append(float(value))
    arrA = np.array(listA)
    arrB = np.array(listB)
    del(listA)
    del(listB)
    return arrA, arrB

def val_shift(nparray, shift):
    listA = list()
    for value in nparray:
        listA.append(value + shift)
    arrA = np.array(listA)
    del(listA)
    return arrA

def axis_major_ticks(nparray):
    # get numeric range
    valrange = (max(nparray) - min(nparray))
    # get 1% of the range
    onep = (valrange / 100.0)
    # set interval to use as spacer before min and after max (default 3% of range)
    spacer = (onep * 3)
    # set tick interval to 10%
    tickint = (onep * 10)
    # get array minimum value
    amin = min(nparray)
    # get array maximum value
    amax = max(nparray)
    # set lower range
    lowrng = (amin - spacer)
    # set higher range
    higrng = (amax + spacer)
    # set minimum ticker value
    mintick = amin
    # set maximum ticker value + 1% (otherwise max value is NOT visible)
    maxtick = (amax + (onep))
    # calculate all values to use as tickers withing defined range
    # to see the end value we set max value as (max + value equal to 1% of range)
    major_ticks = np.arange(mintick, maxtick, tickint)
    # return all elements
    return lowrng, higrng, major_ticks, amin, amax



def chi2_testtype(variable1, variable2, dataframe):
    #check that variables are categorical
    v1dtype = str(dataframe[variable1].dtypes)
    if (v1dtype == 'category'):
        v1dtypes = "PASS"
    else:
        v1dtypes = "FAIL"

    v2dtype = str(dataframe[variable2].dtypes)
    if (v2dtype == 'category'):
        v2dtypes = "PASS"
    else:
        v2dtypes = "FAIL"
    # print verbose results
    print("TEST --> Check that both variable are 'CATEGORICAL'")
    print("Variable: '%s' - Type: '%s' (%s)" % (variable1, v1dtype, v1dtypes))
    print("Variable: '%s' - Type: '%s' (%s)" % (variable2, v2dtype, v2dtypes))

    if (v1dtypes == 'FAIL') or (v2dtypes == 'FAIL'):
        print("Status --> FAIL - Not all variables are categorical, quitting.")
        raise ValueError("Not all variables are categorical.")
    else:
        print("Status --> PASS - All variables are categorical.\n")


def chi2_testlen(variable1, variable2, dataframe):
    #check that data series have at least 2 groups
    v1uq = uniquevals(dataframe[variable1])
    v1len = len(v1uq)
    v1names = list()
    for k, v in v1uq.items():
        v1names.append(str(k))
    if (v1len >= 2):
        v1lens = "PASS"
    else:
        v1lens = "FAIL"

    v2uq = uniquevals(dataframe[variable2])
    v2len = len(v2uq)
    v2names = list()
    for k, v in v2uq.items():
        v2names.append(str(k))
    if (v2len  >= 2):
        v2lens = "PASS"
    else:
        v2lens = "FAIL"
    # print verbose results
    print("TEST --> Check that each variable has at least 2 data groups")
    print("Variable: '%s' - Groups: '%s' - Names: %s (%s)" % (variable1, v1len, v1names, v1lens))
    print("Variable: '%s' - Groups: '%s' - Names: %s (%s)" % (variable2, v2len, v2names, v2lens))

    if (v1lens == 'FAIL') or (v2lens == 'FAIL'):
        print("Status--> FAIL - Not all variables have at least 2 data groups, quitting.")
        raise ValueError("Not all variables have at least 2 data groups.")
    else:
        print("Status --> PASS - All variables have at least 2 data groups.\n")


def chi2_df(variable1, variable2, dataframe):
    v1uq = uniquevals(dataframe[variable1])
    v1len = len(v1uq)
    v2uq = uniquevals(dataframe[variable2])
    v2len = len(v2uq)
    # chi2 degrees of freedom
    # r is number of rows, c is number of columns
    # ((r -1) * (c - 1))
    df = ((v2len - 1) * (v1len - 1))
    return int(df)

def chi2_pdev(observed, expected, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("chi2 % dev - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # calculate percentage deviation
    # (((Observed - Expected) / Expected) * 100)
    observed = Decimal(str(observed))
    expected = Decimal(str(expected))
    residual = (observed - expected)
    val_pdev = ((residual / expected) * 100)
    if round_at:
        val_pdev = round(val_pdev, precision)
    return float(val_pdev)

def chi2_stdres(observed, expected, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("chi2 stdev - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # calculate standardised (Pearson) residual
    # ((Observed - Expected) / sqrt(Expected))
    observed = Decimal(str(observed))
    expected = Decimal(str(expected))
    residual = (observed - expected)
    sqrtexp = xsqrt(expected)
    val_stdres = (residual / Decimal(str(sqrtexp)))
    if round_at:
        val_stdres = round(val_stdres, precision)
    return float(val_stdres)

def chi2_adjusted_residual(observed, expected, rowtot, coltot, total, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("chi2 stdev - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # calculate adjusted residual
    # numerator = (observed - expected)
    # row proportion = ( row marginal total / grand total)
    # column proportion = ( column marginal total / grand total)
    # denominator = sqrt(expected * (1-row proportion) * (1-column proportion))
    # adjusted residual = (numerator / denominator)
    observed = Decimal(str(observed))
    expected = Decimal(str(expected))
    rowtot = Decimal(str(rowtot))
    coltot = Decimal(str(coltot))
    total = Decimal(str(total))
    residual = (observed - expected)
    denominator = xsqrt(expected * (1 - (rowtot / total)) * (1 - (coltot / total)))
    denominator = Decimal(str(denominator))
    val_adjres = (residual / denominator)
    if round_at:
        val_adjres = round(val_adjres, precision)
    return float(val_adjres)

def chi2_contrib(observed, expected, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("chi2 contrib - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # calculate standardised (Pearson) residual
    # ((Observed - Expected) / sqrt(Expected))
    observed = Decimal(str(observed))
    expected = Decimal(str(expected))
    residual = (observed - expected)
    val_contrib = ((residual ** 2) / expected)
    if round_at:
        val_contrib = round(val_contrib, precision)
    return float(val_contrib)


def chi2_pdev_yates(observed, expected, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("chi2 % dev - Yates - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # calculate percentage deviation
    # (((Observed - Expected) / Expected) * 100)
    observed = Decimal(str(observed))
    expected = Decimal(str(expected))
    residual = (abs(observed - expected) - Decimal(str(0.5)))
    val_pdev = ((residual / expected) * 100)
    if round_at:
        val_pdev = round(val_pdev, precision)
    if (observed > expected):
        (val_pdev * 1)
    elif (observed < expected):
        (val_pdev * -1)
    return float(val_pdev)

def chi2_stdres_yates(observed, expected, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("chi2 stdev - Yates - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # calculate standardised (Pearson) residual
    # ((Observed - Expected) / sqrt(Expected))
    observed = Decimal(str(observed))
    expected = Decimal(str(expected))
    residual = (abs(observed - expected) - Decimal(str(0.5)))
    sqrtexp = xsqrt(expected)
    val_stdres = (residual / Decimal(str(sqrtexp)))
    if round_at:
        val_stdres = round(val_stdres, precision)
    if (observed > expected):
        (val_stdres * 1)
    elif (observed < expected):
        (val_stdres * -1)
    return float(val_stdres)

def chi2_contrib_yates(observed, expected, round_at=None):
    # chi2 YATES = ( ( ( abs(observed — expected) — 0.5) / expected) x 100)
    # The resulting value is then given a positive sign if observed > expected
    # and a negative sign if observed < expected.
    if round_at:
        if not isNumber(round_at):
            raise ValueError("chi2 contrib - Yates - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # calculate standardised (Pearson) residual
    # ((Observed - Expected) / sqrt(Expected))
    observed = Decimal(str(observed))
    expected = Decimal(str(expected))
    residual = (abs(observed - expected) - Decimal(str(0.5)))
    val_contrib = ((residual ** 2) / expected)
    if round_at:
        val_contrib = round(val_contrib, precision)
    return float(val_contrib)

def phi_coefficient(chi2, totalelem, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("Phi coeff - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # phi coefficient
    # sqrt(chi2 / total elements)
    # to use ONLY with 2x2 tables
    chi2 = Decimal(str(chi2))
    totalelem = Decimal(str(totalelem))
    phi = xsqrt(chi2 / totalelem)
    if round_at:
        phi = round(phi, precision)
    phi = float(phi)
    # Phi coefficient Interpretation
    # Using table of Rea & Parker (1992)
    # http://files.eric.ed.gov/fulltext/EJ955682.pdf
    # https://c.ymcdn.com/sites/aisnet.org/resource/group/3f1cd2cf-a29b-4822-8581-7b1360e30c71/Spring_2003/kotrlikwilliamsspring2003.pdf
    # .00 and under .10  --> Negligible association
    # .10 and under .20  --> Weak association
    # .20 and under .40  --> Moderate association
    # .40 and under .60  --> Relatively strong association
    # .60 and under .80  --> Strong association
    # .80 and under 1.00 --> Very strong association
    interpretation = ""
    if (phi >= 0) and (phi < 0.1):
        interpretation = ("Negligible association")
    elif (phi >= 0.1) and (phi < 0.2):
        interpretation = ("Weak association")
    elif (phi >= 0.1) and (phi < 0.4):
        interpretation = ("Moderate association")
    elif (phi >= 0.1) and (phi < 0.6):
        interpretation = ("Relatively strong association")
    elif (phi >= 0.1) and (phi < 0.8):
        interpretation = ("Strong association")
    elif (phi >= 0.1) and (phi < 1):
        interpretation = ("Very strong association")
    elif (phi == 1):
        interpretation = ("Perfect match")
    final = "%s (%s)" % (phi, interpretation)
    return final

def cramer_V(chi2, totalelem, minrowcol, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("Cramer's V - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # Cramers's V (V)
    # t = minimum value between (number of rows - 1) and (number of columns - 1)
    # sqrt(chi2 / (total elements * t))
    # to use ONLY with rectangular tables
    # only with tables having different number of rows and columns (3x4, 4x6, etc)
    chi2 = Decimal(str(chi2))
    totalelem = Decimal(str(totalelem))
    minrowcol = Decimal(str(minrowcol))
    t = (minrowcol - Decimal(str(1)))
    cramer = xsqrt(chi2 / (totalelem * t))
    if round_at:
        cramer = round(cramer, precision)
    cramer = float(cramer)
    # Cramer’s Interpretation
    # Using table of Rea & Parker (1992)
    # https://c.ymcdn.com/sites/aisnet.org/resource/group/3f1cd2cf-a29b-4822-8581-7b1360e30c71/Spring_2003/kotrlikwilliamsspring2003.pdf
    # http://files.eric.ed.gov/fulltext/EJ955682.pdf
    # .00 and under .10  --> Negligible association
    # .10 and under .20  --> Weak association
    # .20 and under .40  --> Moderate association
    # .40 and under .60  --> Relatively strong association
    # .60 and under .80  --> Strong association
    # .80 and under 1.00 --> Very strong association
    interpretation = ""
    if (cramer >= 0) and (cramer < 0.1):
        interpretation = ("Negligible association")
    elif (cramer >= 0.1) and (cramer < 0.2):
        interpretation = ("Weak association")
    elif (cramer >= 0.1) and (cramer < 0.4):
        interpretation = ("Moderate association")
    elif (cramer >= 0.1) and (cramer < 0.6):
        interpretation = ("Relatively strong association")
    elif (cramer >= 0.1) and (cramer < 0.8):
        interpretation = ("Strong association")
    elif (cramer >= 0.1) and (cramer < 1):
        interpretation = ("Very strong association")
    elif (cramer == 1):
        interpretation = ("Perfect match")
    final = "%s (%s)" % (cramer, interpretation)
    return final

def contingency_coeff(chi2, totalelem, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("Observed Contingency coeff - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # PEARSON contingency coefficient C
    # sqrt(chi2 / (chi2 + number total elements))
    # to use ONLY with quadratic tables
    # only with tables having same number of rows and columns (3x3, 4x4, etc)
    chi2 = Decimal(str(chi2))
    totalelem = Decimal(str(totalelem))
    cC = xsqrt(chi2 / (chi2 + totalelem))
    if round_at:
        cC = round(cC, precision)
    return float(cC)

def contingency_coeff_corr(chi2, totalelem, minrowcol, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("Observed Contingency Coeff Corrented - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # PEARSON contingency coefficient C - corrected
    # m = minimum value between (number of rows) and (number of columns)
    # numerator = (chi2 * m)
    # denominator = ((chi2 + number total elements) * (m - 1))
    # corrected coefficient = sqrt(numerator / denominator)
    # to use ONLY with quadratic tables
    # only with tables having same number of rows and columns (3x3, 4x4, etc)
    chi2 = Decimal(str(chi2))
    totalelem = Decimal(str(totalelem))
    minrowcol = Decimal(str(minrowcol))
    nom = (chi2 * minrowcol)
    denom = ((chi2 + totalelem) * (minrowcol - Decimal(str(1))))
    cC_corr = xsqrt(nom / denom)
    if round_at:
        cC_corr = round(cC_corr, precision)
    return float(cC_corr)

def standardized_contingency_coeff(obeserved_contingency_coeff, nrows, ncolumns, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("Observed Contingency Coeff Corrented - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    obeserved_contingency_coeff = Decimal(str(obeserved_contingency_coeff))
    nrows = Decimal(str(nrows))
    ncolumns = Decimal(str(ncolumns))
    crows = ((nrows - Decimal(str(1))) / nrows)
    ccols = ((ncolumns - Decimal(str(1))) / ncolumns)
    # calculate contingency coefficient maximum value
    cont_coeff_max = ((crows * ccols) ** (Decimal(str(1))/Decimal(str(4))))
    # calculate standardized value
    cont_coeff_std = (obeserved_contingency_coeff / cont_coeff_max)
    if round_at:
        cont_coeff_std = round(cont_coeff_std, precision)
    cont_coeff_std = float(cont_coeff_std)
    # Standardized Contingency Coefficient Interpretation
    # Analyzing Quantitative Data: From Description to Explanation, By Norman Blaikie, page 100
    # https://books.google.fr/books?id=Tv_-YxqWVQ8C&printsec=frontcover#v=onepage&q&f=false
    # .00 and under .01  --> No association
    # .01 and under .10  --> Negligible association
    # .10 and under .30  --> Weak association
    # .30 and under .60  --> Moderate association
    # .60 and under .75  --> Strong association
    # .75 and under .99 --> Very strong association
    # .99 and 1 --> Perfect association
    interpretation = ""
    if (cont_coeff_std >= 0) and (cont_coeff_std < 0.01):
        interpretation = ("No association")
    elif (cont_coeff_std >= 0.01) and (cont_coeff_std < 0.1):
        interpretation = ("Negligible association")
    elif (cont_coeff_std >= 0.10) and (cont_coeff_std < 0.30):
        interpretation = ("Weak association")
    elif (cont_coeff_std >= 0.30) and (cont_coeff_std < 0.60):
        interpretation = ("Moderate association")
    elif (cont_coeff_std >= 0.60) and (cont_coeff_std < 0.75):
        interpretation = ("Strong association")
    elif (cont_coeff_std >= 0.75) and (cont_coeff_std < 0.99):
        interpretation = ("Very Strong association")
    elif (cont_coeff_std >= 0.99) and (cont_coeff_std <= 1.0):
        interpretation = ("Perfect association")
    final = "%s (%s)" % (cont_coeff_std, interpretation)
    return final

def likelihood_ratio_contrib(observed, expected, round_at=None):
    if round_at:
        if not isNumber(round_at):
            raise ValueError("likelihood ratio contrib - rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 0
    # the method requires if a cell is not valid, negative or zero has to be skipped
    # so we mark any invalid cell as zero therefore will have no impact
    if (observed == 0) or (expected == 0) :
        return 0
    else:
        # calculate standardised (Pearson) residual
        # ((Observed - Expected) / sqrt(Expected))
        observed = Decimal(str(observed))
        expected = Decimal(str(expected))
        ratio = float(observed / expected)
        lratioc = (observed * xlog(ratio, base='e'))
        if round_at:
            lratioc = round(lratioc, precision)
        return float(lratioc)


def chi2_expected_check(nrows, ncolumns, list_all_expected_values):
    # set counter
    exp_neg_vals = 0
    exp_zero = 0
    exp_gt0_lt1_vals = 0
    exp_egt1_lt5_vals = 0
    exp_egt5_lt10_vals = 0
    exp_egt10_vals = 0

    # parse list of expected count and check each value
    for item in list_all_expected_values:
        if (item < 0):
            exp_neg_vals = exp_neg_vals + 1
        elif (item == 0):
            exp_zero = exp_zero + 1
        elif (item > 0) and (item < 1):
            exp_gt0_lt1_vals = exp_gt0_lt1_vals + 1
        elif (item >= 1) and (item < 5):
            exp_egt1_lt5_vals = exp_egt1_lt5_vals + 1
        elif (item >= 5) and (item < 10):
            exp_egt5_lt10_vals = exp_egt5_lt10_vals + 1
        elif (item >= 10):
            exp_egt10_vals = exp_egt10_vals + 1
        else:
            raise ValueError("Unexpected: malformed element in list of expected counts.")

    total_values = len(list_all_expected_values)
    test_logs = list()
    test_fail_1 = False
    test_fail_2 = False
    test_fail_3 = False
    data_isCompliant = True

    # for all table sizes
    test_header = "CHI SQUARED - Testing if all 'expected count' cells satisfy minimum requirements.\n"
    test_logs.append(test_header)

    # Assumtion Check 1: no expected count negative
    test1_text = "Assumption --> No expected count has negative value."
    test_logs.append(test1_text)
    test1_found = int(exp_neg_vals)
    test1_foundp = ((test1_found / total_values) * 100)
    test1_counts = "TEST: %s (%s%%) cells with an expected count less than 0." % (test1_found, test1_foundp)
    test_logs.append(test1_counts)
    if (test1_found == 0):
        test_logs.append("Status --> PASS\n")
    else:
        test_fail_1 = True
        test_logs.append("ERROR: Calculation will produce unrecoverable error 'Domain Error'\n")
        test_logs.append("Status --> FAIL\n")

    # Assumtion Check 2: no expected count equal to zero
    test2_text = "Assumption --> No expected count has a value of zero."
    test_logs.append(test2_text)
    test2_found = int(exp_zero)
    test2_foundp = ((test2_found / total_values) * 100)
    test2_counts = "TEST: %s (%s%%) cells with an expected count equal to 0." % (test2_found, test2_foundp)
    test_logs.append(test2_counts)
    if (test2_found == 0):
        test_logs.append("Status --> PASS\n")
    else:
        test_fail_2 = True
        test_logs.append("ERROR: Calculation will produce unrecoverable error 'Division by zero'\n")
        test_logs.append("Status --> FAIL\n")

    # Assumtion Check 3: no expected count less than 1
    test3_text = "Assumption --> No expected count has a value smaller than 1."
    test_logs.append(test3_text)
    test3_found = int(exp_gt0_lt1_vals)
    test3_foundp = ((test3_found / total_values) * 100)
    test3_counts = "TEST: %s (%s%%) cells with an expected count less than 1." % (test3_found, test3_foundp)
    test_logs.append(test3_counts)
    if (test3_found == 0):
        test_logs.append("Status --> PASS\n")
    else:
        test_fail_3 = True
        test_logs.append("ERROR: No reliable result can be produced with expected count less then 1.")
        test_logs.append("Status --> FAIL\n")

    # calculate 20% cutoff
    cutoff_20 = (total_values * 0.2)

    # for tables equal to 2x2
    if (nrows == 2) and (ncolumns == 2):
        # Assumtion Check 4: no expected count less than 5
        # If any expected counts are less than 5, then some other test should be used
        # Cochran (1952, 1954)
        test4_text = "Assumption (for tables equal to 2x2) --> No expected count has a value smaller than 5."
        test_logs.append(test4_text)
        test4_found = (int(exp_gt0_lt1_vals) + int(exp_egt1_lt5_vals))
        test4_foundp = ((test4_found / total_values) * 100)
        test4_counts = "TEST: %s (%s%%) cells with an expected count less than 5." % (test4_found, test4_foundp)
        test_logs.append(test4_counts)
        if (test4_found == 0):
            test_logs.append("Status --> PASS\n")
        else:
            test_logs.append("WARNING: If any expected counts are less than 5, then some other test should be used.")
            test_logs.append("Status --> FAIL\n")

        # Assumtion Check 5: All expected counts should be 10 or greater.
        test5_text = "Assumption (for tables equal to 2x2) --> No expected count has a value smaller than 10."
        test_logs.append(test5_text)
        test5_found = (int(exp_gt0_lt1_vals) + int(exp_egt1_lt5_vals) + int(exp_egt5_lt10_vals))
        test5_foundp = ((test5_found / total_values) * 100)
        test5_counts = "TEST: %s (%s%%) cells with an expected count less than 10." % (test5_found, test5_foundp)
        test_logs.append(test5_counts)
        if (test5_found < cutoff_20):
            test_logs.append("Status --> PASS\n")
        else:
            test_logs.append("WARNING: All expected counts should be 10 or greater.")
            test_logs.append("Status --> FAIL\n")

    # for tables bigger than 2x2
    else:
        # Assumtion Check 6: No more than 20% of the expected counts are less than 5
        # No more than 20% of the expected counts are less than 5
        # (Yates, Moore & McCabe, 1999, p. 734).
        test6_text = "Assumption (for tables bigger than 2x2) --> No more than 20% of the expected counts are less than 5."
        test_logs.append(test6_text)
        test6_found = (int(exp_gt0_lt1_vals) + int(exp_egt1_lt5_vals))
        test6_foundp = ((test6_found / total_values) * 100)
        test6_counts = "TEST - %s (%s%%) cells with an expected count less than 5." % (test6_found, test6_foundp)
        test_logs.append(test6_counts)
        if (test6_found < cutoff_20):
            test_logs.append("Status --> PASS\n")
        else:
            test_logs.append("WARNING: More than 20% of the expected counts are less than 5, some other test should be used.")
            test_logs.append("Status --> FAIL\n")

    # report critical error if any of the first 3 test conditions are TRUE
    if (test_fail_1 == True) or (test_fail_2 == True) or (test_fail_3 == True):
        data_isCompliant = False
        test_logs.append("DETECTED CRITICAL ERRORS:")
        test_logs.append("Not possible to perform chi squared analysis as data does not meet basic requirements.")
        test_logs.append("A change in data structure or data grouping is required.\n")

    # return test results
    if (data_isCompliant == False):
        return ("FAIL", test_logs)
    else:
        return ("PASS", test_logs)


def chi2_crosstab(variable1, variable2, dataframe, alpha=None, round_at=None, verbose=None):
    # set rounding precision if number is valid
    if round_at:
        if not isNumber(round_at):
            raise ValueError("rounding precision '%s' is not a valid real number." % (ascii(round_at),))
        precision = int(Decimal(str(round_at)))
        if (precision < 0):
            precision = 9

    if alpha:
        if not isNumber(alpha):
            raise ValueError("alpha '%s' is not a valid real number between 0 and 1." % (ascii(alpha),))
        alpha = float(Decimal(str(alpha)))
        if (alpha < 0) or (alpha > 1):
            alpha = 0.05
    else:
        alpha = 0.05

    # check that both data series have same amount of element
    count1 = float(dataframe[variable1].shape[0])
    count2 = float(dataframe[variable2].shape[0])
    if not (count1 == count2):
        raise AssertionError("numeric series do not contain the same amount of observations.")
    # calculate degrees of freedom
    df = chi2_df(variable1, variable2, dataframe)
    # get name of groups for variable 1 (X)
    head_rows = list()
    v1uq = uniquevals(dataframe[variable1])
    for k1, v1 in v1uq.items():
        head_rows.append(str(k1))
    head_rows = sorted(head_rows)
    rows = len(v1uq)
    # get name of groups for variable 2 (Y)
    head_cols = list()
    v2uq = uniquevals(dataframe[variable2])
    for k2, v2 in v2uq.items():
        head_cols.append(str(k2))
    head_cols = sorted(head_cols)
    columns = len(v2uq)

    if (rows < 2) or (columns < 2):
        raise AssertionError("Cannot compute chi squared for table smaller than 2x2")

    # create associations between variable 1 (X) and variable 2 (Y)
    # variable 1 should be EXPECTED VARIABLE
    # variable 2 should be REFERENCE VARIABLE
    pairs = dict()
    for k1, v1 in v1uq.items():
        for k2, v2 in v2uq.items():
            pair = "%s,%s" % (str(k1), str(k2))
            # set each cross-association to value count zero
            pairs[pair] = 0
    # calculate value counts for each association
    for index, row in dataframe.iterrows():
        valvar1 = row[variable1]
        valvar2 = row[variable2]
        pair = "%s,%s" % (str(valvar1), str(valvar2))
        # increment each cross-association
        pairs[pair] = pairs.get(pair, 0) + 1
    # create disctionary to store cross tab relations between variables
    tabrows = OrderedDict()
    # for each variable group of X
    for group1 in head_rows:
        # create a dictionary element with that name and assign a list to it
        tabrows[group1] = list()
        # for each variable group in Y
        for group2 in head_cols:
            # crete variable association string (X,Y)
            hpair = "%s,%s" % (str(group1), str(group2))
            # check that association is in the pre calcolated association list
            if (hpair in pairs):
                # get the value for that association
                valcount = pairs[hpair]
                # append value to the list
                tabrows[group1].append(valcount)
            # if not in pre calcolated association list
            else:
                # raise error to alert of internal
                raise ValueError("Unexpected: pre computed crosstab relation '%s' not present." % (hpair,))
        tabrows[group1].append(v1uq[group1])
    # create list with all column total values
    tabrows["Total"] = list()
    alltot = 0
    for group2 in head_cols:
        tabrows["Total"].append(v2uq[group2])
        alltot = alltot + v2uq[group2]
    tabrows["Total"].append(alltot)
    # copy dictionary so we can work on the copy to add element
    temprows = tabrows.copy()
    # calculate expected counts
    excount = 0
    # lists for internal calculations (max precision, no rounding)
    list_allobserved = list()
    list_allexp = list()
    list_allchi2 = list()
    list_allchi2_yates = list()
    list_alllikelihood = list()
    # number of total elements
    totalelem = tabrows["Total"][-1]
    # create cross tabulation to check each value and check calculated values
    for k, v in tabrows.items():
        if (k == "Total"):
            pass
        else:
            # list for expected count
            head_exp = k + " (expected)"
            list_expected = list()
            sum_expected = Decimal("0")
            # list for residuals
            head_res = k + " (residual)"
            list_residual = list()
            sum_residuals = Decimal("0")
            # list for standardised residuals
            head_stdres = k + " (std. residual)"
            list_stdresidual = list()
            head_stdres_yates = k + " (std. residual, Yates)"
            list_stdresidual_yates = list()
            # list for adjusted residuals
            head_adjresiduals = k + " (adj. residual)"
            list_adjresiduals = list()
            # list for percentage deviation
            head_pdeviation = k + " (% deviation)"
            list_pdeviation = list()
            head_pdeviation_yates = k + " (% deviation, Yates)"
            list_pdeviation_yates = list()
            # list for chi 2 cell contribution
            head_chi2contrib = k + " (chi2 contrib.)"
            list_chi2contrib = list()
            head_chi2contrib_yates = k + " (chi2 contrib., Yates)"
            list_chi2contrib_yates = list()
            # Likelihood ratio
            head_liker = k + " (likelihood ratio)"
            list_likelihood_ratio = list()
            # set step counter to zero
            excount = 0
            # parse each count
            for item in v:
                # set each count to Decimal type
                item = Decimal(item)
                # check if we finished the element counts
                if (excount == columns):
                    # skip last column that has marginal totals
                    pass
                # start parsing and counting
                else:
                    # calculate expected value
                    total_row = Decimal(str(v[-1]))
                    total_col = Decimal(str(tabrows["Total"][excount]))
                    total_tab = Decimal(str(totalelem))
                    list_allobserved.append(float(item))
                    # calculate expected value
                    val_exp = ((total_col * total_row) / total_tab)
                    list_allexp.append(float(val_exp))
                    if round_at:
                        val_expr = round(val_exp, precision)
                    list_expected.append(float(val_expr))
                    sum_expected = sum_expected + val_exp
                    # check for possible division by zero errors
                    if (val_exp == Decimal(0)):
                        list_pdeviation.append("Division by zero")
                        list_pdeviation_yates.append("Division by zero")
                        list_stdresidual.append("Division by zero")
                        list_stdresidual_yates.append("Division by zero")
                        list_adjresiduals.append("Division by zero")
                        list_chi2contrib.append("Division by zero")
                        list_chi2contrib_yates.append("Division by zero")
                        list_likelihood_ratio.append("Division by zero")
                        # safety, this WILL generate an internal error
                        # as we mix string and floats TO BE SURE we spot this
                        list_allchi2.append("Division by zero")
                        list_allchi2_yates.append("Division by zero")
                        list_alllikelihood.append("Division by zero")
                    else:
                        # calculate residual
                        # (Observed - Expected)
                        val_res = (item - Decimal(str(val_exp)))
                        if round_at:
                            val_res = round(val_res, precision)
                        list_residual.append(float(val_res))
                        sum_residuals = sum_residuals + val_res
                        # calculate percentage deviation
                        val_pdev = chi2_pdev(item, val_exp, round_at=precision)
                        list_pdeviation.append(val_pdev)
                        # calculate percentage deviation - YATES correction
                        val_pdevy = chi2_pdev_yates(item, val_exp, round_at=precision)
                        list_pdeviation_yates.append(val_pdevy)
                        # calculate standardised (Pearson) residual
                        val_stdres = chi2_stdres(item, val_exp, round_at=precision)
                        list_stdresidual.append(val_stdres)
                        # calculate standardised (Pearson) residual - YATES correction
                        val_stdresy = chi2_stdres_yates(item, val_exp, round_at=precision)
                        list_stdresidual_yates.append(val_stdresy)
                        # calculate adjusted residual
                        val_adjres = chi2_adjusted_residual(item, val_exp, total_row, total_col, total_tab, round_at=precision)
                        list_adjresiduals.append(val_adjres)
                        # ELEMENTS FOR LATER CALCULATIONS - NEED TO STAY NOT ROUNDED
                        # calculate chi square contribution
                        chi2contrib = chi2_contrib(item, val_exp)
                        list_allchi2.append(chi2contrib)
                        # calculate chi square contribution - YATES correction
                        chi2contriby = chi2_contrib_yates(item, val_exp)
                        list_allchi2_yates.append(chi2contriby)
                        # calculate likelihood ratio (G) contribution
                        likelihoodr = (2 * likelihood_ratio_contrib(item, val_exp))
                        list_alllikelihood.append(likelihoodr)
                        # for each list of data to use in tabular visualization
                        if round_at:
                            # add rounded elements to lists
                            list_chi2contrib.append(round(chi2contrib, precision))
                            list_chi2contrib_yates.append(round(chi2contriby, precision))
                            list_likelihood_ratio.append(round(likelihoodr, precision))
                        else:
                            # add raw value to the list
                            list_chi2contrib.append(chi2contrib)
                            list_chi2contrib_yates.append(chi2contriby)
                            list_likelihood_ratio.append(likelihoodr)
                    # increment step counter
                    excount = excount + 1
                    # clean all intermediate variables at the end of each pass
                    del(total_row)
                    del(total_col)
                    del(total_tab)
                    del(val_exp)
                    del(val_res)
                    del(val_pdev)
                    del(val_pdevy)
                    del(val_stdres)
                    del(val_stdresy)
                    del(val_adjres)
                    del(chi2contrib)
                    del(chi2contriby)
                    del(likelihoodr)
            # check sum of expected counts and add to list
            sum_expected = int(round(sum_expected, 0))
            if not (sum_expected == v1uq[k]):
                raise AssertionError("Unexpected: sum of expected counts for '%s' should be '%s', got '%s'." % (k, v1uq[k], sum_expected))
            else:
                list_expected.append(sum_expected)
                temprows[head_exp] = list_expected
                del(list_expected)
            # check sum of residual values and add to list
            sum_residuals = int(round(sum_residuals, 0))
            if not (sum_residuals == 0):
                raise AssertionError("Unexpected: sum of residuals for '%s' should be zero, got '%s'." % (k, sum_residuals,))
            else:
                sum_residuals = abs(sum_residuals)
                list_residual.append(sum_residuals)
                temprows[head_res] = list_residual
                del(list_residual)
            # add list of percentage deviations to reference dictionary
            temprows[head_pdeviation] = list_pdeviation
            temprows[head_pdeviation_yates] = list_pdeviation_yates
            del(list_pdeviation)
            del(list_pdeviation_yates)
            # add list of standardised residuals to reference dictionary
            temprows[head_stdres] = list_stdresidual
            temprows[head_stdres_yates] = list_stdresidual_yates
            del(list_stdresidual)
            del(list_stdresidual_yates)
            # add list of adjusted residuals to reference dictionary
            temprows[head_adjresiduals] = list_adjresiduals
            del(list_adjresiduals)
            # add list of chi2 cell contributions to reference dictionary
            temprows[head_chi2contrib] = list_chi2contrib
            temprows[head_chi2contrib_yates] = list_chi2contrib_yates
            del(list_chi2contrib)
            del(list_chi2contrib_yates)
            # add list of clikelihood ratios to reference dictionary
            temprows[head_liker] = list_likelihood_ratio
            del(list_likelihood_ratio)
            # reset step counter
            excount = 0
    ###########################################################################
    # prepare tables to print
    # create header line
    freq_headers = [""]
    for item in head_cols:
        freq_headers.append(item)
    freq_headers.append("Total")

    # create value table
    table_fit = list()
    for k, v in temprows.items():
        printrow = list()
        printrow.append(k)
        for value in v:
            printrow.append(value)
        # row name + values + total
        if (len(printrow) == (len(head_cols) + 2)):
                pass
        # row name + values
        elif (len(printrow) == (len(head_cols) + 1)):
                printrow.append("-")
        else:
            raise AssertionError("Elements in list does not match expected number of columns.")
        table_fit.append(printrow)
    # create list to store all values
    table_all = list()
    # create list to store only observation counts
    table_data = list()
    for group in head_rows:
        for line in table_fit:
            line_name = str(line[0])
            if (line_name == group):
                table_data.append(line)
            if (line_name.startswith(group)):
                if ("expected" in line_name):
                    table_data.append(line)
                table_all.append(line)

    for line in table_fit:
        line_name = str(line[0])
        if (line_name.startswith("Total")):
            table_data.append(line)
            table_all.append(line)
    # create tabulate
    print_table_data = str(tabulate(table_data, headers=freq_headers
                                    , tablefmt='pipe'
                                    , stralign='left'
                                    , numalign='left'))
    # create tabulate
    print_table_all = str(tabulate(table_all, headers=freq_headers
                                    , tablefmt='pipe'
                                    , stralign='left'
                                    , numalign='left'))
    ###########################################################################
    # Check for data issues, if found print data table and log, then exit
    status, statuslog = chi2_expected_check(rows, columns, list_allexp)

    if (status == "FAIL"):
        print("Chi Squared - Contingency Table\n")
        print(print_table_data)
        print()
        for line in statuslog:
            print(line)
        print("Quitting!\n")
        print()
        sys.exit(1)
    else:
        if (verbose == True):
            for line in statuslog:
                print(line)
    ###########################################################################
    # perform all calculation
    # total number of cells
    tot_cells = (rows * columns)
    # minimum expected value
    if round_at:
        expected_min = round(min(list_allexp), precision)
    else:
        expected_min = min(list_allexp)
    # minimum observed value
    observed_min = min(list_allobserved)
    # calculate chi value - sum of all contributors - using RAW value list
    chi2 = sum(list_allchi2)
    # calculate chi value std dev
    chi2_stdev = np.array(list_allchi2).std()
    # calculate chi value - yates correction - using RAW value list
    chi2_yates = sum(list_allchi2_yates)
    # calculate chi value std dev - yates correction
    chi2_stdev_yates = np.array(list_allchi2_yates).std()
    # probability value
    p_val = stats.chi2.sf(chi2, df)
    # probability value - YATES
    p_val_yates = stats.chi2.sf(chi2_yates, df)
    # phi coefficient
    phi_coeff = phi_coefficient(chi2, totalelem)

    # contingency coefficient C
    obs_cont_coeff = contingency_coeff(chi2, totalelem)
    if (rows < columns):
        # corrected contingency coefficient
        obs_cont_coeff_corr = contingency_coeff_corr(chi2, totalelem, rows)
    else:
        # corrected contingency coefficient
        obs_cont_coeff_corr = contingency_coeff_corr(chi2, totalelem, columns)
    if (rows == columns):
        cont_coeff_std = standardized_contingency_coeff(obs_cont_coeff, rows, columns)
        cont_coeff_std_corr = standardized_contingency_coeff(obs_cont_coeff_corr, rows, columns)

    # cramer's V
    if (rows < columns):
        cramerV = cramer_V(chi2, totalelem, rows)
    else:
        cramerV = cramer_V(chi2, totalelem, columns)

    ###########################################################################
    # Interpret results
    # chi squared - interpretation
    # http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
    # upper tail, one sided - we use alpha
    chi2_ut_1s = alpha
    # calculate chi 2 critical value
    chi2_CV_ut_1s = stats.chi2.isf(chi2_ut_1s, df)
    # check if we can accept or reject null hypothesis  - chi squared
    if (chi2 > chi2_CV_ut_1s):
        chi2_iterp_ut_1s = "Rejected"
    else:
        chi2_iterp_ut_1s = "Accepted"
    # check if we can accept or reject null hypothesis  - chi squared - YATES
    if (chi2_yates > chi2_CV_ut_1s):
        chi2_iterp_ut_1s_yates = "Rejected"
    else:
        chi2_iterp_ut_1s_yates = "Accepted"
    # lower tail, one sided - we use abs(alpha-1)
    chi2_lt_1s = abs(alpha-1)
    # calculater chi 2 critical value
    chi2_CV_lt_1s = stats.chi2.isf(chi2_lt_1s, df)
    # check if we can accept or reject null hypothesis  - chi squared
    if (chi2 < chi2_CV_lt_1s):
        chi2_iterp_lt_1s = "Rejected"
    else:
        chi2_iterp_lt_1s = "Accepted"
    # check if we can accept or reject null hypothesis  - chi squared - YATES
    if (chi2_yates < chi2_CV_lt_1s):
        chi2_iterp_lt_1s_yates = "Rejected"
    else:
        chi2_iterp_lt_1s_yates = "Accepted"
    # two sided - we use (alpha/2) for upper tail
    chi2_ut_2s = (alpha/2)
    # two sided - we use (abs(1-(alpha/2))) for lower tail
    chi2_lt_2s = (abs(1-(alpha/2)))
    # calculater chi 2 critical value
    chi2_CV_ut_2s = stats.chi2.isf(chi2_ut_2s, df)
    chi2_CV_lt_2s = stats.chi2.isf(chi2_lt_2s, df)
    # check if we can accept or reject null hypothesis  - chi squared
    if (chi2 < chi2_CV_lt_2s) or (chi2 > chi2_CV_ut_2s):
        chi2_iterp_2s = "Rejected"
    else:
        chi2_iterp_2s = "Accepted"
    # check if we can accept or reject null hypothesis  - chi squared - YATES
    if (chi2_yates < chi2_CV_lt_2s) or (chi2 > chi2_CV_ut_2s):
        chi2_iterp_2s_yates = "Rejected"
    else:
        chi2_iterp_2s_yates = "Accepted"

    # Likelihood ratio (G-test) - using RAW value list
    likelihood_ratio = (sum(list_alllikelihood))
    # lower tail, one sided
    if (likelihood_ratio < chi2_CV_lt_1s):
        likelihood_ratio_iterp_lt_1s = "Rejected"
    else:
        likelihood_ratio_iterp_lt_1s = "Accepted"
    # upper tail, one sided
    if (likelihood_ratio > chi2_CV_ut_1s):
        likelihood_ratio_iterp_ut_1s = "Rejected"
    else:
        likelihood_ratio_iterp_ut_1s = "Accepted"
    # two sided - we use (alpha/2) for upper tail
    if (likelihood_ratio < chi2_CV_lt_2s) or (likelihood_ratio > chi2_CV_ut_2s):
        likelihood_ratio_iterp_2s = "Rejected"
    else:
        likelihood_ratio_iterp_2s = "Accepted"


    ###########################################################################
    # add all results to list for later printing
    all_details = list()
    all_details.append("\n===========================")
    all_details.append("Contingency Table")
    all_details.append("Table Size: '%sx%s'" % (rows, columns))
    all_details.append("Number of cells: '%s'" % (tot_cells,))
    all_details.append("Total number of elements: '%s'" % (totalelem,))
    all_details.append("Observed minimum value: '%s'" % (observed_min,))
    all_details.append("Expected minimum value: '%s'" % (expected_min,))

    all_details.append("\nChi Squared")
    all_details.append("Pearson chi2: '%s'" % (chi2,))
    all_details.append("Pearson chi2 (std. dev): '%s'" % (chi2_stdev,))
    all_details.append("Degrees of freedom (df): '%s'" % (df,))
    all_details.append("p-value (Pearson chi2): '%s'" % (p_val,))
    all_details.append("Critical value, Lower tail, one-sided (alpha=%s, df=%s): %s" % (alpha, df, chi2_CV_lt_1s))
    all_details.append("Critical value, Upper tail, one-sided (alpha=%s, df=%s): %s" % (alpha, df, chi2_CV_ut_1s))
    all_details.append("Critical value, two-sided (alpha=%s, df=%s, value: %s,%s" % (chi2_ut_2s, df, chi2_CV_lt_2s, chi2_CV_ut_2s))
    all_details.append("Pearson chi2, Null hypothesis, Lower tail, one-sided: '%s'" % (chi2_iterp_lt_1s,))
    all_details.append("Pearson chi2, Null hypothesis, Upper tail, one-sided: '%s'" % (chi2_iterp_ut_1s,))
    all_details.append("Pearson chi2, Null hypothesis, Two-sided: '%s'" % (chi2_iterp_2s,))

    all_details.append("\nChi Squared - Yates Continuity Corrections")
    all_details.append("Yates chi2: '%s'" % (chi2_yates,))
    all_details.append("Yates chi2 (std. dev): '%s'" % (chi2_stdev_yates,))
    all_details.append("Degrees of freedom (df): '%s'" % (df,))
    all_details.append("p-value (Yates chi2): '%s'" % (p_val_yates,))
    all_details.append("Critical value, Lower tail, one-sided (alpha=%s, df=%s): %s" % (alpha, df, chi2_CV_lt_1s))
    all_details.append("Critical value, Upper tail, one-sided (alpha=%s, df=%s): %s" % (alpha, df, chi2_CV_ut_1s))
    all_details.append("Critical value, two-sided (alpha=%s, df=%s, value: %s,%s" % (chi2_ut_2s, df, chi2_CV_lt_2s, chi2_CV_ut_2s))
    all_details.append("Yates chi2, Null hypothesis, Lower tail, one-sided: '%s'" % (chi2_iterp_lt_1s_yates,))
    all_details.append("Yates chi2, Null hypothesis, Upper tail, one-sided: '%s'" % (chi2_iterp_ut_1s_yates,))
    all_details.append("Yates chi2, Null hypothesis, Two-tailed: '%s'" % (chi2_iterp_2s_yates,))

    all_details.append("\nChi Squared - Log-Likelihood ratio")
    all_details.append("Log-Likelihood ratio (G-test): '%s'" % (likelihood_ratio,))
    all_details.append("Critical value, Lower tail, one-sided (alpha=%s, df=%s): %s" % (alpha, df, chi2_CV_lt_1s))
    all_details.append("Critical value, Upper tail, one-sided (alpha=%s, df=%s): %s" % (alpha, df, chi2_CV_ut_1s))
    all_details.append("Critical value, two-sided (alpha=%s, df=%s, value: %s,%s" % (chi2_ut_2s, df, chi2_CV_lt_2s, chi2_CV_ut_2s))
    all_details.append("G-test, Null hypothesis, Lower tail, one-sided: '%s'" % (likelihood_ratio_iterp_lt_1s,))
    all_details.append("G-test, Null hypothesis, Upper tail, one-sided: '%s'" % (likelihood_ratio_iterp_ut_1s,))
    all_details.append("G-test, Null hypothesis, Two-tailed: '%s'" % (likelihood_ratio_iterp_2s,))

    if (rows == columns):
        all_details.append("\nContingency coefficient")
        all_details.append("Observed contingency coefficient (C): '%s'" % (obs_cont_coeff,))
        all_details.append("Observed contingency coefficient corrected (C corr): '%s'" % (obs_cont_coeff_corr,))
        all_details.append("Standardized contingency coefficient (C std): '%s'" % (cont_coeff_std,))
        all_details.append("Standardized contingency coefficient corrected (C corr std): '%s'" % (cont_coeff_std_corr,))

    all_details.append("\nMeasures of Associations")
    if (rows == 2) and (columns == 2):
        all_details.append("Phi coefficient: '%s'" % (phi_coeff,))
    all_details.append("Cramer's V (V): '%s'" % (cramerV,))
    all_details.append("===========================\n")

    if (verbose == True):
        # print full table with all calculations
        print(print_table_all)
        # print all calculation with details
        for line in all_details:
            print(line)

    return (chi2, p_val, df, chi2_iterp_2s)

###############################################################################

dataset = 'marscrater_clean.csv'

data = pd.read_csv(dataset)

# setting variables to CATEGORICAL
data["CRATER_ID"] = data["CRATER_ID"].astype('category')
data["CRATER_NAME"] = data["CRATER_NAME"].astype('category')
data["MORPHOLOGY_EJECTA_1"] = data["MORPHOLOGY_EJECTA_1"].astype('category')
data["MORPHOLOGY_EJECTA_2"] = data["MORPHOLOGY_EJECTA_2"].astype('category')
data["MORPHOLOGY_EJECTA_3"] = data["MORPHOLOGY_EJECTA_3"].astype('category')

# setting variables to NUMERIC - FLOAT
data["LATITUDE_CIRCLE_IMAGE"] = data["LATITUDE_CIRCLE_IMAGE"].astype('float64')
data["LONGITUDE_CIRCLE_IMAGE"] = data["LONGITUDE_CIRCLE_IMAGE"].astype('float64')
data["DIAM_CIRCLE_IMAGE"] = data["DIAM_CIRCLE_IMAGE"].astype('float64')
data["DEPTH_RIMFLOOR_TOPOG"] = data["DEPTH_RIMFLOOR_TOPOG"].astype('float64')

# setting variables to NUMERIC - INT
data["NUMBER_LAYERS"] = data["NUMBER_LAYERS"].astype('int')

###############################################################################

# add new array to dataframe
data = pd.concat([data, pd.DataFrame(columns=['LONGITUDE_EAST_360'])], ignore_index=True)
data = pd.concat([data, pd.DataFrame(columns=['QUADRANGLE'])], ignore_index=True)
data = pd.concat([data, pd.DataFrame(columns=['HEMISPHERE'])], ignore_index=True)
data = pd.concat([data, pd.DataFrame(columns=['LAYERS'])], ignore_index=True)

# calculate new values
for index, row in data.iterrows():
    clon = row["LONGITUDE_CIRCLE_IMAGE"]
    # if value not positive
    if (clon < 0):
        # calculate new value
        elon = (360 - abs(clon))
        # set new value
        data.set_value(index, 'LONGITUDE_EAST_360', elon)
    else:
        # set value
        data.set_value(index, 'LONGITUDE_EAST_360', clon)

    # create categorical QUADRANGLE
    cquad = row["CRATER_ID"]
    cquad = cquad.split("_")[0].replace("'","").strip()
    cquad = "mc%s" % (cquad,)
    # set value
    data.set_value(index, 'QUADRANGLE', cquad)

    # create categorical HEMISPHERE
    cemis = row["LATITUDE_CIRCLE_IMAGE"]

    if (cemis >= -90 and cemis < 0):
        cemis = "South"
    elif (cemis >= 0 and cemis <= 90):
        cemis = "North"
    else:
        raise AssertionError("unexpected Latitude value")
    # set value
    data.set_value(index, 'HEMISPHERE', cemis)

    # create categorical LAYERS
    clay = row["NUMBER_LAYERS"]
    if (clay == 0):
        clay = "no_layers"
    elif (clay > 0):
        clay = "has_layers"

    #if (clay == 0):
    #    clay = "none"
    #elif (clay == 1):
    #    clay = "single"
    #elif (clay == 2):
    #    clay = "double"
    #elif (clay > 2):
    #    clay = "multi"
    #else:
    #    raise AssertionError("unexpected Layer value")

    # set value
    data.set_value(index, 'LAYERS', clay)


# set new column data type
data["LONGITUDE_EAST_360"] = data["LONGITUDE_EAST_360"].astype('float64')
data["QUADRANGLE"] = data["QUADRANGLE"].astype('category')
data["HEMISPHERE"] = data["HEMISPHERE"].astype('category')
data["LAYERS"] = data["LAYERS"].astype('category')

###############################################################################

'''
# WEEK 1

import statsmodels.formula.api as smf

# ANOVA- "DIAM_CIRCLE_IMAGE" ~ "HEMISPHERE"
anova = data[["DIAM_CIRCLE_IMAGE", "HEMISPHERE"]]

model1 = smf.ols(formula='DIAM_CIRCLE_IMAGE ~ C(HEMISPHERE)', data=anova)
results1 = model1.fit()
print("Performing ANOVA analysys between 'DIAM_CIRCLE_IMAGE' and 'HEMISPHERE'")
print()
print(results1.summary())
print()

print("Means for 'DIAM_CIRCLE_IMAGE' by Hemisphere")
m1 = anova.groupby("HEMISPHERE").mean()
print(m1)
print()

print("Standard Deviation for 'DIAM_CIRCLE_IMAGE' by Hemisphere")
std1 = anova.groupby("HEMISPHERE").std()
print(std1)
print()

import statsmodels.stats.multicomp as multi
mc1 = multi.MultiComparison(anova["DIAM_CIRCLE_IMAGE"], anova["HEMISPHERE"])
results2 = mc1.tukeyhsd()
print("Performing Tukey HSDT, or Honestly Significant Difference Test.")
print(results2.summary())
print()
'''
###############################################################################

# WEEK 2

# chi2- "HEMISPHERE" ~ "LAYERS"
chi2_sub = data[["HEMISPHERE", "LAYERS"]]

# get information on variables
print("\nCHI SQUARED - Data Type and count\n")
print(descriptive(chi2_sub, "df"))

var1 = "HEMISPHERE"
var2 = "LAYERS"
subchi2 = chi2_sub

print("\nCHI SQUARED - Data Validation Tests\n")
chi2_testtype(var1, var2, subchi2)
chi2_testlen(var1, var2, subchi2)

# crosstab(var1, var2, subchi2)
chi_value, p_value, degrees_freedom, hypothesis = chi2_crosstab(var1, var2,
                                                    subchi2,
                                                    alpha=0.05,
                                                    round_at=9,
                                                    verbose=True)

print("chi squared = %s" % (chi_value,))
print("df = %s" % (degrees_freedom,))
print("p_value = %s" % (p_value,))
print("Null hypothesis = %s" % (hypothesis,))
print()


from scipy.stats import chi2_contingency
ct1 = pd.crosstab(chi2_sub["HEMISPHERE"], chi2_sub["LAYERS"])
#cs1 = chi2_contingency(ct1)
chi2, p, dof, ex = chi2_contingency(ct1, correction=False)
print("Results using python scipy chi2 funtion (correction=False)")
print("chi squared = %s" % (chi2,))
print("df = %s" % (dof,))
print("p_value = %s" % (p,))
print()
chi2, p, dof, ex = chi2_contingency(ct1, correction=True)
print("Results using python scipy chi2 funtion (correction=True)")
print("chi squared = %s" % (chi2,))
print("df = %s" % (dof,))
print("p_value = %s" % (p,))
print()

