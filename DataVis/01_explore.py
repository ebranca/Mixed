# -*- coding: utf-8 -*-
import sys

from collections import OrderedDict
from tabulate import tabulate

import numpy as np
import pandas as pd
###############################################################################

def typecheck(dataframe, select=None):
    import numbers
    import string
    # get column names
    columns = dataframe.columns.values
    # set result list
    # check each column
    for col in columns:
        # if we want only specific columns, and column is not in list, skip
        if select and not (col in select):
            continue
        # get series corresponding to column name
        series = dataframe[col]
        # set counters
        nelem = 0
        nstring = 0
        nfloat = 0
        ninteger = 0
        nmissing = 0
        ninvalid = 0
        # set column name as ascii text
        scol = ascii(col)
        # check each element in column
        for elem in series:
            # increment element counter
            nelem = nelem + 1
            # convert string to lower + remove leade/trail spaces + remove line end
            elem = ascii(elem).strip().strip("\r\n").strip("\r").strip("\n")
            if (len(elem) > 0):
                # check if element as string is correct or not
                if (elem.isprintable()):
                    xelem = elem
                    for c in ["0","1","2","3","4","5","6","7","8","9","L","e","+","-","."]:
                        xelem=xelem.replace(c,"")
                    # if element without all valid chars has size
                    if (len(xelem) > 0):
                        # mark as string
                        nstring = nstring + 1
                    # if element without all valid chars has NO size
                    else:
                        tfloat = False
                        tint = False
                        # test for float conversion
                        try:
                            felem = float(elem)
                            del(felem)
                            tfloat = True
                        except:
                            tfloat = False
                        # test for int conversion
                        try:
                            ielem = int(elem)
                            del(ielem)
                            tint = True
                        except:
                            tint = False
                        # if both test fails, mark string
                        if (tfloat == False) and (tint == False):
                            nstring = nstring + 1
                        # now we should have a number
                        else:
                            # check is string is a real number
                            if (isinstance(float(elem), numbers.Real)):
                                # check if integer using NEW "float(num).is_integer()"
                                # https://docs.python.org/2/library/stdtypes.html#float.is_integer
                                # https://docs.python.org/3/library/stdtypes.html#float.is_integer
                                # convert to float
                                elem = float(elem)
                                # test if float is an integer
                                if (elem.is_integer()):
                                    ninteger = ninteger + 1
                                # otherwise it shoild only be a float
                                else:
                                    nfloat = nfloat + 1
                            else:
                                # if test fails something is wrong
                                print(ascii(elem))
                                raise ValueError("unexpected float format")
                else:
                    # if strange stuff is present count as invalid
                    ninvalid = ninvalid + 1
            else:
                # if element is empty set missing
                nmissing = nmissing + 1
        # calculate percentage
        nstringp = ((nstring / nelem) * 100)
        nfloatp = ((nfloat / nelem) * 100)
        nintegerp = ((ninteger / nelem) * 100)
        nmissingp = ((nmissing / nelem) * 100)
        ninvalidp = ((ninvalid / nelem) * 100)
        # prepare strings
        res_string = "Strings: %s (%s%%)\n" % (nstring, nstringp)
        res_sint = "Integers: %s (%s%%)\n" % (ninteger, nintegerp)
        res_sfloat = "Floats: %s (%s%%)\n" % (nfloat, nfloatp)
        res_sinv = "Invalid: %s (%s%%)\n" % (ninvalid, ninvalidp)
        res_smiss = "Missing: %s (%s%%)\n" % (nmissing, nmissingp)
        # check if we are missing anything
        check = (nstring + ninteger + nfloat + ninvalid + nmissing)
        if not (check == nelem):
            print("Strings: %s" % (nstring,))
            print("Integers: %s" % (ninteger,))
            print("Floats: %s" % (nfloat,))
            print("Invalid: %s" % (ninvalid,))
            print("Missing: %s" % (nmissing,))
            print("Not-Accoun: %s" % ((nelem - check),))
            raise AssertionError("ERROR: Data not accounted")
        # check if we are seeing nothing
        if ((nstring + ninteger + nfloat + ninvalid + nmissing ) == 0):
            print("Column Name: '%s' - No data detected" % (col,))
            continue
        # bases on stats prepare strings for suggestions
        # if we have strings
        if (nstring > 0):
            # is strings are the only data type provide suggestion
            if (nstring > 0) and ((ninteger + nfloat) == 0):
                suggest = "Suggested variable type: Categorical\n"
                suggest = suggest + "Load dataset -> data = pandas.read_csv('filename_dataset')\n"
                suggest = suggest + "Instruction (1) -> data['%s'] = data['%s'].astype('category')\n" % (col, col)
            # is strings are mixed with other stuff, no guess can be made
            else:
                suggest = "Suggested variable type: Unknown (both strings and numbers are present)\n"
        # if no strings
        else:
            # if we have ints but no floats,suggest to set as int
            if (ninteger > 0) and (nfloat == 0):
                suggest = "Suggested variable type: Numerical (integer)\n"
                suggest = suggest + "Load dataset -> data = pandas.read_csv('filename_dataset')\n"
                suggest = suggest + "Instruction (1) -> pandas.to_numeric(data['%s'], errors='coerce')\n" % (col,)
                suggest = suggest + "Instruction (2) -> data['%s'] = data['%s'].astype('int')\n" % (col, col)
            # with both ints and floats we set as float to avoind data loss
            else:
                suggest = "Suggested variable type: Numerical (float)\n"
                suggest = suggest + "Load dataset -> data = pandas.read_csv('filename_dataset')\n"
                suggest = suggest + "Instruction (1) -> pandas.to_numeric(data['%s'], errors='coerce')\n" % (col,)
        # if data type has been somhow guessed provide stats and suggestion
        result = "Column Name: %s\n%s%s%s%s%s%s" % (scol,
                                                    res_string,
                                                    res_sint,
                                                    res_sfloat,
                                                    res_sinv,
                                                    res_smiss,
                                                    suggest)
        # print to stdout the string with the results
        print(result)

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
            # get MISSING or NOT NUMERIC items (NaN, nulls, strings, etc..)
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

def isNumber(pyobject):
    import numbers
    return isinstance(pyobject, numbers.Real)

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

def freqency_table(nparr, numbins, minval=None, maxval=None, binning=True):
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
    tot_elem = data['LONGITUDE_CIRCLE_IMAGE'].shape[0]
    # sort the array in ASCENDING order
    nparr = np.sort(nparr)
    # get the binnig
    binsize, minval, maxval, numbins, binned = manualbin(nparr, minval, maxval, numbins=numbins, binning=True)
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
            interval = "(%s <-> %s]" % (inter[1], inter[2])
        else:
            interval = "(%s <-> %s)" % (inter[1], inter[2])
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
pd.to_numeric(data['LATITUDE_CIRCLE_IMAGE'], errors='coerce')
pd.to_numeric(data['LONGITUDE_CIRCLE_IMAGE'], errors='coerce')
pd.to_numeric(data['DIAM_CIRCLE_IMAGE'], errors='coerce')
pd.to_numeric(data['DEPTH_RIMFLOOR_TOPOG'], errors='coerce')

# setting variables to NUMERIC - INT
pd.to_numeric(data['NUMBER_LAYERS'], errors='coerce')
data["NUMBER_LAYERS"] = data["NUMBER_LAYERS"].astype('int')

###############################################################################

variables = ['LATITUDE_CIRCLE_IMAGE', 'LONGITUDE_CIRCLE_IMAGE', 'DIAM_CIRCLE_IMAGE']
typecheck(data, select=variables)

print("Data Statistics - Dataframe")
print(descriptive(data, 'df'))
print()

print("Data Statistics - Numeric Interval")
print(descriptive(data, 'num'))
print()

print("Data Statistics - Numeric Distribution")
print(descriptive(data, 'dist'))
print()

print("Frequency Analysis - variable 'LATITUDE_CIRCLE_IMAGE'")
nparray = data['LATITUDE_CIRCLE_IMAGE']
latitude = freqency_table(nparray, 10, -90, 90)
print(latitude)

print("Frequency Analysis - variable 'LONGITUDE_CIRCLE_IMAGE'")
nparray = data['LONGITUDE_CIRCLE_IMAGE']
longitude = freqency_table(nparray, 10, -180, 180)
print(longitude)

print("Frequency Analysis - variable 'DIAM_CIRCLE_IMAGE'")
nparray = data['DIAM_CIRCLE_IMAGE']
diameter = freqency_table(nparray, 10)
print(diameter)

sys.exit(0)
