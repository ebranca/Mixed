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
        if select and not (col in select):
            continue
        series = dataframe[col]
        nelem = 0
        nstring = 0
        nfloat = 0
        ninteger = 0
        nmissing = 0
        ninvalid = 0
        scol = ascii(col)
        # check each element in column
        for elem in series:
            # increment element counter
            nelem = nelem + 1
            # convert string to lower + remove leade/trail spaces + remove line end
            elem = str(elem).lower().strip().strip("\r\n").strip("\r").strip("\n")
            if (len(elem) > 0):
                # check if element as string is correct or not
                if (elem.isprintable()):
                        # check if alphanumeric (has only letters + numbers)
                        if (elem.isalpha()):
                            nstring = nstring + 1
                        elif (elem.isalnum()):
                            nstring = nstring + 1
                        else:
                            try:
                                # set element to float
                                elem = float(elem)
                                # check if element is number
                                if (isinstance(elem, numbers.Real)):
                                    # check if integer using NEW "float(num).is_integer()"
                                    # https://docs.python.org/2/library/stdtypes.html#float.is_integer
                                    # https://docs.python.org/3/library/stdtypes.html#float.is_integer
                                    if (elem.is_integer()):
                                        ninteger = ninteger + 1
                                    else:
                                        nfloat = nfloat + 1
                                else:
                                    raise AssertionError("Unexpected element type.")

                            except:
                                nstring = nstring + 1
                else:
                    # if strange stuff is present count as invalid
                    ninvalid = ninvalid + 1
            else:
                # if element is empty set missing
                nmissing = nmissing + 1

        nstringp = ((nstring / nelem) * 100)
        nfloatp = ((nfloat / nelem) * 100)
        nintegerp = ((ninteger / nelem) * 100)
        nmissingp = ((nmissing / nelem) * 100)
        ninvalidp = ((ninvalid / nelem) * 100)

        res_string = "Strings: %s (%s%%)\n" % (nstring, nstringp)
        res_sint = "Integers: %s (%s%%)\n" % (ninteger, nintegerp)
        res_sfloat = "Floats: %s (%s%%)\n" % (nfloat, nfloatp)
        res_sinv = "Invalid: %s (%s%%)\n" % (ninvalid, ninvalidp)
        res_smiss = "Missing: %s (%s%%)\n" % (nmissing, nmissingp)

        # get an idea of which kind of data we have
        if (nstring > 0):
            if (nstring > 0) and ((ninteger + nfloat) == 0):
                suggest = "Suggested variable type: Categorical\n"
                commands = "Load dataset -> data = pandas.read_csv('filename_dataset')\n"
                commands = commands + "Instruction (1) -> data['%s'] = data['%s'].astype('category')\n" % (col, col)
            else:
                suggest = "Suggested variable type: Unknown (both strings and numbers are present)\n"
        else:
            if (ninteger > 0) and (nfloat == 0):
                suggest = "Suggested variable type: Numerical (integer)\n"
                commands = "Load dataset -> data = pandas.read_csv('filename_dataset')\n"
                commands = commands + "Instruction (1) -> pandas.to_numeric(data['%s'], errors='coerce')\n" % (col,)
                commands = commands + "Instruction (2) -> data['%s'] = data['%s'].astype('int')\n" % (col, col)
            else:
                suggest = "Suggested variable type: Numerical (float)\n"
                # get the list of instructions to format the variable
                commands = "Load dataset -> data = pandas.read_csv('filename_dataset')\n"
                commands = commands + "Instruction (1) -> pandas.to_numeric(data['%s'], errors='coerce')\n" % (col,)

        if commands:
            result = "Column Name: %s\n%s%s%s%s%s%s%s" % (scol,
                                                          res_string,
                                                          res_sint,
                                                          res_sfloat,
                                                          res_sinv,
                                                          res_smiss,
                                                          suggest,
                                                          commands)
        else:
            result = "Column Name: %s\n%s%s%s%s%s%s" % (scol,
                                                        res_string,
                                                        res_sint,
                                                        res_sfloat,
                                                        res_sinv,
                                                        res_smiss,
                                                        suggest)

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

variables = ['LATITUDE_CIRCLE_IMAGE', 'LONGITUDE_CIRCLE_IMAGE', 'DIAM_CIRCLE_IMAGE']
typecheck(data, select=variables)

sys.exit(0)

###############################################################################

'''
Output Produced

Data Statistics - Dataframe
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| Column                 | Type                |   Elements |   Valid |   Valid (%) |   Missing |   Missing (%) |
+========================+=====================+============+=========+=============+===========+===============+
| CRATER_ID              | categorical         |     384343 |  384343 |  100        |         0 |        0      |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| CRATER_NAME            | categorical         |     384343 |     987 |    0.256802 |    383356 |       99.7432 |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| LATITUDE_CIRCLE_IMAGE  | numerical (float64) |     384343 |  384343 |  100        |         0 |        0      |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| LONGITUDE_CIRCLE_IMAGE | numerical (float64) |     384343 |  384343 |  100        |         0 |        0      |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| DIAM_CIRCLE_IMAGE      | numerical (float64) |     384343 |  384343 |  100        |         0 |        0      |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| DEPTH_RIMFLOOR_TOPOG   | numerical (float64) |     384343 |  384343 |  100        |         0 |        0      |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| MORPHOLOGY_EJECTA_1    | categorical         |     384343 |   44625 |   11.6107   |    339718 |       88.3893 |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| MORPHOLOGY_EJECTA_2    | categorical         |     384343 |   19476 |    5.06735  |    364867 |       94.9327 |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| MORPHOLOGY_EJECTA_3    | categorical         |     384343 |    1293 |    0.336418 |    383050 |       99.6636 |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+
| NUMBER_LAYERS          | numerical (int32)   |     384343 |  384343 |  100        |         0 |        0      |
+------------------------+---------------------+------------+---------+-------------+-----------+---------------+

Data Statistics - Numeric Interval
+------------------------+---------+----------+----------+----------+------------+----------+---------+--------------+
| Variable               |   count |      min |      max |    range |       mean |   median |    mode |     variance |
+========================+=========+==========+==========+==========+============+==========+=========+==============+
| LATITUDE_CIRCLE_IMAGE  |  384343 |  -86.7   |   85.702 |  172.402 | -7.16537   |  -10.038 | -23.634 | 1130.05      |
+------------------------+---------+----------+----------+----------+------------+----------+---------+--------------+
| LONGITUDE_CIRCLE_IMAGE |  384343 | -179.997 |  179.997 |  359.994 | 10.2071    |   12.852 | -53.5   | 9337.99      |
+------------------------+---------+----------+----------+----------+------------+----------+---------+--------------+
| DIAM_CIRCLE_IMAGE      |  384343 |    1     | 1164.22  | 1163.22  |  3.55669   |    1.53  |   1.01  |   73.8223    |
+------------------------+---------+----------+----------+----------+------------+----------+---------+--------------+
| DEPTH_RIMFLOOR_TOPOG   |  384343 |   -0.42  |    4.95  |    5.37  |  0.0758375 |    0     |   0     |    0.04907   |
+------------------------+---------+----------+----------+----------+------------+----------+---------+--------------+
| NUMBER_LAYERS          |  384343 |    0     |    5     |    5     |  0.0648353 |    0     |   0     |    0.0929572 |
+------------------------+---------+----------+----------+----------+------------+----------+---------+--------------+

Data Statistics - Numeric Distribution
+------------------------+---------+-----------+----------+------------+------------+------------+---------+----------+-----------+
| Variable               |   count |       std |      min |   Q1 (25%) |   Q2 (50%) |   Q3 (75%) |     IQR |      max |      skew |
+========================+=========+===========+==========+============+============+============+=========+==========+===========+
| LATITUDE_CIRCLE_IMAGE  |  384343 | 33.6162   |  -86.7   |    -30.914 |    -10.038 |     17.274 |  48.188 |   85.702 |  0.190226 |
+------------------------+---------+-----------+----------+------------+------------+------------+---------+----------+-----------+
| LONGITUDE_CIRCLE_IMAGE |  384343 | 96.6333   | -179.997 |    -58.752 |     12.852 |     89.328 | 148.08  |  179.997 | -0.167513 |
+------------------------+---------+-----------+----------+------------+------------+------------+---------+----------+-----------+
| DIAM_CIRCLE_IMAGE      |  384343 |  8.59199  |    1     |      1.18  |      1.53  |      2.55  |   1.37  | 1164.22  | 23.5265   |
+------------------------+---------+-----------+----------+------------+------------+------------+---------+----------+-----------+
| DEPTH_RIMFLOOR_TOPOG   |  384343 |  0.221518 |   -0.42  |      0     |      0     |      0     |   0     |    4.95  |  4.43317  |
+------------------------+---------+-----------+----------+------------+------------+------------+---------+----------+-----------+
| NUMBER_LAYERS          |  384343 |  0.304889 |    0     |      0     |      0     |      0     |   0     |    5     |  5.68366  |
+------------------------+---------+-----------+----------+------------+------------+------------+---------+----------+-----------+

Frequency Analysis - variable 'LATITUDE_CIRCLE_IMAGE'
| Interval          |   Frequency |   Frequency (%) |   Cumulative Freq. |   Cumulative Freq. (%) |
|-------------------+-------------+-----------------+--------------------+------------------------|
| (-90.0 <-> -72.0] |        5453 |        1.41878  |               5453 |                1.41878 |
| (-72.0 <-> -54.0] |       26622 |        6.92663  |              32075 |                8.34541 |
| (-54.0 <-> -36.0] |       45365 |       11.8033   |              77440 |               20.1487  |
| (-36.0 <-> -18.0] |       78347 |       20.3847   |             155787 |               40.5333  |
| (-18.0 <-> 0.0]   |       77433 |       20.1468   |             233220 |               60.6802  |
| (0.0 <-> 18.0]    |       57158 |       14.8716   |             290378 |               75.5518  |
| (18.0 <-> 36.0]   |       49520 |       12.8843   |             339898 |               88.4361  |
| (36.0 <-> 54.0]   |       27231 |        7.08508  |             367129 |               95.5212  |
| (54.0 <-> 72.0]   |       15350 |        3.99383  |             382479 |               99.515   |
| (72.0 <-> 90)     |        1864 |        0.484983 |             384343 |              100       |

Frequency Analysis - variable 'LONGITUDE_CIRCLE_IMAGE'
| Interval            |   Frequency |   Frequency (%) |   Cumulative Freq. |   Cumulative Freq. (%) |
|---------------------+-------------+-----------------+--------------------+------------------------|
| (-180.0 <-> -144.0] |       33124 |         8.61834 |              33124 |                8.61834 |
| (-144.0 <-> -108.0] |       21185 |         5.512   |              54309 |               14.1303  |
| (-108.0 <-> -72.0]  |       27935 |         7.26825 |              82244 |               21.3986  |
| (-72.0 <-> -36.0]   |       43119 |        11.2189  |             125363 |               32.6175  |
| (-36.0 <-> 0.0]     |       48396 |        12.5919  |             173759 |               45.2094  |
| (0.0 <-> 36.0]      |       51676 |        13.4453  |             225435 |               58.6546  |
| (36.0 <-> 72.0]     |       42231 |        10.9878  |             267666 |               69.6425  |
| (72.0 <-> 108.0]    |       42305 |        11.0071  |             309971 |               80.6496  |
| (108.0 <-> 144.0]   |       39460 |        10.2669  |             349431 |               90.9164  |
| (144.0 <-> 180)     |       34912 |         9.08355 |             384343 |              100       |

Frequency Analysis - variable 'DIAM_CIRCLE_IMAGE'
| Interval               |   Frequency |   Frequency (%) |   Cumulative Freq. |   Cumulative Freq. (%) |
|------------------------+-------------+-----------------+--------------------+------------------------|
| (1.0 <-> 117.322]      |      384160 |    99.9524      |             384160 |                99.9524 |
| (117.322 <-> 233.644]  |         148 |     0.0385073   |             384308 |                99.9909 |
| (233.644 <-> 349.966]  |          24 |     0.00624442  |             384332 |                99.9971 |
| (349.966 <-> 466.288]  |           6 |     0.00156111  |             384338 |                99.9987 |
| (466.288 <-> 582.61]   |           2 |     0.000520369 |             384340 |                99.9992 |
| (582.61 <-> 698.932]   |           1 |     0.000260184 |             384341 |                99.9995 |
| (698.932 <-> 815.254]  |           0 |     0           |             384341 |                99.9995 |
| (815.254 <-> 931.576]  |           0 |     0           |             384341 |                99.9995 |
| (931.576 <-> 1047.898] |           0 |     0           |             384341 |                99.9995 |
| (1047.898 <-> 1164.22) |           2 |     0.000520369 |             384343 |               100      |

Column Name: 'LATITUDE_CIRCLE_IMAGE'
Strings: 0 (0.0%)
Integers: 381 (0.09913020401048021%)
Floats: 383962 (99.90086979598952%)
Invalid: 0 (0.0%)
Missing: 0 (0.0%)
Suggested variable type: Numerical (float)
Load dataset -> data = pandas.read_csv('filename_dataset')
Instruction (1) -> pandas.to_numeric(data['LATITUDE_CIRCLE_IMAGE'], errors='coerce')

Column Name: 'LONGITUDE_CIRCLE_IMAGE'
Strings: 0 (0.0%)
Integers: 360 (0.09366633449809154%)
Floats: 383983 (99.90633366550192%)
Invalid: 0 (0.0%)
Missing: 0 (0.0%)
Suggested variable type: Numerical (float)
Load dataset -> data = pandas.read_csv('filename_dataset')
Instruction (1) -> pandas.to_numeric(data['LONGITUDE_CIRCLE_IMAGE'], errors='coerce')

Column Name: 'DIAM_CIRCLE_IMAGE'
Strings: 0 (0.0%)
Integers: 5014 (1.3045638921484195%)
Floats: 379329 (98.69543610785158%)
Invalid: 0 (0.0%)
Missing: 0 (0.0%)
Suggested variable type: Numerical (float)
Load dataset -> data = pandas.read_csv('filename_dataset')
Instruction (1) -> pandas.to_numeric(data['DIAM_CIRCLE_IMAGE'], errors='coerce')

'''

