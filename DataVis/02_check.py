# -*- coding: utf-8 -*-
import sys

from collections import OrderedDict
from tabulate import tabulate

import numpy as np
import pandas as pd

###############################################################################


def isNumber(number):
    import numbers
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
    #tot_elem = data[].shape[0]
    tot_elem = len(nparr)
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

# set new column data type
data["LONGITUDE_EAST_360"] = data["LONGITUDE_EAST_360"].astype('float64')

print("Frequency Analysis - variable 'LATITUDE_CIRCLE_IMAGE'")
nparray = data['LATITUDE_CIRCLE_IMAGE']
latitude = freqency_table(nparray, 10, -90, 90)
print(latitude)

print("Frequency Analysis - variable 'LONGITUDE_EAST_360'")
nparray = data['LONGITUDE_EAST_360']
longitude = freqency_table(nparray, 10, 0, 360)
print(longitude)

print("Frequency Analysis - variable 'DIAM_CIRCLE_IMAGE'")
nparray = data['DIAM_CIRCLE_IMAGE']
diameter = freqency_table(nparray, 10)
print(diameter)
