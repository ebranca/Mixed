# -*- coding: utf-8 -*-
import sys
# import io

from collections import OrderedDict
from tabulate import tabulate

import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    # splits a dictionary into 2 nparray, one for K, une for V
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
    elif (cemis == 0):
        cemis = "Equator"
    elif (cemis > 0 and cemis <= 90):
        cemis = "North"
    else:
        raise AssertionError("unexpected Latitude value")
    # set value
    data.set_value(index, 'HEMISPHERE', cemis)

# set new column data type
data["LONGITUDE_EAST_360"] = data["LONGITUDE_EAST_360"].astype('float64')
data["QUADRANGLE"] = data["QUADRANGLE"].astype('category')
data["HEMISPHERE"] = data["HEMISPHERE"].astype('category')

###############################################################################

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

