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

###############################################################################

def print_longitude_hist(dataframe):
    fig = plt.figure(figsize=(9,6))
    ax = plt.subplot(111)
    ax.set_xlim(0, 360)
    numBins = 18
    ax.hist(dataframe,numBins,color='orange',alpha=0.8)
    # add graph TITLE
    ax.set_title("Mars Craters - Frequency by Longitude (System: Planetocentric)",fontsize=14)
    # add graph X axis lable
    ax.set_xlabel("Longitude East (degrees)",fontsize=12)
    # add graph Y axis lable
    ax.set_ylabel("Frequency",fontsize=12)
    # show plot figure
    #plt.show()
    # save plot figure
    filename = 'graph_1.png'
    plt.savefig(filename, dpi = 150)
    plt.close()

###############################################################################

def print_latitude_hist(dataframe):
    fig = plt.figure(figsize=(9,6))
    ax = plt.subplot(111)
    ax.set_xlim(-90, 90)
    numBins = 18
    ax.hist(dataframe,numBins,color='green',alpha=0.8)
    # add graph TITLE
    ax.set_title("Mars Craters - Frequency by Latitude (System: Planetocentric)",fontsize=14)
    # add graph X axis lable
    ax.set_xlabel("Latitude (degrees)",fontsize=12)
    # add graph Y axis lable
    ax.set_ylabel("Frequency",fontsize=12)
    # show plot figure
    #plt.show()
    # save plot figure
    filename = 'graph_2.png'
    plt.savefig(filename, dpi = 150)
    plt.close()

###############################################################################

def print_diameter_scatter(dataframe):
    # count unique values
    counters = uniquevals(dataframe)
    # split count in nparrays (X -> values), (Y -> frequency)
    arrx, arry = splitdict(counters)
    # prepare plot figure
    fig = plt.figure(figsize=(9,6))
    ax = plt.subplot(111)
    # set X axis parameters
    lowrngx, higrngx, major_ticks_x, xmin, xmax =axis_major_ticks(arrx)
    ax.set_xticks(major_ticks_x)
    ax.set_xlim([lowrngx, higrngx])
    # set Y axis parameters
    lowrngy, higrngy, major_ticks_y, ymin, ymax =axis_major_ticks(arry)
    ax.set_yticks(major_ticks_y)
    ax.set_ylim([lowrngy, higrngy])
    # draw line to mark X range minimum value
    # draw a line from the point (x1, y1) to the point (x2, y2)
    # but for matplotlib has to be expressed as (x1, x2), (y1, y2)
    plt.plot((xmin, xmin),
             (lowrngy, higrngy),
             linewidth=2,
             linestyle=":",
             solid_capstyle="butt",
             marker='+',
             c="red")
    # draw line to mark X range maximum value
    # draw a line from the point (x1, y1) to the point (x2, y2)
    # but for matplotlib has to be expressed as (x1, x2), (y1, y2)
    plt.plot((xmax, xmax),
             (lowrngy, higrngy),
             linewidth=2,
             linestyle=":",
             solid_capstyle="butt",
             marker='+',
             c="red")
    # draw a scatter plot with (s default: 20)
    ax.scatter(arrx, arry,
               s=15,
               c='blue',
               marker='o')
    # draw a plot - this draws a line connecting each dot of the scatter plot
    ax.plot(arrx, arry,
            linewidth=1.0,
            linestyle="-",
            solid_capstyle="butt")
    # add graph TITLE
    ax.set_title("Mars Craters - Frequency by diameter (km)",fontsize=14)
    # add graph X axis lable
    ax.set_xlabel("Diameter (Km)",fontsize=12)
    # add graph Y axis lable
    ax.set_ylabel("Frequency",fontsize=12)

    # show plot figure
    #plt.show()
    # save plot figure
    filename = 'graph_3.png'
    plt.savefig(filename, dpi = 150)
    plt.close()

def print_diameter_scatter_log(dataframe):
    # count unique values
    counters = uniquevals(dataframe)
    # split count in nparrays (X -> values), (Y -> frequency)
    arrx, arry = splitdict(counters)
    # prepare plot figure

    fig = plt.figure(figsize=(9,6))
    ax = plt.subplot(111)

#    fig, ax = plt.subplots(figsize=(9, 6))

    # FIRST transform the values to log base 10
    ax.set_xscale('log')
    # SECOND set axis limits
    ax.set_xlim((arrx.min() - 0.15), (arrx.max() + 50))
    ax.set_ylim(arry.min(), arry.max())
    # draw a scatter plot with (s default: 20)
    ax.scatter(arrx, arry,
               s=20,
               c='blue',
               marker='.')
    # draw a plot - this draws a line connecting each dot of the scatter plot
    ax.plot(arrx, arry,
            linewidth=1.5,
            linestyle="-",
            c='blue',
            solid_capstyle="butt")

    # add graph TITLE
    ax.set_title("Mars Craters - Frequency by diameter (km)",fontsize=14)
    # add graph X axis lable
    ax.set_xlabel("Diameter (Km) [scale: log base 10 (x)]",fontsize=12)
    # add graph Y axis lable
    ax.set_ylabel("Frequency",fontsize=12)

    # show plot figure
    #plt.show()
    # save plot figure
    filename = 'graph_4.png'
    plt.savefig(filename, dpi = 150)
    plt.close()


###############################################################################

def bivar_scatter_latlon(lat, lon):
    fig, ax = plt.subplots(figsize=(12, 7))
    # set major ticks every 20
    x_major_ticks = np.arange(0, 361, 20)
    ax.set_xticks(x_major_ticks)
    y_major_ticks = np.arange(-90, 91, 10)
    ax.set_yticks(y_major_ticks)

    ax.set_xlim([0, 360])
    ax.set_ylim([-90, 90])

    ax.scatter(lon,
               lat,
               s=0.5,
               c='purple',
               marker='.')

    ax.set_title("Crater Distribution on Mars Surface (System: Planetocentric)",fontsize=14)
    ax.set_xlabel("LONGITUDE East (degrees)",fontsize=12)
    ax.set_ylabel("LATITUDE (degrees)",fontsize=12)

    # show plot figure
    #plt.show()
    # save plot figure
    filename = 'graph_5.png'
    plt.savefig(filename, dpi = 150)
    plt.close()

def bivar_hexbin_latlon(lat, lon):
    fig, ax = plt.subplots(figsize=(13, 7))
    # set major ticks every 20
    x_major_ticks = np.arange(0, 361, 20)
    ax.set_xticks(x_major_ticks)
    y_major_ticks = np.arange(-90, 91, 10)
    ax.set_yticks(y_major_ticks)

    ax.set_xlim([0, 360])
    ax.set_ylim([-90, 90])

    plt.hexbin(lon,
               lat,
               gridsize=140,
               cmap=mpl.cm.plasma,
               bins=None)

    cb = plt.colorbar()
    cb.set_label('Frequency')

    ax.set_title("Crater Distribution on Mars Surface (System: Planetocentric)",fontsize=14)
    ax.set_xlabel("LONGITUDE East (degrees)",fontsize=12)
    ax.set_ylabel("LATITUDE (degrees)",fontsize=12)

    # show plot figure
    #plt.show()
    # save plot figure
    filename = 'graph_6.png'
    plt.savefig(filename, dpi = 150)
    plt.close()


###############################################################################

def bivar_scatter_londia():
    arrx = data["LONGITUDE_EAST_360"]
    arry = data["DIAM_CIRCLE_IMAGE"]

    # prepare plot figure
    fig, ax = plt.subplots(figsize=(12, 7))
    # FIRST transform the values to log base 10
    ax.set_yscale('log')
    # SECOND set axis limits
    ax.set_xlim(arrx.min(), arrx.max())
    ax.set_ylim(arry.min(), arry.max())
    # draw a scatter plot with (s default: 20)
    ax.scatter(arrx, arry,
               s=0.4,
               c='blue',
               marker='.')

    # add graph TITLE
    ax.set_title("Mars Craters - Diameter by Longitude East (Planetocentric)",fontsize=14)
    # add graph X axis lable
    ax.set_xlabel("LONGITUDE East (degrees)",fontsize=12)
    # add graph Y axis lable
    ax.set_ylabel("Diameter (Km) [scale: log base 10 (y)]",fontsize=12)

    # show plot figure
    #plt.show()
    # save plot figure
    filename = 'graph_7.png'
    plt.savefig(filename, dpi = 150)
    plt.close()

###############################################################################

# plot
#sns.set_style('white')
sns.set_color_codes()

# VARIABLE "LONGITUDE_EAST_360" - exploratory scatter plot
nparr = data["LONGITUDE_EAST_360"]
print_longitude_hist(nparr)

# VARIABLE "LATITUDE_CIRCLE_IMAGE" - exploratory scatter plot
nparr = data["LATITUDE_CIRCLE_IMAGE"]
print_latitude_hist(nparr)

# VARIABLE "DIAM_CIRCLE_IMAGE" - exploratory scatter plot
nparr = data["DIAM_CIRCLE_IMAGE"]
print_diameter_scatter(nparr)
print_diameter_scatter_log(nparr)

# BIVARIATE- "LONGITUDE_EAST_360" + "LATITUDE_CIRCLE_IMAGE"
latitude = data["LATITUDE_CIRCLE_IMAGE"]
longitude = data["LONGITUDE_EAST_360"]
bivar_scatter_latlon(latitude, longitude)
bivar_hexbin_latlon(latitude, longitude)

# BIVARIATE- "LONGITUDE_EAST_360" + "DIAM_CIRCLE_IMAGE"
bivar = data[["LONGITUDE_EAST_360", "DIAM_CIRCLE_IMAGE"]]
bivar_scatter_londia()

sys.exit(0)

