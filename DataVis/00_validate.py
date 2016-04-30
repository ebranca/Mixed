# -*- coding: utf-8 -*-

import sys
import io

from collections import OrderedDict
import string
from tabulate import tabulate

#import os

#import numpy as np
#import pandas as pd

###############################################################################
## HELPER FUnCTIONS
###############################################################################

def isprintable(data_string):
    printset = set(string.printable)
    isprintable = set(data_string).issubset(printset)
    return isprintable


###############################################################################

dataset = 'marscrater_pds.csv'

# iterate over the file to count lines
linecount = 0
with io.open(dataset, 'rb') as rdt:
    for line in rdt:
        linecount = linecount + 1

#print("Total lines detected in the source file: %s" % (linecount,))

# now we have the number of lines and we can start loading and cleaning data

#variables = OrderedDict()
variables = list()

valid_datalines = list()
invalid_datalines = list()

unique_ids = OrderedDict()

validlines = 0
invalidlines = 0
emptylines = 0

ID_valid = 0

NAME_valid = 0
NAME_missing = 0

LATITUDE_valid = 0
LATITUDE_missing = 0

LONGITUDE_valid = 0
LONGITUDE_missing = 0

DIAM_CIRCLE_valid = 0
DIAM_CIRCLE_missing = 0

DEPTH_RIMFLOOR_valid = 0
DEPTH_RIMFLOOR_missing = 0

EJECTA_1_valid = 0
EJECTA_1_missing = 0

EJECTA_2_valid = 0
EJECTA_2_missing = 0

EJECTA_3_valid = 0
EJECTA_3_missing = 0

LAYERS_valid = 0
LAYERS_missing = 0


counter = 0
with io.open(dataset, 'rb') as rdc:
    for line in rdc:
        counter = counter + 1

        # original file starts with UTF8-BOM characters "\xef\xbb\xbf"
        # we have to decode the bytes using "utf-8-sig" string to remove BOM
        # from: '\ufeffCRATER_ID,....
        # to:   'CRATER_ID,.......
        line = line.decode("utf-8-sig")

        # then we have to remove line endings to keep only data
        # from: ,NUMBER_LAYERS\r\n'
        # to:   ,NUMBER_LAYERS'
        line = line.strip("\r\n").strip("\n").strip("\r")

        #print(ascii(line))

        # if line is not empty we process it
        line = line.strip()
        if (len(line) > 0):
            # now that we have removed all useless stuff we check if we have data
            # we are reading a CSV so each line has to be split at ","
            sline = line.split(",")

            # we expect a file with 10 variables (colums) so we check this
            if not (len(sline) == 10):
                # line is invalid, we count it and then skip it
                invalidlines = invalidlines + 1
                print("FAIL-0")
                pass

            # if we are reading line 1 --> count number of variables (columns)
            if counter == 1:
                for var in sline:
                    variables.append(var)
                #print(variables)

                valid_datalines.append(line + "\n")
                validlines = validlines + 1
                continue

            # if we are reading any other line after the first one
            else:
                ##############################################################
                # DATA VALIDATION - START
                ##############################################################
                # now we split each data line and check each variable
                ID = sline[0]

                ##############################################################
                # DATA VALIDATION - ID (unique id)
                ##############################################################
                # ID is supposed to be valid and UNIQUE for each record
                # if any problem is found the record will be skipped
                # and the invalid row saved in a second file

                # we expect an ID like "30-013690" so we test for this
                # first check that field length is equal to 9 characters
                if (len(ID) == 9):
                    # check that the separator is present and in proper place
                    if ("-" in ID) and (ID[2] == "-"):
                        # split the string at the separator
                        sid = ID.split("-")
                        # check that each section contain only digits
                        if (sid[0].isdigit()) and (sid[1].isdigit()):
                            # check that each section has proper length
                            if (len(sid[0]) == 2) and (len(sid[1]) == 6):
                                # convert string in ascii
                                xID = ascii(sline[0])
                                # check that ID record in UNIQUE
                                if not (xID in unique_ids):
                                    # if ID string IS NOT in dictionary
                                    # then all is good as string is UNIQUE
                                    # so we save "ID -> line number"
                                    unique_ids[xID] = int(counter - 1)
                                    ID_valid = ID_valid + 1

                                    # if all tests are OK then change "-" with "_"
                                    # so we can use is as index or ID in python
                                    # Identifiers (also referred to as names)
                                    # are described by the following lexical definitions:
                                    # identifier ::=  (letter|"_") (letter | digit | "_")*
                                    # letter     ::=  lowercase | uppercase
                                    # lowercase  ::=  "a"..."z"
                                    # uppercase  ::=  "A"..."Z"
                                    # digit      ::=  "0"..."9"
                                    ID = xID.replace("-", "_")

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-1")
                                    continue
                            else:
                                # WE HAVE A PROBLEM, string goes in special list
                                invalid_datalines.append(line + "\n")
                                invalidlines = invalidlines + 1
                                print("FAIL-2")
                                continue
                        else:
                            # WE HAVE A PROBLEM, string goes in special list
                            invalid_datalines.append(line + "\n")
                            invalidlines = invalidlines + 1
                            print("FAIL-3")
                            continue
                    else:
                        # WE HAVE A PROBLEM, string goes in special list
                        invalid_datalines.append(line + "\n")
                        invalidlines = invalidlines + 1
                        print("FAIL-4")
                        continue
                else:
                    # WE HAVE A PROBLEM, string goes in special list
                    invalid_datalines.append(line + "\n")
                    invalidlines = invalidlines + 1
                    print("FAIL-5")
                    continue

                ##############################################################
                # DATA VALIDATION - NAME (crater name)
                ##############################################################
                NAME = sline[1]

                # remove leading and trailing spaces
                NAME = NAME.strip()

                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in NAME):
                    NAME = NAME.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(NAME) > 0):

                    # check that all characters are UTF-8 printable
                    if (isprintable(NAME)):
                        # if all chars are OK, be sure value is a string
                        NAME = str(NAME)

                    else:
                        # if strange chars are present, force convert to ASCII
                        NAME = ascii(NAME)
                    NAME_valid = NAME_valid + 1

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    NAME = "NA"
                    NAME_missing = NAME_missing + 1

                ##############################################################
                # DATA VALIDATION - LATITUDE (float, from -90 to +90)
                ##############################################################
                LATITUDE = sline[2]

                # remove leading and trailing spaces
                LATITUDE = LATITUDE.strip()
                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in LATITUDE):
                    LATITUDE = LATITUDE.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(LATITUDE) > 0):
                    # check that all characters are UTF-8 printable
                    if (isprintable(LATITUDE)):
                        # we remove all valid chars, if anything remains, string is BAD
                        if (LATITUDE.strip('1234567890-+.')):
                            # WE HAVE A PROBLEM, string goes in special list
                            invalid_datalines.append(line + "\n")
                            invalidlines = invalidlines + 1
                            print("FAIL-6")
                            continue

                        else:
                            # now that we have a clean string we test if is valid
                            if ("." in LATITUDE):
                                # if we have a float we split into sections and check each
                                sLATITUDE = LATITUDE.split(".")

                                # if first section has sign we remove it
                                if (("-" or "+") in sLATITUDE[0]):
                                    sLATITUDE[0] = sLATITUDE[0].strip("+-")

                                # string may be like ".194", if so we add a leading zero
                                if (len(sLATITUDE[0]) == 0):
                                    sLATITUDE[0] = "0"

                                # then check if each section is composed only of digits
                                if (sLATITUDE[0].isdigit() and sLATITUDE[1].isdigit()):
                                    xLATITUDE = float(LATITUDE)

                                    # then we check if the values are within accepted range
                                    if ((xLATITUDE < (-90.0)) or (xLATITUDE > (90.0))):
                                        # WE HAVE A PROBLEM, string goes in special list
                                        invalid_datalines.append(line + "\n")
                                        invalidlines = invalidlines + 1
                                        print("FAIL-7")
                                        continue

                                    else:
                                        # if all tests are OK then we can set the value
                                        LATITUDE = xLATITUDE
                                        LATITUDE_valid = LATITUDE_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-8")
                                    continue

                            # if string is not a float
                            else:

                                # if first section has sign we remove it
                                if (("-" or "+") in LATITUDE):
                                    LATITUDE = LATITUDE.strip("+-")

                                # then check if each section is composed only of digits
                                if (LATITUDE.isdigit()):
                                    xLATITUDE = float(LATITUDE)

                                    # then we check if the values are within accepted range
                                    if ((xLATITUDE < (-90.0)) or (xLATITUDE > (90.0))):
                                        # WE HAVE A PROBLEM, string goes in special list
                                        invalid_datalines.append(line + "\n")
                                        invalidlines = invalidlines + 1
                                        print("FAIL-9")
                                        continue

                                    else:
                                        # if all tests are OK then we can set the value
                                        LATITUDE = xLATITUDE
                                        LATITUDE_valid = LATITUDE_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-10")
                                    continue

                    else:
                        # WE HAVE A PROBLEM, string goes in special list
                        invalid_datalines.append(line + "\n")
                        invalidlines = invalidlines + 1
                        print("FAIL-11")
                        continue

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    LATITUDE = "NA"
                    LATITUDE_missing = LATITUDE_missing + 1

                ##############################################################
                # DATA VALIDATION - LONGITUDE (float, from -180 to +180)
                ##############################################################
                LONGITUDE = sline[3]

                # remove leading and trailing spaces
                LONGITUDE = LONGITUDE.strip()
                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in LONGITUDE):
                    LONGITUDE = LONGITUDE.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(LONGITUDE) > 0):
                    # check that all characters are UTF-8 printable
                    if (isprintable(LONGITUDE)):
                        # we remove all valid chars, if anything remains, string is BAD
                        if (LONGITUDE.strip('1234567890-+.')):
                            # WE HAVE A PROBLEM, string goes in special list
                            invalid_datalines.append(line + "\n")
                            invalidlines = invalidlines + 1
                            print("FAIL-12")
                            continue

                        else:
                            # now that we have a clean string we test if is valid
                            if ("." in LONGITUDE):
                                # if we have a float we split into sections and check each
                                sLONGITUDE = LONGITUDE.split(".")

                                # if first section has sign we remove it
                                if (("-" or "+") in sLONGITUDE[0]):
                                    sLONGITUDE[0] = sLONGITUDE[0].strip("+-")

                                # string may be like ".194", if so we add a leading zero
                                if (len(sLONGITUDE[0]) == 0):
                                    sLONGITUDE[0] = "0"

                                # then check if each section is composed only of digits
                                if (sLONGITUDE[0].isdigit() and sLONGITUDE[1].isdigit()):
                                    xLONGITUDE = float(LONGITUDE)

                                    # then we check if the values are within accepted range
                                    if ((xLONGITUDE < (-180.0)) or (xLONGITUDE > (180.0))):
                                        # WE HAVE A PROBLEM, string goes in special list
                                        invalid_datalines.append(line + "\n")
                                        invalidlines = invalidlines + 1
                                        print("FAIL-13")
                                        continue

                                    else:
                                        # if all tests are OK then we can set the value
                                        LONGITUDE = xLONGITUDE
                                        LONGITUDE_valid = LONGITUDE_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-14")
                                    continue

                            # if string is not a float
                            else:

                                # if first section has sign we remove it
                                if (("-" or "+") in LONGITUDE):
                                    LONGITUDE = LONGITUDE.strip("+-")

                                # then check if each section is composed only of digits
                                if (LONGITUDE.isdigit()):
                                    xLONGITUDE = float(LONGITUDE)

                                    # then we check if the values are within accepted range
                                    if ((xLONGITUDE < (-180.0)) or (xLONGITUDE > (180.0))):
                                        # WE HAVE A PROBLEM, string goes in special list
                                        invalid_datalines.append(line + "\n")
                                        invalidlines = invalidlines + 1
                                        print("FAIL-15")
                                        continue

                                    else:
                                        # if all tests are OK then we can set the value
                                        LONGITUDE = xLONGITUDE
                                        LONGITUDE_valid = LONGITUDE_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-16")
                                    continue

                    else:
                        # WE HAVE A PROBLEM, string goes in special list
                        invalid_datalines.append(line + "\n")
                        invalidlines = invalidlines + 1
                        print("FAIL-17")
                        continue

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    LONGITUDE = "NA"
                    LONGITUDE_missing = LONGITUDE_missing + 1

                ##############################################################
                # DATA VALIDATION - DIAM_CIRCLE (float)
                ##############################################################
                DIAM_CIRCLE = sline[4]

                # remove leading and trailing spaces
                DIAM_CIRCLE = DIAM_CIRCLE.strip()
                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in DIAM_CIRCLE):
                    DIAM_CIRCLE = DIAM_CIRCLE.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(DIAM_CIRCLE) > 0):
                    # check that all characters are UTF-8 printable
                    if (isprintable(DIAM_CIRCLE)):
                        # we remove all valid chars, if anything remains, string is BAD
                        if (DIAM_CIRCLE.strip('1234567890-+.')):
                            # WE HAVE A PROBLEM, string goes in special list
                            invalid_datalines.append(line + "\n")
                            invalidlines = invalidlines + 1
                            print("FAIL-18")
                            continue

                        else:
                            # now that we have a clean string we test if is valid
                            if ("." in DIAM_CIRCLE):
                                # if we have a float we split into sections and check each
                                sDIAM_CIRCLE = DIAM_CIRCLE.split(".")

                                # if first section has sign we remove it
                                if (("-" or "+") in sDIAM_CIRCLE[0]):
                                    sDIAM_CIRCLE[0] = sDIAM_CIRCLE[0].strip("+-")

                                # string may be like ".194", if so we add a leading zero
                                if (len(sDIAM_CIRCLE[0]) == 0):
                                    sDIAM_CIRCLE[0] = "0"

                                # then check if each section is composed only of digits
                                if (sDIAM_CIRCLE[0].isdigit() and sDIAM_CIRCLE[1].isdigit()):

                                    # if all tests are OK then we can set the value
                                    DIAM_CIRCLE = float(DIAM_CIRCLE)
                                    DIAM_CIRCLE_valid = DIAM_CIRCLE_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-19")
                                    continue

                            # if string is not a float
                            else:

                                # if first section has sign we remove it
                                if (("-" or "+") in DIAM_CIRCLE):
                                    DIAM_CIRCLE = DIAM_CIRCLE.strip("+-")

                                # then check if each section is composed only of digits
                                if (DIAM_CIRCLE.isdigit()):

                                    # if all tests are OK then we can set the value
                                    DIAM_CIRCLE = float(DIAM_CIRCLE)
                                    DIAM_CIRCLE_valid = DIAM_CIRCLE_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-20")
                                    continue

                    else:
                        # WE HAVE A PROBLEM, string goes in special list
                        invalid_datalines.append(line + "\n")
                        invalidlines = invalidlines + 1
                        print("FAIL-21")
                        continue

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    DIAM_CIRCLE = "NA"
                    DIAM_CIRCLE_missing = DIAM_CIRCLE_missing + 1


                ##############################################################
                # DATA VALIDATION - DEPTH_RIMFLOOR (float)
                ##############################################################
                DEPTH_RIMFLOOR = sline[5]


                # remove leading and trailing spaces
                DEPTH_RIMFLOOR = DEPTH_RIMFLOOR.strip()
                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in DEPTH_RIMFLOOR):
                    DEPTH_RIMFLOOR = DEPTH_RIMFLOOR.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(DEPTH_RIMFLOOR) > 0):
                    # check that all characters are UTF-8 printable
                    if (isprintable(DEPTH_RIMFLOOR)):
                        # we remove all valid chars, if anything remains, string is BAD
                        if (DEPTH_RIMFLOOR.strip('1234567890-+.')):
                            # WE HAVE A PROBLEM, string goes in special list
                            invalid_datalines.append(line + "\n")
                            invalidlines = invalidlines + 1
                            print("FAIL-22")
                            continue

                        else:
                            # now that we have a clean string we test if is valid
                            if ("." in DEPTH_RIMFLOOR):
                                # if we have a float we split into sections and check each
                                sDEPTH_RIMFLOOR = DEPTH_RIMFLOOR.split(".")

                                # if first section has sign we remove it
                                if (("-" or "+") in sDEPTH_RIMFLOOR[0]):
                                    sDEPTH_RIMFLOOR[0] = sDEPTH_RIMFLOOR[0].strip("+-")

                                # string may be like ".194", if so we add a leading zero
                                if (len(sDEPTH_RIMFLOOR[0]) == 0):
                                    sDEPTH_RIMFLOOR[0] = "0"

                                # then check if each section is composed only of digits
                                if (sDEPTH_RIMFLOOR[0].isdigit() and sDEPTH_RIMFLOOR[1].isdigit()):

                                    # if all tests are OK then we can set the value
                                    DEPTH_RIMFLOOR = float(DEPTH_RIMFLOOR)
                                    DEPTH_RIMFLOOR_valid = DEPTH_RIMFLOOR_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-23")
                                    continue

                            # if string is not a float
                            else:

                                # if first section has sign we remove it
                                if (("-" or "+") in DEPTH_RIMFLOOR):
                                    DEPTH_RIMFLOOR = DEPTH_RIMFLOOR.strip("+-")

                                # then check if each section is composed only of digits
                                if (DEPTH_RIMFLOOR.isdigit()):

                                    # if all tests are OK then we can set the value
                                    DEPTH_RIMFLOOR = float(DEPTH_RIMFLOOR)
                                    DEPTH_RIMFLOOR_valid = DEPTH_RIMFLOOR_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-24")
                                    continue

                    else:
                        # WE HAVE A PROBLEM, string goes in special list
                        invalid_datalines.append(line + "\n")
                        invalidlines = invalidlines + 1
                        print("FAIL-25")
                        continue

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    DEPTH_RIMFLOOR = "NA"
                    DEPTH_RIMFLOOR_missing = DEPTH_RIMFLOOR_missing + 1

                ##############################################################
                # DATA VALIDATION - EJECTA_1 (crater name)
                ##############################################################
                EJECTA_1 = sline[6]

                # remove leading and trailing spaces
                EJECTA_1 = EJECTA_1.strip()
                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in EJECTA_1):
                    EJECTA_1 = EJECTA_1.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(EJECTA_1) > 0):
                    # check that all characters are UTF-8 printable
                    if (isprintable(EJECTA_1)):
                        # if all chars are OK, be sure value is a string
                        EJECTA_1 = str(EJECTA_1)
                    else:
                        # if strange chars are present, force convert to ASCII
                        EJECTA_1 = ascii(EJECTA_1)
                    EJECTA_1_valid = EJECTA_1_valid + 1

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    EJECTA_1 = "NA"
                    EJECTA_1_missing = EJECTA_1_missing + 1

                ##############################################################
                # DATA VALIDATION - EJECTA_2 (crater name)
                ##############################################################
                EJECTA_2= sline[7]

                # remove leading and trailing spaces
                EJECTA_2 = EJECTA_2.strip()
                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in EJECTA_2):
                    EJECTA_2 = EJECTA_2.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(EJECTA_2) > 0):
                    # check that all characters are UTF-8 printable
                    if (isprintable(EJECTA_2)):
                        # if all chars are OK, be sure value is a string
                        EJECTA_2 = str(EJECTA_2)
                    else:
                        # if strange chars are present, force convert to ASCII
                        EJECTA_2 = ascii(EJECTA_2)
                    EJECTA_2_valid = EJECTA_2_valid + 1

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    EJECTA_2 = "NA"
                    EJECTA_2_missing = EJECTA_2_missing + 1

                ##############################################################
                # DATA VALIDATION - EJECTA_3 (crater name)
                ##############################################################
                EJECTA_3 = sline[8]

                # remove leading and trailing spaces
                EJECTA_3 = EJECTA_3.strip()
                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in EJECTA_3):
                    EJECTA_3 = EJECTA_3.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(EJECTA_3) > 0):
                    # check that all characters are UTF-8 printable
                    if (isprintable(EJECTA_3)):
                        # if all chars are OK, be sure value is a string
                        EJECTA_3 = str(EJECTA_3)
                    else:
                        # if strange chars are present, force convert to ASCII
                        EJECTA_3 = ascii(EJECTA_3)
                    EJECTA_3_valid = EJECTA_3_valid + 1

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    EJECTA_3 = "NA"
                    EJECTA_3_missing = EJECTA_3_missing + 1

                ##############################################################
                # DATA VALIDATION - LAYERS (crater name)
                ##############################################################
                LAYERS = sline[9]

                # remove leading and trailing spaces
                LAYERS = LAYERS.strip()
                # check if line ending are present and remove them
                if (("\r\n" or "\n" or "\r") in LAYERS):
                    LAYERS = LAYERS.strip("\r\n").strip("\n").strip("\r")

                # check that the string is NOT empty
                if (len(LAYERS) > 0):
                    # check that all characters are UTF-8 printable
                    if (isprintable(LAYERS)):
                        # we remove all valid chars, if anything remains, string is BAD
                        if (LAYERS.strip('1234567890-+.')):
                            # WE HAVE A PROBLEM, string goes in special list
                            invalid_datalines.append(line + "\n")
                            invalidlines = invalidlines + 1
                            print("FAIL-26")
                            continue

                        else:
                            # now that we have a clean string we test if is valid
                            if ("." in LAYERS):
                                # if we have a float we split into sections and check each
                                sLAYERS = LAYERS.split(".")

                                # if first section has sign we remove it
                                if (("-" or "+") in sLAYERS[0]):
                                    sLAYERS[0] = sLAYERS[0].strip("+-")

                                # string may be like ".194", if so we add a leading zero
                                if (len(sLAYERS[0]) == 0):
                                    sLAYERS[0] = "0"

                                # then check if each section is composed only of digits
                                if (sLAYERS[0].isdigit() and sLAYERS[1].isdigit()):

                                    # if all tests are OK then we can set the value
                                    LAYERS = int(LAYERS)
                                    LAYERS_valid = LAYERS_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-27")
                                    continue

                            # if string is not a float
                            else:

                                # if first section has sign we remove it
                                if (("-" or "+") in LAYERS):
                                    LAYERS = LAYERS.strip("+-")

                                # then check if each section is composed only of digits
                                if (LAYERS.isdigit()):

                                    # if all tests are OK then we can set the value
                                    LAYERS = float(LAYERS)
                                    LAYERS_valid = LAYERS_valid + 1

                                else:
                                    # WE HAVE A PROBLEM, string goes in special list
                                    invalid_datalines.append(line + "\n")
                                    invalidlines = invalidlines + 1
                                    print("FAIL-28")
                                    continue

                    else:
                        # WE HAVE A PROBLEM, string goes in special list
                        invalid_datalines.append(line + "\n")
                        invalidlines = invalidlines + 1
                        print("FAIL-29")
                        continue

                # if string is empty
                else:
                    # we assign the value "NA" to the string (NA -> Not Available)
                    LAYERS = "NA"
                    LAYERS_missing = LAYERS_missing + 1

                ##############################################################
                # DATA VALIDATION - END
                ##############################################################

            validlines = validlines + 1

            # now we create new lines for our data

            newdata = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (ID,
                                                         NAME,
                                                         LATITUDE,
                                                         LONGITUDE,
                                                         DIAM_CIRCLE,
                                                         DEPTH_RIMFLOOR,
                                                         EJECTA_1,
                                                         EJECTA_2,
                                                         EJECTA_3,
                                                         LAYERS)

            valid_datalines.append(newdata + "\n")

        # if line is empty we count it and continue
        else:
            emptylines = emptylines + 1


        #sys.exit(0)



# double check line count
if not (linecount == (validlines + invalidlines + emptylines)):
    raise AssertionError("ERROR: (valid + invalid + empty) NOT equal to linecount")

if not (linecount == (ID_valid + 1)):
    raise AssertionError("ERROR: ID count NOT equal to linecount")

if not (linecount == (NAME_valid + NAME_missing + 1)):
    raise AssertionError("ERROR: NAME (valid + missing) NOT equal to linecount")

if not (linecount == (LATITUDE_valid + LATITUDE_missing + 1)):
    raise AssertionError("ERROR: LATITUDE (valid + missing) NOT equal to linecount")

if not (linecount == (LONGITUDE_valid + LONGITUDE_missing + 1)):
    raise AssertionError("ERROR: LONGITUDE (valid + missing) NOT equal to linecount")

if not (linecount == (DIAM_CIRCLE_valid + DIAM_CIRCLE_missing + 1)):
    raise AssertionError("ERROR: DIAM_CIRCLE (valid + missing) NOT equal to linecount")

if not (linecount == (DEPTH_RIMFLOOR_valid + DEPTH_RIMFLOOR_missing + 1)):
    raise AssertionError("ERROR: DEPTH_RIMFLOOR (valid + missing) NOT equal to linecount")

if not (linecount == (EJECTA_1_valid + EJECTA_1_missing + 1)):
    raise AssertionError("ERROR: EJECTA_1 (valid + missing) NOT equal to linecount")

if not (linecount == (EJECTA_2_valid + EJECTA_2_missing + 1)):
    raise AssertionError("ERROR: EJECTA_2 (valid + missing) NOT equal to linecount")

if not (linecount == (EJECTA_3_valid + EJECTA_3_missing + 1)):
    raise AssertionError("ERROR: EJECTA_3 (valid + missing) NOT equal to linecount")

if not (linecount == (LAYERS_valid + LAYERS_missing + 1)):
    raise AssertionError("ERROR: LAYERS (valid + missing) NOT equal to linecount")


# write new cleaned dataset to disk
new_dataset = 'marscrater_clean.csv'

with io.open(new_dataset, 'w') as wds:
    for line in valid_datalines:
        wds.write(line)


###############################################################################
#### REPORTING
###############################################################################

#print("Valid Lines: %s" % (validlines,))
#print("Invalid Lines: %s" % (invalidlines,))
#print("Empty Lines: %s" % (emptylines,))

tab1_headers = ["File", "Valid Lines", "Invalid Lines", "Empty Lines", "Total Lines"]
tab1 = [[dataset, validlines, invalidlines, emptylines, linecount]]

print("File Statistics - lines")
print(tabulate(tab1, headers=tab1_headers, tablefmt='grid'))
print()

#print("Valid DATA Lines: %s" % (len(valid_datalines),))
#print("Invalid DATA Lines: %s" % (len(invalid_datalines),))

tab2_headers = ["File", "Valid HEADER Lines", "Valid DATA Lines", "Invalid DATA Lines", "Total Lines"]
tab2 = [[dataset, 1, (len(valid_datalines)-1), len(invalid_datalines), linecount]]

print("File Statistics - DATA lines")
print(tabulate(tab2, headers=tab2_headers, tablefmt='grid'))
print()

#print("Valid IDs: %s" % (ID_valid,))

#print("Valid NAME: %s" % (NAME_valid,))
#print("MISSING NAME: %s" % (NAME_missing,))

#print("Valid LATITUDE: %s" % (LATITUDE_valid,))
#print("MISSING LATITUDE: %s" % (LATITUDE_missing,))

#print("Valid LONGITUDE: %s" % (LONGITUDE_valid,))
#print("MISSING LONGITUDE: %s" % (LONGITUDE_missing,))

#print("Valid DIAM_CIRCLE: %s" % (DIAM_CIRCLE_valid,))
#print("MISSING DIAM_CIRCLE: %s" % (DIAM_CIRCLE_missing,))

#print("Valid DEPTH_RIMFLOOR: %s" % (DEPTH_RIMFLOOR_valid,))
#print("MISSING DEPTH_RIMFLOOR: %s" % (DEPTH_RIMFLOOR_missing,))

#print("Valid EJECTA_1: %s" % (EJECTA_1_valid,))
#print("MISSING EJECTA_1: %s" % (EJECTA_1_missing,))

#print("Valid EJECTA_2: %s" % (EJECTA_2_valid,))
#print("MISSING EJECTA_2: %s" % (EJECTA_2_missing,))

#print("Valid EJECTA_3: %s" % (EJECTA_3_valid,))
#print("MISSING EJECTA_3: %s" % (EJECTA_3_missing,))

#print("Valid LAYERS: %s" % (LAYERS_valid,))
#print("MISSING LAYERS: %s" % (LAYERS_missing,))


tab3_headers = ["Variable", "Valid Values", "Missing Values", "Total Values"]
tab3 = [["CRATER_ID", ID_valid, 0, (linecount-1)],
        ["CRATER_NAME", NAME_valid, NAME_missing, (linecount-1)],
        ["LATITUDE_CIRCLE_IMAGE", LATITUDE_valid, LATITUDE_missing, (linecount-1)],
        ["LONGITUDE_CIRCLE_IMAGE", LONGITUDE_valid, LONGITUDE_missing, (linecount-1)],
        ["DIAM_CIRCLE_IMAGE", DIAM_CIRCLE_valid, DIAM_CIRCLE_missing, (linecount-1)],
        ["DEPTH_RIMFLOOR_TOPOG", DEPTH_RIMFLOOR_valid, DEPTH_RIMFLOOR_missing, (linecount-1)],
        ["MORPHOLOGY_EJECTA_1", EJECTA_1_valid, EJECTA_1_missing, (linecount-1)],
        ["MORPHOLOGY_EJECTA_2", EJECTA_2_valid, EJECTA_2_missing, (linecount-1)],
        ["MORPHOLOGY_EJECTA_3", EJECTA_3_valid, EJECTA_3_missing, (linecount-1)],
        ["NUMBER_LAYERS", LAYERS_valid, LAYERS_missing, (linecount-1)]
       ]

print("Data Statistics - Variables")
print(tabulate(tab3, headers=tab3_headers, tablefmt='grid'))
print()

sys.exit(0)
