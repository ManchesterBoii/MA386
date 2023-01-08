# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:49:15 2020

@author: Andrew
"""

def column(matrix, col):
    '''
    Inputs
        matrix: nested list of data
        column: number indicating which entry on each sublist we want data from
    Outputs:
        list containing entry in each row of the matrix from specified col
    '''
    data = []
    for i in range(len(matrix)):
        data.append(matrix[i][col])
    return data

def readcsv(file_name, numHeader = 1, dtFormat = "%m-%d-%Y"):
    
    """  
    readcsv() is a function that reads data from a csv file and assigns
    it to a data structure.
    Inputs: fileName - is a string without the .csv (which is assumed);
            numHeader - is an integer that gives the number of lines of headers
                     before the data starts;
            dtFormat - is either not assigned if there are no dates
                     in the data or a date format such as '%m/%d/%Y' as an
                     example of a date format.  The date must also be in the
                     first column for this function to work properly.
    Outputs: head - header lines as one large text
             data - the data from the columns that comes after the header
             
    Format date properly source: https://www.w3schools.com/python/python_datetime.asp
    """
    import datetime as dt
    infile = open(file_name + ".csv", "r")
    header = []
    for i in range(numHeader):
        h = infile.readline().split(',')
        h[-1] = h[-1].strip()
        header.extend(h)
    data = []
    for line in infile:
        row = line.split(',')
        row[-1] = row[-1].strip()
        for i,e in enumerate(row):
            try:
                row[i] = dt.datetime.strptime(e, dtFormat)
            except:
                try:
                    row[i] = int(row[i])
                except:
                    try:
                        row[i] = float(row[i])
                    except:
                        pass
        data.append(row)
    for i in range(len(data)):
        try:
            data[i][0] = data[i][0].strftime(dtFormat)
        except:
            pass
    infile.close()
    return header, data

def MA(ts,N):
    """
    MA(time_series,N) is a function that calculates the moving average of N
    days and returns a series with N-1 less elements than the original
    Inputs: time_series - a series of numbers that represent some measurement
            over the course of n discrete periods of time (days, minutes, ...)
            N - the number of elements to average

    For example say the series 1,5,3,2,6,5,4 was the series and N = 3,
    then (1+5+3)/3 = 3, (5+3+2)/3 = 3.333, (3+2+6)/3 = 3.667, (2+6+5)/3=4.333,
    (6+5+4)/3 = 5 => {3.000, 3.333, 3.667, 4.333, 5.000} is the output series.
    """
    m = len(ts)
    ma = []
    if N > m:
        return "Not possible"
    for i in range(m-N+1):
        ma.append(sum(ts[i:i+N])/N)
    return ma

def writecsv(header, data, csvName, decimals, fmt = '.csv'):
    """
    writecsv() writes a set of data and header to a csv file with specific
    column widths - most of the time the col_width = 0

    Inputs: header - list containing column header text that goes at the top of the file
            data - nested list of the data that will be written to the file
            csvName - the name of the text file where the data is written
            decimals - the number of decimals for the data being written

    Outputs: nothing is returned to the program when this function is called
            csvName.csv is created with the header and data in it.
    """
    import numpy as np
    outfile = open(csvName + fmt, 'w')
    for i in range(len(header)):
        outfile.write(header[i])
        if i != len(header) - 1:
            outfile.write(',')
        else:
            outfile.write('\n')
    fmt = '.' + str(decimals) + 'f'
    for i in range(len(data)):
        if isinstance(data[i], str):
            outfile.write(data[i] + '\n')
            continue
        for j in range(len(data[i])):
            if isinstance(data[i][j], float):
                outfile.write(f"{data[i][j]:{fmt}}")
            else:
                outfile.write(data[i][j])
            if j != len(data[i]) - 1:
                outfile.write(',')
            else:
                outfile.write('\n')
    outfile.close()