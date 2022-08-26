# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:52:18 2022

@author: Pablo
"""

import numpy as np
from functions import addmeasureerror, writetocsv, givetofunc, addnoise
import re
import matplotlib.pyplot as plt
xrange = [1, 5]
N = 10000
coeffse = [2, 3]  # for exp log
coeffsp = [2, 5, 4, 6, 7]  # for polynomial
order = len(coeffsp) - 1
# name new function have to name it here and in givetofunc

error = .01


def functioninputs():
    """
    Interacting with the user, will take inputs (as text) of the functions to create
    and assign them to a function f, getting also the necessary coefficients

    Returns
    -------
    function : string name of function chosen
        DESCRIPTION.
    coeffs : list
        list of the coefficients for the function.

    """
    while True:
        # if you add a new mathfunction have to name it here
        functionp = "polynomial"
        functione = "expo"
        functionl = "log"
        functiont = "sin"
        functionpw = "pow"
        functionea = "expa"
        functionname = input("what type of function do you want to create:  ")
        if "pol" in functionname.lower():
            function = functionp
            string = input(
                "write the coefficients separated by commas from a_0 to a_n:  ")
            res = re.split(", |,", string.strip('][')) #to separate them with , and if given a list

            coeffs = [float(x) for x in res]
            break
        elif "expa" in functionname.lower():
            function = functionea
            string = input(
                "write a and b and c (a * b**x + c) separated by commas:  ")
            res = re.split(", |,", string.strip(']['))
            coeffs = [float(x) for x in res]
            break
        elif "expo" in functionname.lower():
            function = functione
            string = input(
                "write a and b and c (ae^bx + c) separated by commas:  ")
            res = re.split(", |,", string.strip(']['))
            coeffs = [float(x) for x in res]
            break
        elif "log" in functionname.lower():
            function = functionl
            string = input("write a and b (aln(bx)+ c) separated by commas:  ")
            print("the log is equivalent to")
            res = re.split(", |,", string.strip(']['))
            coeffs1 = [float(x) for x in res]
            # a ln(bx)+ c = a (ln b + ln x)+ c = alnx + aln b * c = aln x + C
            coeffs = [0, 0]
            coeffs[0] = coeffs1[0]
            coeffs[1] = coeffs1[0] * np.log(coeffs1[1]) * coeffs1[2]
            print(str(coeffs[0])+" * ln(x) + " + str(coeffs[1]))
            break

        elif "sin" in functionname.lower():
            function = functiont
            string = input(
                "write a and b and c and d (asin(bx + c) + d) separated by commas:  ")
            res = re.split(", |,", string.strip(']['))
            coeffs = [float(x) for x in res]
            break
        elif "pow" in functionname.lower():
            function = functionpw
            string = input(
                "write a and b and c (a * x**b + c) separated by commas:  ")
            res = re.split(", |,", string.strip(']['))
            coeffs = [float(x) for x in res]
            break

    return function, coeffs


filename = input(
    "welcome to the creator of data, please enter the filename you want to create (without.csv added):  ")

i = input("Choose 1 (1) or 2 (2) functions f = f1 + f2:  ")
if "1" in i:
    f1, coeffs1 = functioninputs()
if "2" in i:
    print("for f1")
    f1, coeffs1 = functioninputs()
    print("for f2")
    f2, coeffs2 = functioninputs()

string = input("write the range of x separated by commas:  ")
res = re.split(", |,", string.strip(']['))
xrange = [float(x) for x in res]
N = int(input("Number of points to plot:  "))


x = np.linspace(xrange[0], xrange[1], N)
if "1" in i:
    print(coeffs1, f1)
    y = givetofunc(x, coeffs1, f1) #calculate the y values for data
elif "2" in i:
    print(coeffs1, f1)
    print(coeffs2, f2)
    y = givetofunc(x, coeffs1, f1) + givetofunc(x, coeffs2, f2)
else:
    raise ValueError("please select a number")
while True:
    e = int(input("Do you want to add random noise from the environment (1) or error in the measure apparatus? (2):  "))
    if e == 2: #error in the measure
        error = float(
            input("What will be the mean error (in decimal) ex. 5% write 0.05:  "))
        ye = addmeasureerror(y, error)
        break
    elif e == 1: #completely random noise
        error = float(
            input("What will be the noise (in std) ex. write 0.05:  "))
        ye = addnoise(y, error)
        break
data = np.vstack((x, ye)).T
if "1" in i: #we write to a csv file
    writetocsv(data, filename, f1, coeffs1)
    print("Data created")
if "2" in i:
    listaf = [f1, f2]
    listaco = [coeffs1, coeffs2]
    writetocsv(data, filename, listaf, listaco)
    print("Data created")
    
plt.plot (x,ye)