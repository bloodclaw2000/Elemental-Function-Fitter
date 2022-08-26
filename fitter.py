# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:48:58 2022

@author: Pablo
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time as time


def functioncheckers(funcstring):
    """
    adds all the functions that happen in a string to both a list of the fits of the function
    and a list of string names of them
    """
    funclist = []
    names = []
    # add here to add new mathfunctions
    if "pol" in funcstring and f.polifit not in funclist:
        funclist.append(f.polifit)
        names.append("polinomial")
    if "expo" in funcstring and f.expfit not in funclist:
        funclist.append(f.expfit)
        names.append("exponential")
    if "log" in funcstring and f.logfit not in funclist:
        funclist.append(f.logfit)
        names.append("logarithmic")
    if "trig" in funcstring and f.trigfit not in funclist:
        funclist.append(f.trigfit)
        names.append("trigonometric")
    if "pow" in funcstring and f.powerfit not in funclist:
        funclist.append(f.powerfit)
        names.append("power")
    if "expa" in funcstring and f.expa not in funclist:
        funclist.append(f.expafit)
        names.append("expa")
    if not funclist:
        print("please write again the names of the functions")
        return funclist, names, 0
    return funclist, names, 1
#


def sendtocmd():
    """
    using console, get a list of the parameters needed to do the fit, having also defaults
    returns   #[ maxerrorrel, minstd, knownerr,mincoeff, polmax, full_output]
     ["maximum relative error", "minimal standard deviation",
                "known error", "minimal coefficient"]
    """
    #possible args (for console mainly)
   
    arglist = []
    argnames = ["maximum relative error", "minimal standard deviation",
                "known error", "minimal coefficient"]
    defaults = [10e-5, None, None, 1e-3, 10]
    for i in range(len(argnames)):
        while True:
            tmp = input("please provide the " +
                        argnames[i] + " or write none or d for default (" + str(defaults[i]) + "):   ")
            try:
                tmp = float(tmp)
                arglist.append(tmp)
                break
            except ValueError:
                if "d" in tmp.lower():
                    tmp = defaults[i]
                    arglist.append(tmp)
                    break
                elif "no" in tmp.lower():
                    tmp = 0
                    arglist.append(tmp)
                    break
    tmp = input(
        "please provide the maximum order (for polinomial fitting or write none or d for default (" + str(defaults[-1]) + "):   ")
    try:
        tmp = int(tmp)
        arglist.append(tmp)
    except ValueError:
        arglist.append(defaults[i+1])

    while True:
        tmp = input("Do you want the full data of the fitting? (y/n):  ")
        if "y" in tmp:
            arglist.append(True)
            break
        elif "n" in tmp:
            arglist.append(False)
            break
    return arglist


""" First check wether arguments are provided (remember, 0 is program, 1 is file of values to read, 
3 string of funcs "polinomial, exponential, logarithmic, trigonometric..."", for more in options in readme)"""

while True:
    database = input(
        "please enter the filename you want to load (without.csv added):  ")
    try:

        data = f.readcsv(database)
        x, y = np.split(data, 2, axis=1)
        x = x.flatten()
        y = y.flatten()
        break
    except ValueError:
        print("not valid filename")
        continue
i = input("Choose 1 (1) or 2 (2) functions to fit f = f1 + f2:  ")
if "2" in i:
    nfun = 2
else:
    nfun = 1
retorn = 0
while retorn == 0:
    print("possible funcs polinomial, exponential, logarithmic, trigonometric, power, expa")
    funcstring = input("please enter the funcs to choose the fit from:  ")
    funclist, names, retorn = functioncheckers(funcstring)

arglist = sendtocmd()
# we start the fitting

# for 1 function
tic = time.perf_counter()
if nfun == 1:
    if arglist[-1]:  # if full_output
        results, ind, allcoeff, errstds, codes, errmds = f.comparefits(
            x, y, funclist, *arglist)
        toc = time.perf_counter()
        print("full output---")
        print("winner func", names[ind])
        print(results)
        print("all other coeffs and data (in order of typed):")
        for i in range(len(allcoeff)):
            print("coeffs for", names[i])
            print(allcoeff[i])
            print("error std")
            print(errstds[i])
            print("code", codes[i])
            print("errmd")
            print(errmds[i])
            print("-------------------------------------")

    else:  # results is the winning coefficient, and ind will give us what index is it from funclist
        results, ind = f.comparefits(x, y, funclist, *arglist)
        print("We found that it is most likely that the function is a",
              names[ind])
        print("the coeffs are", results)
        
        toc = time.perf_counter()
# for 2 functions
elif nfun == 2:
    if arglist[-1]:  # if full_output
        results, ind, index, winnfunc, allcoeff, codes, errmds, fs = f.comparelincombfits(
            x, y, names, *arglist)
        toc = time.perf_counter()
        print("full output---")
        print("winner func", names[index[ind][0]], names[index[ind][1]])
        print(results)
        print("all other coeffs and data (in order of typed):")
        for i in range(len(index)):
            print("coeffs for", names[index[i][0]], names[index[i][1]])
            print(allcoeff[i])
            print("code", codes[i])
            print("errmd")
            print(errmds[i])
            print("-------------------------------------")

    else:  # results is the winning coefficient, and ind will give us what index is it from funclist
        results, ind, index, winnfunc, errmd = f.comparelincombfits(
            x, y, names, *arglist)
        print("We found that it is most likely that the function is a",
              names[index[ind][0]], names[index[ind][1]])
        print("the coeffs are", results)
        print("the correlation between variables is", errmd)
    toc = time.perf_counter()

print("Total time for fitting %.4f" % (toc-tic), "seconds ")
if nfun == 1:
    plt.figure()
    plt.plot(x, y, label="data")

    if arglist[-1]:
        for i in range(len(allcoeff)):
            if codes[i] != 0:
                plt.plot(x, f.givetofunc(
                    x, allcoeff[i], names[i]), label="fitted " + names[i])
    else:
        plt.plot(x, f.givetofunc(
            x, results, names[ind]), label="fitted " + names[ind])
    plt.legend()
    plt.show()


elif nfun == 2:
    plt.figure()
    plt.plot(x, y, label="data")
    plt.plot(x, winnfunc(x, *results), label="fitted " +
             names[index[ind][0]] + " " + names[index[ind][1]])
    plt.legend()
    plt.show()
# if full_out
