# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:14:28 2022

@author: Pablo
"""

import numpy as np
import os
from mathfunctions import polynomial, log, exp, sin, power, expa, polforcurvefit
# change it later
from mathfunctions import polynomiallc, explc, loglc, sinlc, powerlc, expalc, const
from scipy.optimize import curve_fit
from inspect import signature
from scipy.optimize import differential_evolution
import warnings
# when adding new mathfunc create its own polifit.


def givetofunc(x, coeffs, function):
    """
    Getting x data, will return f(x) chosen the function and the coefficients, 
    giving error when not correct
    """
    # add new function have to add it here too
    y = np.zeros_like(x)
    if "pol" in function:

        y = polynomial(x, coeffs)

    elif "log" in function:
        if len(coeffs) != 2:
            raise ValueError("wrong coeffs")
        if np.any(x <= 0):
            raise ValueError("you can't log that")
        y = log(x, coeffs[0], coeffs[1])
    elif "expo" in function:
        if len(coeffs) != 3:
            raise ValueError("wrong coeffs")
        y = exp(x, coeffs[0], coeffs[1], coeffs[2])
    elif "sin" in function:
        if len(coeffs) != 4:
            raise ValueError("lacking coefficients")
        y = sin(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3])
    elif "pow" in function:
        if len(coeffs) != 3:
            raise ValueError("wrong coeffs")
        y = power(x, *coeffs)
    elif "expa" in function:
        if len(coeffs) != 3:
            raise ValueError("wrong coeffs")
        y = expa(x, *coeffs)
    return y


def addmeasureerror(data, RSE):
    """
    Add errors of measure given RSE is a relative standard error, expressed as decimal
    so e.g. 1% error is 0.01. So we'll take error = 0.01 = (x'-x)/x where x is the value of y
    x' is the value with noise. We'll suppose that error will be the relative standard deviation
    for a normal distribution that gives me the noise. So as  \sigma = RSE * x

    """
    size = data.shape
    sigma = abs(RSE * data)
    noise = np.random.normal(0, sigma, size)
    noised = data + noise
    return noised


def addnoise(data, sigma):
    """
    Add noise given noiselevel is the standard deviation of the noise.
    """
    size = data.shape
    noise = np.random.normal(0, sigma, size)
    noised = data + noise
    return noised
# dataaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-----------------


def writetocsv(data, filename, functionname, coeffs):
    """
    For testing and informative purposes we'll add how to get the original function, as a way to check
    wich data is wich
    """
    if type(filename) != str:
        raise ValueError("Not correct filename")
    string = " ".join(str(i) for i in coeffs)
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, "data", filename,)

    np.savetxt(filename+'.csv', data, delimiter=',',
               header=str(functionname) + string)   # data is an array

    return 1


def readcsv(filename):
    """
    Reads a csv file, discarding the first line (header), with a , delimiter
    """
    if type(filename) != str:
        raise ValueError("Not correct filename")
    try:
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, "data", filename,)
        data = np.loadtxt(filename + ".csv", skiprows=1, delimiter=",")
    except IOError:
        raise ValueError
    return data
# --------------------------------------------------------------------------------
# Fitting functiooons.


def polifit(x, y, polmax=10, knownerror=None, minstd=None, maxerrorrel=10e-5,
            mincoeff=1e-3):
    """
    Fit a function using curvefit to a polynomial. Returns the array of of popts and errors
    of the coefficients up to one of the limits is reached. Alsol returns the limit reached

    Parameters
    ----------
    polmax : int, maximum order of the polynomial  not for other functions
        DESCRIPTION. The default is 10.
    knownerror : if the error is known, pass array for the relative errors of the data.
    else pass none
        DESCRIPTION. The default is None.
    minstd : float, optional
        maximum standard deviation for the fitting (better not used). The default is None.
    maxerrorrel : float, optional
        Maximum relative error between the data and the fit.The default is 10e-5.
    mincoeff : float, optional
        Min popt allowed, for getting away with higuer orders that give no extra
        information. The default is 1e-3.

    Returns
    -------
    array of popts.
    array of erros (perr in curvefit).
    int Code of limit (0 max order reached), 1  min reached,
    2 maxerrorrel reached, 3 mincoeff reached 


    """
    perr = []
    popts = []
    for i in range(1, polmax+1):

        popt, pcov = curve_fit(polforcurvefit, x, y,
                               p0=np.ones(i), sigma=knownerror)
        perr.append(np.sqrt(np.diag(pcov)))
        popts.append(popt)
        # we move the graph to not have problems dividing by 0
        if np.any(y == 0):
            y2 = y - np.max(y) - 1
            if np.any(y2 == 0):
                y2 = y + 1 + np.min(y)
        else:
            y2 = y
        if minstd != None and np.std((y - polynomial(x, popt)) / y2) < minstd :
            return popts[-1], perr[-1], 1
        if maxerrorrel != None and np.all(abs(y - polynomial(x, popt)) / y2 < maxerrorrel):
            return popts[-1], perr[-1], 2
        for a in popt:
            if mincoeff != None and  abs(a) < mincoeff :
                return popts[-2], perr[-2], 3

    # now we choose the best fitting function:
    return popts, perr, 0


def logfit(x, y, knownerror=None, minstd=None, maxerrorrel=10e-5):
    """
    Same as polyfit but 0 is it's absolutely not a log
    """
    try:
        popt, pcov = curve_fit(log, x, y, sigma=knownerror)
    except RuntimeError:
        return None, None, 0
    perr = (np.sqrt(np.diag(pcov)))

    if np.any(y == 0):
        y2 = y - np.max(y) - 1
        if np.any(y2 == 0):
            y2 = y + 1 + np.min(y)
    else:
        y2 = y
    if minstd != None and np.std((y - log(x, *popt)) / y2) < minstd:
        return popt, perr, 1
    if maxerrorrel != None and np.all(abs(y - log(x, *popt)) / y2 < maxerrorrel):
        return popt, perr, 2
    else:
        return popt, perr, 3


def expfit(x, y, knownerror=None, minstd=None, maxerrorrel=10e-5):
    """
    Same as polyfit but 0 is it's absolutely not an exp
    """
    try:
        popt, pcov = curve_fit(exp, x, y, sigma=knownerror)
    except RuntimeError:
        return None, None, 0
    perr = (np.sqrt(np.diag(pcov)))

    if np.any(y == 0):
        y2 = y - np.max(y) - 1
        if np.any(y2 == 0):
            y2 = y + 1 + np.min(y)
    else:
        y2 = y
    if minstd != None and np.std((y - exp(x, *popt)) / y2) < minstd:
        return popt, perr, 1
    if maxerrorrel != None and np.all(abs(y - exp(x, *popt)) / y2 < maxerrorrel):
        return popt, perr, 2
    else:
        return popt, perr, 3


def trigfit(x, y, knownerror=None, minstd=None, maxerrorrel=10e-5):
    """
    Same as polyfit but 0 is it's absolutely not a sin
    """
    try:
        popt, pcov = curve_fit(sin, x, y, sigma=knownerror)
    except RuntimeError:
        return None, None, 0
    perr = (np.sqrt(np.diag(pcov)))

    if np.any(y == 0):
        y2 = y - np.max(y) - 1
        if np.any(y2 == 0):
            y2 = y + 1 + np.min(y)
    else:
        y2 = y
    if minstd != None and np.std((y - sin(x, *popt)) / y2) < minstd:
        return popt, perr, 1
    if maxerrorrel != None and np.all(abs(y - sin(x, *popt)) / y2 < maxerrorrel):
        return popt, perr, 2
    else:
        return popt, perr, 3


def powerfit(x, y, knownerror=None, minstd=None, maxerrorrel=10e-5):
    """
    Same as polyfit but 0 is it's absolutely not an power (x^a)
    """
    try:
        popt, pcov = curve_fit(power, x, y, sigma=knownerror)
    except RuntimeError:
        return None, None, 0
    perr = (np.sqrt(np.diag(pcov)))

    if np.any(y == 0):
        y2 = y - np.max(y) - 1
        if np.any(y2 == 0):
            y2 = y + 1 + np.min(y)
    else:
        y2 = y
    if minstd != None and np.std((y - power(x, *popt)) / y2) < minstd:
        return popt, perr, 1
    if maxerrorrel != None and np.all(abs(y - power(x, *popt)) / y2 < maxerrorrel):
        return popt, perr, 2
    else:
        return popt, perr, 3


def expafit(x, y, knownerror=None, minstd=None, maxerrorrel=10e-5):
    """
    Same as polyfit but 0 is it's absolutely not an expa (a^x)
    """
    try:
        popt, pcov = curve_fit(expa, x, y, sigma=knownerror)
    except RuntimeError:

        return None, None, 0
    perr = (np.sqrt(np.diag(pcov)))

    if np.any(y == 0):
        y2 = y - np.max(y) - 1
        if np.any(y2 == 0):
            y2 = y + 1 + np.min(y)
    else:
        y2 = y
    if minstd != None and np.std((y - expa(x, *popt)) / y2) < minstd:
        return popt, perr, 1
    if maxerrorrel != None and  np.all((y - expa(x, *popt)) / y2 < maxerrorrel):
        return popt, perr, 2
    else:
        return popt, perr, 3


def ffit(x, y, f, p0, knownerror=None, minstd=None, maxerrorrel=10e-5):
    """
    Function for a fit but general (like  polifit). Used for a combination of functions fitting
    """
    try:
        popt, pcov = curve_fit(f, x, y, p0=p0, sigma=knownerror)
    except RuntimeError:
        return None, None, 0
    perr = (np.sqrt(np.diag(pcov)))

    if np.any(y == 0):
        y2 = y - np.max(y) - 1
        if np.any(y2 == 0):
            y2 = y + 1 + np.min(y)
    else:
        y2 = y
    if minstd != None and np.std((y - f(x, *popt)) / y2) < minstd :
        return popt, perr, 1
    if maxerrorrel != None and np.all(abs(y - f(x, *popt)) / y2 < maxerrorrel):
        return popt, perr, 2
    else:
        return popt, perr, 3

    # now we choose the best fitting function:
    return popt, perr, 0


def comparelincombfits(x, y, funcnames, maxerrorrel=10e-5, minstd=None, knownerror=None,
                       mincoeff=1e-3, polmax=5, full_output=False):
    """
    Will make a possible lineal combination of 2 from every funcnames, using a 
    differential evolution to get possible initial parameters, and curvefit for the model
    fitting. 

    Parameters
    ----------
    x : xdata.
    y : ydata.
    funcnames : list
        list of names of all functions to try to fit.
    polmax : int, maximum order of the polynomial  not for other functions
        DESCRIPTION. The default is 10.
    knownerror : if the error is known, pass array for the relative errors of the data.
    else pass none
        DESCRIPTION. The default is None.
    minstd : float, optional
        maximum standard deviation for the fitting (better not used). The default is None.
    maxerrorrel : float, optional
        Maximum relative error between the data and the fit.The default is 10e-5.
    mincoeff : float, optional
        Min popt allowed, for getting away with higuer orders that give no extra
        information. The default is 1e-3.
    full_output : For all data from the fits

    Raises
    ------
    ValueError
        If the fit failed for some reason.

    Returns
    -------
    re, ind, index, fs[ind], errmd[ind]
    Where re is best fit coefficients, ind is the index in the array of data passed.
    index is the 2 functions relative to the funcnames passed, so for funcnames
    = [pol,log, trig] for the lineal combination {pol,log} index = (1,2)
    fs is the function created for the combination, errmd is the covariance.
    if full_output:
        returns results, codes, errmd, fs.
        results:all coefficient, resulting 
        codes for each fit call (see polifit),
        errmd of all of them and fs all functions created of the fit
    """
    results = []
    errstd = []
    codes = []
    errmd = []
    funclist = []
    names = []
    index = []
    fs = []
    print(funcnames)
    for a in funcnames:
        if "pol" in a and polynomiallc not in funclist:
            funclist.append(polynomiallc)

            names.append("polinomial")
        if "expo" in a and explc not in funclist:
            funclist.append(explc)
            names.append("exponential")
        if "log" in a and loglc not in funclist:
            funclist.append(loglc)
            names.append("logarithmic")
        if "trig" in a and sinlc not in funclist:
            funclist.append(sinlc)
            names.append("trigonometric")
        if "pow" in a and powerlc not in funclist:
            funclist.append(powerlc)
            names.append("power")
        if "expa" in a and expalc not in funclist:
            funclist.append(expalc)
            names.append("expa")

    for i in range(len(funclist)-1):
        for j in range(i, len(funclist)):
            warnings.filterwarnings("ignore")
            print(funcnames[i], funcnames[j])
            if funclist[i] == polynomiallc and funclist[j] == polynomiallc or (i, j) in index:
                continue  # it's not necessary to do that, comparefits should've given it
            if funclist[i] != polynomiallc and funclist[j] != polynomiallc:
                lf1 = len(signature(funclist[i]).parameters) - 1
                lf2 = len(signature(funclist[j]).parameters) - 1
                paramnumbers = lf1 + lf2

                def f(x, *args):

                    return funclist[i](x, *args[:lf1]) + funclist[j](x, *args[lf1:]) + const(x, args[-1])
                index.append((i, j))
                fs.append(f)

                def sumOfSquaredError(parameterTuple):
                    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
                    val = f(x, *parameterTuple)
                    return np.sum((y - val) ** 2.0)

                def generate_Initial_Parameters(paramnumbers):
                    # min and max used for bounds
                    maxX = max(x)
                    minX = min(x)
                    maxY = max(y)
                    minY = min(y)
                    maxXY = max(maxX, maxY, abs(minX), abs(minY))

                    parameterBounds = []
                    for i in range(paramnumbers):
                        parameterBounds.append([-maxXY, maxXY])

                    # "seed" the numpy random number generator for repeatable results
                    result = differential_evolution(
                        sumOfSquaredError, parameterBounds)
                    return result.x
                tries = 1
                err = None
                while err is None and tries < 6:
                    try:
                        geneticParameters = generate_Initial_Parameters(
                            paramnumbers)
                        #print (geneticParameters)
                        res, err, cod = ffit(x, y, f, geneticParameters, minstd=minstd,
                                             maxerrorrel=maxerrorrel, knownerror=knownerror)
                        print("tries", tries)
                    except ValueError:
                        print("errrrororororor")
                        res, err, cod = None, None, 0
                    tries += 1
                results.append(res)
                #print (err)
                errstd.append(err)
                codes.append(cod)

                if cod == 0:
                    errmd.append(np.nan)
                else:
                    errmd.append(np.linalg.norm(errstd[-1]))

            else:
                # now for a polynomial its much harder (we have to look at each possible order)
                # first we'll call the polynomial 1 and the other 2:
                if funclist[i] == polynomiallc:
                    pol = funclist[i]
                    func2 = funclist[j]

                else:
                    pol = funclist[j]
                    func2 = funclist[i]

                perr = []
                popts = []
                pcovs = []
                lf1 = len(signature(func2).parameters)
                z = 0
                suc = 0

                def comparison(f, popt):
                    if  minstd != None and np.std((y - f(x, *popt)) / y2) < minstd :
                        return 1, 1
                    if maxerrorrel != None and np.all(abs(y - f(x, *popt)) / y2 < maxerrorrel):
                        return 1, 2
                    for a in popt:
                        if mincoeff != None and  abs(a) < mincoeff :
                            return 1, 3
                    
                    return 0, 0
                for z in range(1, polmax+1):

                    paramnumbers = z + lf1

                    def f(x, *args):

                        return pol(x, *args[:z]) + func2(x, *args[z:-1]) + const(x, args[-1])

                    def sumOfSquaredError(parameterTuple):
                        # warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
                        val = f(x, *parameterTuple)
                        return np.sum((y - val) ** 2.0)

                    def generate_Initial_Parameters(paramnumbers):
                        # min and max used for bounds
                        maxX = max(x)
                        minX = min(x)
                        maxY = max(y)
                        minY = min(y)
                        maxXY = max(maxX, maxY, abs(minX), abs(minY))

                        parameterBounds = []
                        for i in range(paramnumbers):
                            parameterBounds.append([-maxXY, maxXY])

                        # "seed" the numpy random number generator for repeatable results
                        result = differential_evolution(
                            sumOfSquaredError, parameterBounds, seed=5)
                        return result.x

                    try:
                        geneticParameters = generate_Initial_Parameters(
                            paramnumbers)
                        popt, pcov = curve_fit(
                            f, x, y, p0=geneticParameters, sigma=knownerror)
                    except RuntimeError:
                        continue
                        continue
                    perr.append(np.sqrt(np.diag(pcov)))
                    popts.append(popt)
                    pcovs.append(pcov)
                    # we move the graph to not have problems dividing by 0
                    if np.any(y == 0):
                        y2 = y - np.max(y) - 1
                        if np.any(y2 == 0):
                            y2 = y + 1 + np.min(y)
                    else:
                        y2 = y
                    suc, code = comparison(f, popt)
                    if suc == 1:
                        if code == 3:
                            results.append(popts[-2])
                            errstd.append(pcovs[-2])
                            codes.append(code)
                        else:
                            results.append(popts[-1])
                            errstd.append(pcovs[-1])
                            codes.append(code)
                        fs.append(f)
                        break
                if suc == 0:
                    if (len(popts) != 0):
                        results.append(popts[-1])
                        errstd.append(pcovs[-1])

                    else:
                        results.append(None)
                        errstd.append(None)

                    codes.append(0)
                    fs.append(f)
                if codes[-1] == 0:
                    errmd.append(np.nan)
                else:
                    errmd.append(np.linalg.norm(errstd[-1]))
                if funclist[i] == polynomiallc:
                    index.append((i, j))

                else:
                    index.append((j, i))

    # now we have all the errors for the combinations of functions (compare)
    #print (errmd)
    try:
        ind = np.nanargmin(errmd)
    except ValueError:
        raise ValueError("no fit found")
    re = results[ind]

    if full_output:
        return re, ind, index, fs[ind], results, codes, errmd, fs
    else:
        return re, ind, index, fs[ind], errmd[ind]


def comparefits(x, y, funclist, maxerrorrel=10e-5, minstd=None, knownerror=None,
                mincoeff=1e-3, polmax=10, full_output=False):
    """
    Will make a fit given a list of functions using curvefit. Will then choose the best fit
    for the data assuming minimal square error

    Parameters
    ----------
    x : xdata.
    y : ydata.
    funclist : list
        list of  all functions to try to fit.
    polmax : int, maximum order of the polynomial  not for other functions
        DESCRIPTION. The default is 10.
    knownerror : if the error is known, pass array for the relative errors of the data.
    else pass none
        DESCRIPTION. The default is None.
    minstd : float, optional
        maximum standard deviation for the fitting (better not used). The default is None.
    maxerrorrel : float, optional
        Maximum relative error between the data and the fit.The default is 10e-5.
    mincoeff : float, optional
        Min popt allowed, for getting away with higuer orders that give no extra
        information. The default is 1e-3.
    full_output : For all data from the fits

    Raises
    ------
    ValueError
        If the fit failed for some reason.

    Returns
    -------
    returns re, index
    Where re is best fit coefficients, index is index of the function in the funclist list

    if full_output:
        returns results, errstd, codes, errmd:. 
        results all coefficients in order of funclist,
        errst the standard deviation of each result
        codes :resulting codes for each fit call (see polifit),
        errmd of all of them
    """
    results = []
    errstd = []
    codes = []
    errmd = []
    warnings.filterwarnings("ignore")
    for i in range(len(funclist)):
        if funclist[i] == polifit:
            res, err, cod = funclist[i](x, y, polmax=polmax,
                                        minstd=minstd, maxerrorrel=maxerrorrel,
                                        knownerror=knownerror, mincoeff=mincoeff)
        else:
            res, err, cod = funclist[i](x, y, minstd=minstd,
                                        maxerrorrel=maxerrorrel, knownerror=knownerror)
        results.append(res)
        errstd.append(err)
        codes.append(cod)
        if codes[i] == 0:
            errmd.append(np.nan)
        else:
            errmd.append(np.linalg.norm(errstd[i]))
    index = np.nanargmin(errmd)
    if type(results[index]) == list:
        re = results[index][-1]
    else:
        re = results[index]
    if full_output:

        return re, index, results, errstd, codes, errmd
    else:
        return re, index
