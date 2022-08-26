# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:57:09 2022

@author: Pablo
"""


import numpy as np

# for 1 fun


def polynomial(x, coeffs):
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i]*(x**i)
    return y


def log(x, a, c):
    try:
        a * np.log(x) + c
    except RuntimeWarning:
        raise RuntimeError
    return a * np.log(x) + c


def exp(x, a, b, c):
    return a * np.exp(b*x) + c


def sin(x, a, b, c, d):
    return a * np.sin(b*x + c) + d


def power(x, a, b, c):
    return a * x**b + c


def expa(x, a, b, c):
    return a * b**x + c


def polforcurvefit(*args):
    x = args[0]
    i = 0
    y = 0
    for a in args[1:]:
        y += a*(x**i)
        i += 1
    return y

# for lincombs


def polynomiallc(*args):
    x = args[0]
    i = 1
    y = 0
    for a in args[1:]:
        y += a*(x**i)
        i += 1
    return y


def loglc(x, a):
    try:
        a * np.log(x)
    except RuntimeWarning:
        raise RuntimeError
    return a * np.log(x)


def explc(x, a, b):
    return a * np.exp(b*x)


def sinlc(x, a, b, c):
    return a * np.sin(b*x + c)


def powerlc(x, a, b):
    return a * x**b


def expalc(x, a, b):
    return a * b**x


def const(x, a):
    return a
