# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:02:26 2017

@author: insanesac
"""

import cv2
import numpy as np
import math
from PIL import Image

def entropy(img):
    """calculate the entropy of an image"""
    histogram = img.histogram()
    histogram_length = sum(histogram)
 
    samples_probability = [float(h) / histogram_length for h in histogram]
 
    ent = -sum([p * math.log(p, 2.0) for p in samples_probability if p != 0])
    
    return ent

def count(img):
    count = np.array(img.histogram())
    bin_loc = np.zeros(256)
    for i in range(0,256):    
     bin_loc[i] = i 
    return count, bin_loc.astype(int)

def skew(count,bin_loc):
    total = sum(count)
    mean = (sum(bin_loc*count))/total
    var = (sum(((bin_loc-mean)**2)*count))/(total-1.0)
    std = math.sqrt(var)
    skew = (sum(((bin_loc-mean)**3)*count))/((total-1.0)*(std**3))
    kurtosis = (sum(((bin_loc-mean)**4)*count))/((total-1.0)*(std**4))
    if np.isnan(skew):
        skew = 0;
        kurtosis = 0;
    return skew,kurtosis
    
def feature(img):    
    b, g, r  = cv2.split(img)

    r = Image.fromarray(r)
    g = Image.fromarray(g)
    b = Image.fromarray(b)

    enr = entropy(r)
    eng = entropy(g)
    enb = entropy(b)

    xr,yr = count(r)
    xg,yg = count(g)
    xb,yb = count(b)

    skr,kur = skew(xr,yr)
    skg,kug = skew(xg,yg)
    skb,kub = skew(xb,yb)

    mer = np.mean(r)
    meg = np.mean(g)
    meb = np.mean(b)

    stdr = np.std(r) 
    stdg = np.std(g)
    stdb = np.std(b)

    mat = np.array([mer, meg, meb, stdr, stdg, stdb, enr, eng, enb, skr, skg, skb, kur, kug, kub])

    return mat  