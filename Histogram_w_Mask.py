# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 00:16:58 2021

@author: hbaldwin
"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal  
#import sklearn
#from sklearn.metrics import mean_squared_error 
import seaborn as sns; sns.set(color_codes=True)
#import scipy.stats
#from scipy.stats import *
#import math
#import statistics
#import matplotlib as mpl
from scipy.stats import kurtosis
from scipy.stats import skew
from numpy import int64

#import raster dataset
input_file = r"C:\Users\hbaldwin\Desktop\FSH\Input_for_Histograms\6ha\InSAR_2009.tif"
input_open = gdal.Open(input_file)
input = input_open.ReadAsArray()

#clean
input_rav = np.array(input).ravel()

input_nan = np.where(input_rav<0, np.NaN, input_rav)
input_nan_v2 = np.where(input_nan>245, np.NaN, input_nan)

input_bad = ~np.logical_or(np.isnan(input_nan), np.isnan(input_nan_v2))
#input_bad = ~np.isnan(input_nan)

input_clean = np.compress(input_bad, input_nan_v2)
print(input_clean)

#get basic stats
input_max = input_clean.max()
input_min = input_clean.min()
input_std = input_clean.std()
input_median = np.median(input_clean)
input_count = len(input_clean)

#get frequencies to make histogram
frequency = {}
frequency['<2'] = 0
frequency['<4'] = 0
frequency['<6'] = 0
frequency['<8'] = 0
frequency['<10'] = 0
frequency['<12'] = 0
frequency['<14'] = 0
frequency['<16'] = 0
frequency['<18'] = 0
frequency['<20'] = 0
frequency['<22'] = 0
frequency['<24'] = 0
frequency['<26'] = 0
frequency['<28'] = 0
frequency['<30'] = 0
frequency['<32'] = 0
frequency['<34'] = 0
frequency['<36'] = 0
frequency['<38'] = 0
frequency['<40'] = 0
frequency['<42'] = 0
frequency['<44'] = 0
frequency['<46'] = 0
frequency['>46'] = 0
#frequency['<48'] = 0
#frequency['<50'] = 0
#frequency['>50'] = 0

for val in input_clean:
    if val <= 2:
        frequency['<2'] += 1
    elif val <= 4:
        frequency['<4'] += 1
    elif val <= 6:
        frequency['<6'] += 1
    elif val <= 8:
        frequency['<8'] += 1
    elif val <= 10:
        frequency['<10'] += 1
    elif val <= 12:
        frequency['<12'] += 1
    elif val <= 14:
        frequency['<14'] += 1
    elif val <= 16:
        frequency['<16'] += 1
    elif val <= 18:
        frequency['<18'] += 1
    elif val <= 20:
        frequency['<20'] += 1
    elif val <= 22:
        frequency['<22'] += 1
    elif val <= 24:
        frequency['<24'] += 1
    elif val <= 26:
        frequency['<26'] += 1
    elif val <= 28:
        frequency['<28'] += 1
    elif val <= 30:
        frequency['<30'] += 1
    elif val <= 32:
        frequency['<32'] += 1
    elif val <= 34:
        frequency['<34'] += 1
    elif val <= 36:
        frequency['<36'] += 1
    elif val <= 38:
        frequency['<38'] += 1
    elif val <= 40:
        frequency['<40'] += 1
    elif val <= 42:
        frequency['<42'] += 1
    elif val <= 44:
        frequency['<44'] += 1
    elif val <= 46:
        frequency['<46'] += 1
    else:
        frequency['>46'] += 1
#    elif val <= 48:
#        frequency['<48'] += 1
#    elif val <= 50:
#        frequency['<50'] += 1
#    else:
#        frequency['>50'] += 1

#make frequency into a list for input into barchart                   
freq_y = [int64(frequency['<2']), frequency['<4'], frequency['<6'], frequency['<8'], frequency['<10'],
     frequency['<12'], frequency['<14'], frequency['<16'], frequency['<18'], 
     frequency['<20'], frequency['<22'], frequency['<24'], frequency['<26'], 
     frequency['<28'], frequency['<30'], frequency['<32'], frequency['<34'], 
     frequency['<36'], frequency['<38'], frequency['<40'], frequency['<42'], 
     frequency['<44'], frequency['<46'], frequency['>46']]
     #frequency['<48'], frequency['<50'], 
     #frequency['>50']]

#convert to area to make the different resolutions comparable

#area for GEDI (25m diameter circle)
#area = np.array(freq_y)*3.14*(25/2)

#area for LiDAR-based FSH 25m
#area = np.array(freq_y)*25*25

#area for SAR-based estimates (bs, insar, fusion)
area = np.array(freq_y)*245*245
#set area type to accommodate large numbers and avoid errors
area = area.astype('int64')

#area for GLAD product (30m resolution)
#area = np.array(freq_y)*30*30
print(freq_y[0])
print(area[0])     
#area[0] = freq_y[0]*30*30    
    
#create masks to plot by color

#reference: https://stackoverflow.com/questions/33476401/color-matplotlib-bar-chart-based-on-value/63374092#63374092?newreg=1ad2f5619c4b4ca1a8a82f6690398366
col = []

x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 
     40, 42, 44, 46, 48]#, 50, 52] 

#total = input_nan.count()
for val in x:
    if val <= 5:
        col.append('#ffffcc')
    elif val <= 7:
        col.append('#c2e699')
    elif val <= 10:
        col.append('#78c679')
    elif val <= 12:
        col.append('#31a354')
    else:
        col.append('#006837')

#reference https://www.statology.org/skewness-kurtosis-python/
#The kurtosis of a normal distribution is 3.
#If a given distribution has a kurtosis less than 3, it is said to be playkurtic, which means it tends to produce fewer and less extreme outliers than the normal distribution.
#If a given distribution has a kurtosis greater than 3, it is said to be leptokurtic, which means it tends to produce more outliers than the normal distribution.

kurtosis = str(round(kurtosis(area), 2))
skew = str(round(skew(area), 2))

str(skew)

#Adding text inside a rectangular box by using the keyword 'bbox'
#plt.text(0.5, 0.0, "Skewness = " + skew)

#get basic stats
input_max = str(round(input_clean.max()))
input_min = str(round(input_clean.min()))
input_std = str(round(input_clean.std()))
input_median = str(round(np.median(input_clean)))
input_count = str(round(len(input_clean)))

# these are matplotlib.patch.Patch properties
textstr = "skewness = " + skew + "\n" + "kurtosis = " + kurtosis + "\n" + "max = " + input_max + "\n" + "median = " + input_median + "\n" + "min = " + input_min + "\n" + "std dev = " + input_std + "\n" + "count = " + input_count  

plt.text(35, 460000000, textstr, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.xlabel("Height (m)", fontsize = 10)
plt.ylabel("Area (m2)",fontsize = 10)
plt.bar(x, area, color = col, width=2)
plt.show()


plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':250})