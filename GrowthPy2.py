# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 23:21:03 2014

@author: christopherrivera
"""
#these are functions to plot the growth data. 

from pandas import *
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sb
from os.path import join
from os import getcwd
from copy import copy
from scipy.stats import linregress

def Log(x):
    '''Calculates the natural log of every entry in an array.
    
    Args:
        x (numeric iterable):
    Returns:
        iterable.
    
    '''
    return [log(i) for i in x]

def movingRegression(x,y,N=17):
    '''Fits a moving linear regression to an x, y data series.
    
    Args:
            x (list or numpy.array): The domain of the data
            y (list or numpy.array): The range. 
            N (int): THe window for the moving regression. 
   Returns:
        A list of tuples (m,c,X,Y), where m is the slope, c is the y-intercept, X is midpoint, Y is the midpoint value'''    
    
    #Quality checks
    if N%2 ==0: 
        N = N+1
        print 'The number of points for the fit should be odd, an addioinal point was added.'

    if N <2: 
        print "There are not enough points for the fit."
        return
   
    elif N> len(x):
        print 'The number of points for the fit excess the number of points in the series.'
    n = (N-1)/2    #calculate the points on the sides. 
    
    #Get an interval to calculate the moving regression. 
    l = len(x) 
    L = range(n, l- n)   

    #Calculate the number of points to the right and left of the midpoint.    
    n = (N-1)/2   
    #fit moving linear regressions. 
    fits =   [linregress(x[i-n:i+n], y[i-n:i+n]) for i in  L]  
    return [(fits[i][0], fits[i][1], x[i+n], y[i+n]) for i in range(len(fits))]   #this returns a list of tupples (m,c,x-midpoint,y-midpoint)
    
def findInflection(x,y, N=17):
    '''Find the inflection point of an x, y series. 
    
    Args:
        x (numeric list or numpy.array): data for domain. 
        y (numeric list or numpy.array): data for range.
        N (int): window size for moving linear regression.
    Returns:
        a tupple (m,c,x,y) where m is the slope, c is the y intercept of the fit and x is the x cordinate, y is the y cordinate. .'''
    return max(movingRegression(x,y, N))   #get the inflection slope. 

def predict(m, c, xmin, xmax, stepsize = 0.1):
    '''Predicts a line.
    Args:
        m (float): slope
        c (float): y-intercept
        xmin (float): minimum x-value
        ymin (float): minimum y-value. 
    Returns:
        An X and Y numeric array. '''
   
   #Generate an array of X to predict on. 
    X = arange(xmin, xmax, stepsize)    
    #Calculate the prediction and return
    return (X,[c+m*x for x in X ])
    
    
def plotNormalizedSlope(x,y,N=17, xlabel = 'X', ylabel= 'Normalized Slope / Y'):
    '''Normalizes an x,y, estimates and normalizes a slope and plots the data. \n
    Args:
        x (numeric iterable). \n
        y (numeric iterable). \n
        N (int): window size. \n
        xlabel (str): X axis label. \n
        ylabel (str): Y axis label. \n
    Returns:
        Normalized data, and normalized slope. '''

    #Calculate a moving linear Regression. 
    fits = movingRegression(x,y,N)
    
    #From the fits, get the x interval and the normalized slopes
     #x interval for the slopes
    x_slopes = [f[2] for f in fits]  
    #get thenormalized slopes
    slopes =  [f[0] for f in fits]            
            
    #Get the coordinates for the Inflection point to plot.  
    inflectionPoint= max(fits)              #get the tupple for the inflection point.
    #get the x coordinate for the inflection point and normalize it. 
    inflection_x  = inflectionPoint[2]  
    #get a normalized slope at the infleciton point.             
    inflection_slope_y = (inflectionPoint[0] - min(slopes))/(max(slopes)-min(slopes))
    #get the normalized y at the inflection point           
    inflection_y  = (y[list(x).index(inflection_x)] - min(y))/(max(y) -min(y))    
    
    #Plot the Data
    plt.plot(x_slopes, normalize(slopes), x, normalize(y))  
    plt.plot(inflection_x,inflection_y,'g.', inflection_x,inflection_slope_y, 'b.', markersize= 10)
    
    #modify the plot
    ax = plt.gca()   #get the current axes so can modify
    ax.set_ylim(0,1.1)   #set the  lines
    ax.set_xlabel(xlabel, size=15)
    ax.set_ylabel(ylabel, size =15)
    plt.legend(('Normalized Slope', 'Normalized Y'), fontsize = 14)
    
def plotInflecionSlopeOnGraph(x,y,N= 17, xlabel = 'X', ylabel = 'Y', legend='Y versus X', stepsize = 0.1):
    '''Plots a graph with the inflection point and the slope throught the inflection point'''
    
    #get the inflection point. 
    I = findInflection(x,y,N)
    Sx, Sy = predict(m = I[0], c= I[1],xmin= 0,xmax =  max(x), stepsize=stepsize)   #get the corrdinates for the max line. 
    
    #plot the data
    plt.plot(x,y, Sx,Sy)
    plt.plot(I[2], I[3], '.', markersize=20) #Plot the inflecion point
    #modify the plot
    ax = plt.gca()   #get the current axes so can modify
    ax.set_xlabel(xlabel, size=15)
    ax.set_ylabel(ylabel, size =15)
    plt.legend((legend, 'Slope Through Inflection Point'), fontsize = 14)
    
def normalize(x):
    '''Nomalize an array s.t. z = (x-min(x))/(max(x)-min(x). \n
    Args: 
        x (numeric iterable): series
    
    Returns:
        a normalized series. 
    
    '''
    return (x-min(x))/(max(x)-min(x))

def movingaverage(interval, window_size):
    '''Returns a moving average for an array
    Args:
        interval (list/array): 
        window_size (int): The size of the window.'''
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')


def zero(x, N=5):
    ''' Zeros a series by subtracting the mean of the smallest N points from all points
    Args
        x (iterable): numeric list or numpy.array
        N (int): number of points to include for average        
    Returns
        A zeroed array/list'''
    #calculte the mean for the N smallest points and subtract it from the list
    return x- mean(sorted(x)[:N])


def rollingMeanDf(df, window=5):
    '''Calculates a rolling mean for each column vector in a data frame using apply method This computes a moving 
    average assuming with center ==True.
    Args:
        df (pandas.DataFrame): DataFrame containg numeric vectors
        window (int): Window size
        
    Returns: 
        A dataframe contianing the moving averages. '''
    min_periods = None
    freq = None
    center =True
    return df.apply(rolling_mean, axis = 0, args=[ window,min_periods, freq, center] )
              
class Growth(object):
    '''Import and formats data from a Tecan Plate reader experiment for plotting and analysis
    
    Attributes:
        time (numpy array): Contains the time data. \n
        intensity (pandas DataFrame): Contains the Intesnity Data \n  
        MetaData (pandas Dataframe): Initially a string, but upon creation by the getMetaData function, becomes a Data frame that holds the metadata.  \n
    
    '''
    def __init__(self,fin,meta ='', Minutes = True):
        '''Read in and parses a Tecan Excel file, and generates an Growth Object.
        
        Args:
            fin (str): Absolute path or handle to the excel file. 
            meta (str): If not equal to '', Path to to the MetaData excel file. 
            minutes (bool): If True, the time data is converted from seconds to minutes. 
    
        Returns: 
            A Growth Object'''
        
        self.data  = read_excel(fin)
        
        #create an index object from the first column to search. 
        self.index = Index(self.data.iloc[:,0])
        
        #Get the location of the first and final row for the data. 
        self.start = self.index.get_loc('Time [s]')    #THe first row begins with tim
        
        #get the time
        self.time = array(self.data.iloc[self.start, 1:])
        self.Time = 'Time (sec)'       #a label for the plotting
        #if Minutes = True, convert the time (which is in seconds to minutes)
        if Minutes == True:
            self.time = self.time/60.0
            self.Time = 'Time (min)'
        
        #Get the data (assume that the data is the last row)
        self.intensity = self.data.iloc[self.start+2 : -4, 1:]    #get the intensity information
        self.intensity = self.intensity.transpose()               #transpose the data
        self.intensity.columns = self.data.iloc[self.start+2 : -4, 0]  #rename the columns
        self.intensity.columns.name = 'Sample Well'        
        self.intensity.index = arange(size(self.intensity.index,0))    #rename the index(rows)
        if meta=='':
            self.MetaData =''
        else: 
            self.getMetaData(meta)
        self.estimateGrowthParameters()
        
    def estimateMaxRate(self):
        '''Estimates the maximum growth rates by fitting a line exponential (linear part) of a growth curve. This function picks points automatically ranging from the the points greater than 0.5 of the floor and 0.95 of the plateau.
        Args:
            None
        Returns:
            None
        '''
        
        #Calculate the Log of the data
        self.Log  = self.intensity.apply(Log)
        #calculate deltaa between the floor and the plateau
        delta = self.plateau-self.floor
        
        #calculate a lower and upper threshold. 
        lower_threshold = 0.2*delta + self.floor
        upper_threshold = 0.8*delta + self.floor
        
        #get the points that are below it 
        
   
    def estimateGrowthParameters(self, window=5):
        '''Estimates the log time, maximal growth rate and the plateau/ Carrying capacity from data
        Args:
            window (int): The window size to be used for the rolling mean.
        
        Returns:
            A tuple (tlag, uMax, floor,plateau)
            '''
        #Compute a rolling Mean     
        self.rolling = rollingMeanDf(self.intensity,window=5)
        
        #compute the floor  and floor inex
        self.floor = self.rolling.min(axis=0)
        self.floor_index= self.rolling.idxmin(axis=0)
        
        #compute the plateau and the plateau index
        self.plateau = self.rolling.max(axis=0)
        self.plateau_index = self.rolling.idxmax(axis=0)
        
        #compute the maximal growth ratee for each column
        self.estimateMaxRate()
        
    def plot(self, save =False, path = '', logY=False):
        #Plots the intensity data. 
        self.intensity.plot(x=self.time, grid = False, logy=logY)
        plt.xlabel(self.Time, size = 14)
        plt.ylabel('OD (600 nM)',size=14)
        
        #plt.show()
        #If save ==True: plot 
        if save ==True:
            plt.savefig(path)
        
    def getMetaData(self,meta):
        '''Loads and parses meta data from a excel file
        
        Args
            meta (str): Path to the meta data'''
        self.MetaData = read_excel(meta)
    
    def subset(self,Value='', Parameter = 'Condition' ):
        '''Subset on a given value and parameter return a new growth object.
        
        Args:
            Value (string): The name of the value to subset on.
            Parameter (string): The name of column header to subset on.
        Return:
            Growth object'''
        
        if type(self.MetaData) ==str:
            print 'There is no Meta Data present for subsetting. '
        elif Value =='':
            print 'No valid value for subsetting has been set'
        else:
            i = self.MetaData.columns.get_loc(Parameter)  #get the columm
            Index = self.MetaData.iloc[:,i]
            Index= [j for j in range(len(Index)) if Index[j] ==Value]

            NewGrowth = copy(self)
            NewGrowth.intensity = NewGrowth.intensity.iloc[:, Index]
            NewGrowth.MetaData = NewGrowth.MetaData.iloc[Index, :]
            
            return NewGrowth
    
    def zero(self, N=5, zero = False):
        '''Zero the intensity data by the first N points. if zero == True,then it converts all negative numbers to zero. '''
        self.intensity -= self.intensity.iloc[0:N, :].mean(axis=0) #take the average the first N points and subtract the average from the original data
        if zero ==True:
            self.intensity[self.intensity<0] =0
        
if __name__=='__main__':
    fin = join(getcwd(),'Example.xlsx')
    meta = join(getcwd(), 'MetaData.xlsx')
    
    g = Growth(fin, meta ,Minutes= True)
    x=g.intensity.iloc[:,0]
    

    plotNormalizedSlope(g.time,x)


    
   
    