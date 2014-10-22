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
        A tupple (m,c,x,y) where m is the slope, c is the y intercept of the fit and x is the x cordinate, y is the y cordinate. .'''
    return max(movingRegression(x,y, N))   #get the inflection slope. 

def predict(m, c, xmin, xmax, stepsize = 0.1):
    '''Predicts a line on an interval given parameters for slope and y-intercept.
    
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
    
def predict2(m, c, X):
    '''Predicts a line with a user provided array.
    
    Args:
        m (float): slope
        c (float): y-intercept
        X (numeric itrable)

    Returns:
        A predicted Y numeric array. '''

    #Calculate the prediction and return it
    return [c+m*x for x in X ]
    
def plotNormalizedSlope(x,y,N=17, xlabel = 'X', ylabel= 'Normalized Slope / Y'):
    '''Normalizes an x,y, estimates and normalizes a slope and plots the data. \n

    Args:
        x (numeric iterable). Data for the domain. \n
        y (numeric iterable). Data for the range. \n
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
    '''Plots a graph with the inflection point and the slope throught the inflection point.
    
    Args:
        x (numeric iterable). Data for the domain. \n
        y (numeric iterable). Data for the range. \n
        N (int): window size. \n
        xlabel (str): Label for the x-axis. \n
        ylabel (str): Label for the y-axis \n
        Legend (str): Label for the Legend. \n 
        stepsize (float): step size for prediction. \n 
        
    Returns:
       Bull.  plots a graph. 
    '''
    
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
        x (numeric iterable): series. \n
    
    Returns:
        A normalized series. 
    '''
    return (x-min(x))/(max(x)-min(x))

def movingAverage(interval, window_size):
    '''Returns a moving average for an array. \n
    
    Args:
        interval (numeric iterable): Data. \n
        window_size (int): The size of the window. \n
    
    Returns: 
        numpy array with the moving average. '''
        
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
    average assuming with center = True.
    
    Args:
        df (pandas.DataFrame): DataFrame containg numeric vectors. \n
        window (int): Window size. \n
        
    Returns: 
        A dataframe contianing the moving averages. 
    '''
    #set up the arguments for the rolling_mean function. 
    min_periods = None
    freq = None
    center =True
    
    #use data.Frame.apply method to apply the rolling_mean function to every series. 
    return df.apply(rolling_mean, axis = 0, args=[ window,min_periods, freq, center] )
              
class Growth(object):
    '''Read in and parses a Tecan Excel file, and generates a Growth Object.
        
        Attributes:
            time (numpy array): Contains the time data. \n
            intensity (pandas DataFrame): Contains the Intesnity Data \n  
            MetaData (pandas Dataframe): Initially a string, but upon creation by the getMetaData function, becomes a Data frame that holds the metadata.  \n
        
        Args:
            fin (str): Absolute path or handle to the excel file. \n
            meta (str): If not equal to '', Path to to the MetaData excel file. \n
            minutes (bool): If True, the time data is converted from seconds to minutes. \n
    
        Returns: 
            A Growth Object'''
            
    def __init__(self,fin,meta ='', Minutes = True):
        #Inits the growth object
        
        #Read in the data/ 
        self.data  = read_excel(fin)
        
        #create an index object from the first column to search. 
        self.index = Index(self.data.iloc[:,0])
        
        #Get the location of the first and final row for the data. 
        self.start = self.index.get_loc('Time [s]')    #THe first row begins with time
        
        #get the time
        self.time = self.data.iloc[self.start, 1:]
        #change the index labels.  
        self.time.index = arange(len(self.time))
        self.Time = 'Time (sec)'       #a label for the plotting
        
        #if Minutes = True, convert the time (which is in seconds to minutes)
        if Minutes == True:
            self.time = self.time/60.0
            self.Time = 'Time (min)'
        
        #Get the data (assume that the data is the last row)
        self.intensity = self.data.iloc[self.start+2 : -4, 1:]    #get the intensity information
        self.intensity = self.intensity.transpose()               #transpose the data
        self.intensity.columns = self.data.iloc[self.start+2 : -4, 0]  #rename the columns
        self.intensity.columns.name = 'Sample'        
        self.intensity.index = arange(size(self.intensity.index,0))    #rename the index(rows)
        
        #get the metaData. 
        if meta=='':
            self.MetaData =''
        else: 
            self.getMetaData(meta)
        #Get the length of the columns
        self.length = len(self.intensity.columns)
        #Call estimateGrowthParameters to estimate the growth parameters. 
        self.estimateGrowthParameters()
            
    def estimateMaxRate(self, index ):
        '''helper function that fits a line and returns the maximum rate.''' 
        
        #Get the pertinent data
        x = self.intensity.iloc[:,index]
        logx = self.Log.iloc[:,index]
        lower_t = self.lower_threshold[index]
        upper_t = self.upper_threshold[index]
        #subset the log data based on the lower_t, upper_t and delta. 
        
        #Calculate a boolean that determines the points that are greater than the lower threshold and less thatn the upper threshold
        b = logical_and(x>lower_t, x<upper_t)
        #fit a line to this data.
        fit =  linregress(self.time[b],logx[b] )
        
        #get the lower indices for the time points that are fit. 
        Index = x.index[b]
        lower_idx = Index[0]  
        upper_idx = Index[-1]
       
        #return a tupple with the slope, intercept, and indexes for the tmin and tmax,
        return (fit[0], fit[1], lower_idx, upper_idx)
        
    def estimateMaxRates(self):
        '''Estimates the maximum growth rates by fitting a line to the data in with the y log transformed. This function picks points automatically ranging from the the points greater than 0.2 of the floor and 0.8 of the plateau.
        
        Args:
            None
        
        Returns:
            None
        '''
        
        #calculate delta change between the floor and the plateau
        self.delta = self.plateau-self.floor
        
        #calculate a lower and upper threshold. 
        self.lower_threshold = 0.2*self.delta + self.floor
        self.upper_threshold = 0.8*self.delta + self.floor
        
        #use the estimateMaxRate to get a tupple
        self.MaxRateTupples = [self.estimateMaxRate(i) for i in range(self.length)]
       
        #get the max rates
        self.MaxRates = [tupple[0] for tupple in self.MaxRateTupples]
       
    
    def estimateLagTime(self, index=0):
        '''returns the lag times for the particular series'''
        
        #predict the line that goes through the max rate for the plateau
        
        #First calculate the prediction
        #get the required parameters
        m, c ,xmin, xmax = self.MaxRateTupples[index]
        
        #use the parameters and the the floor to solve for the time in in which the prediciton hits fllor. 
        return (log(self.floor[index])-c)/m
     
    def estimateLagTimes(self):
        '''uses estimateLagTime to estimate the lag time using a for loop'''
        ###estimate the Lag time for each. 
        self.lagTimes = [self.estimateLagTime(i) for i in range(self.length)]
        
    def estimateGrowthParameters(self, window=5):
        '''Estimates the log time, maximal growth rate and the plateau/ Carrying capacity from data
        Args:
            window (int): The window size to be used for the rolling mean.
        
        Returns:
            A tuple (tlag, uMax, floor,plateau)
            '''
        
        #Calculate the Log of the data which will be used in the estimation of the max rate. 
        self.Log  = self.intensity.apply(Log)

        #Compute a rolling Mean     
        self.rolling = rollingMeanDf(self.intensity,window=5)

        #Determine the values and index for the floor
        self.floor = self.rolling.min(axis=0)
        self.floor_index= self.rolling.idxmin(axis=0)

        #Determine the values and index for the plateau
        self.plateau = self.rolling.max(axis=0)
        self.plateau_index = self.rolling.idxmax(axis=0)

        #compute the maximal growth ratefor each column
        self.estimateMaxRates()

        #compute the LagTimes: 
        self.estimateLagTimes()

        #make a data frame with the parameters for easy printing. 
        self.formatParameters()

    def plot(self, save =False, path = '', logY=False):
        #Plots the intensity data. 
        self.intensity.plot(x=self.time, grid = False, logy=logY)
        plt.xlabel(self.Time, size = 14)
        plt.ylabel('OD (600 nM)',size=14)
        
        #plt.show()
        #If save ==True: plot 
        if save ==True:
            plt.savefig(path)
    
    def plotMaxRate(self, index):
        '''Plots the log of a a single growth curve and the its predicted growth rate.
        
        Args:
            Index (int): The index of growth curve to be plotted. \n
            
        Returns:
            None
        '''
        
        #Get the the parameters for the prediction
        m,c, xmin, xmax = g.MaxRateTupples[index]
        #Predict the line for the slope. 
        x_pred,y_pred = predict(m,c,self.time[self.floor_index[index]],g.time[self.plateau_index[index]])
        
        #plot the data
        plot(self.time, self.Log.iloc[:,index], x_pred, y_pred)
        #set the axis
        axis([min(self.time), 
              max(self.time),   
              round(log(self.floor[index]),1)- 0.1, 
              round(log(self.plateau[index]),1) +0.1 ])     
        
    def getMetaData(self,meta):
        '''Loads and parses meta data from a excel file
        
        Args:
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
            NewGrowth.length = len(NewGrowth.intensity.columns)
            NewGrowth.estimateGrowthParameters()
            return NewGrowth
    
    def zero(self, N=5, zero = False):
        '''Zero the intensity data by the first N points. if zero == True,then it converts all negative numbers to zero. '''
        self.intensity -= self.intensity.iloc[0:N, :].mean(axis=0) #take the average the first N points and subtract the average from the original data
        if zero ==True:
            self.intensity[self.intensity<0] =0
            
    def formatParameters(self):
        '''returns the Parameters in a pandas DataFrame''' 
        self.Parameters =DataFrame({  'Max Growth Rates': self.MaxRates, 'Lag Times (s)': self.lagTimes, 'Floor': self.floor, 'Plateau': self.plateau, 'Delta OD600': self.delta})
        
    def PlotParameters(self , x='Concentration'):
        '''Plots the parameters on scatter plots versus the concentrations.
        Args: 
            x (array): If x is left to concentration (which we assume it will be) this will plot the data versus concentration.
            '''
        if x=='Concentration':
            x = self.MetaData.Concentration
        
        #Generate the Plots for parameters
        fig,ax= plt.subplots(1,3, figsize = (9,2.2))
        ax[0].scatter(x, self.Parameters[[0]])
        ax[0].set_xlabel('Concentration')
        ax[0].set_ylabel('Delta OD600')
        ax[1].scatter(x,self.Parameters[[1]])
        ax[1].set_xlabel('Concentration')
        ax[1].set_ylabel('Lag Times (Min)')
        ax[2].scatter(x, self.Parameters[[2]])
        ax[2].set_xlabel('Concentration')
        ax[2].set_ylabel('Max Growth Rate')
        fig.tight_layout()
        
        
    def AverageIntensities(self):
        ''' Average the intensities'''
        M = self.MetaData.copy().iloc[:,1:]
        #make an array of tupples for the metadata and make a multiindex
        M = [tuple(M.iloc[i,:]) for i in range(len(M))]
        self.intensity.columns = MultiIndex.from_tuples(M , names = ['Amino Acid', 'Concentration', 'Replicate'])
        
        #get the avereage intesnity. 
        self.MeanIntensity = g.intensity.groupby(axis=1, level = ['Amino Acid', 'Concentration']).apply(mean, axis = 1)
        
        
                        
if __name__=='__main__':
    path = '/Users/christopherrivera/Desktop/PlateReaderGrowthExperiments/May222014'
    fin = join(path, '5-23-14-platereder.xlsx')
    meta = join(path, 'metaDataMay222014.xlsx')
    
    g = Growth(fin, meta ,Minutes= True)
    
    ##make a copy of the meta data
    #M = g.MetaData.copy().iloc[:,1:]
    #make an array of tupples for the metadata and make a multiindex
    #M = [tuple(M.iloc[i,:]) for i in range(len(M))]
    #g.intensity.columns = MultiIndex.from_tuples(M , names = ['Amino Acid', 'Concentration', 'Replicate'])
    
    #get the avereage intesnity. 
    #g.MeanIntensity = g.intensity.groupby(axis=1, level = ['Amino Acid', 'Concentration']).apply(mean, axis = 1)