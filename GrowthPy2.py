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
    '''calculates the log of every entry in an array'''
    return [log(i) for i in x]

def movingRegression(x,y,N=17):
    '''Fits a moving linear regression to a set of points using the linregress function, 
    and returns a list of tupples (m,c,X,Y) where m is the slope, c is the y-intercept, X is midpoint, Y is the midpoint value 
    the function assumes that the Number of points for the fit is odd, if N is even, it add an addional point. '''
    
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
    
    #Get a truncated interval for moving linear regression. 
    l = len(x) 
    L = range(n, l- n)   #L containts the indexs corresponding to x minus the first n and last n indicies. 
        
    n = (N-1)/2   #calcuate the number of points to the left and right of the center. 
    #fit moving linear regressions. 
    fits =   [linregress(x[i-n:i+n], y[i-n:i+n]) for i in  L]  
    return [(fits[i][0], fits[i][1], x[i+n], y[i+n]) for i in range(len(fits))]   #this returns a list of tupples (m,c,x-midpoint,y-midpoint)
    
def findInflection(x,y, N=17):
    '''Finds the inflection point in a curve by finding the point with max slope, 
    Returns a tupple for the inflection point (m,c,x,y).'''
    return max(movingRegression(x,y, N))   #get the inflection slope. 

def predict(m, c,xmin, xmax, stepsize = 0.1):
    '''Returns the the prediction of a linear regression for drawing and stuff, you know for kids. '''
    #Inputs m: slope, c:y-intecept, xmin- minium x-value, xmax  = maximum x value, stepsize: step size in an interval
   
   #Generate an array of X to predict on. 
    X = arange(xmin, xmax, stepsize)    
    #Calculate the prediction and return
    return (X,[c+m*x for x in X ])
    
    
def plotNormalizedSlope(x,y,N=17, xlabel = 'X', ylabel= 'Normalized Slope / Y'):
    '''Normalizes the move slope and the Y values of a data set and plots these on the same graph for 
    easy visualization.'''
    #The inputs are as above. 
    #Calculate a moving linear Regression. 
    fits = movingRegression(x,y,N)
    
    #From the fits, get the x interval and the normalized slopes
     #x interval for the slopes
    x2 = [f[2] for f in fits]  
    #get thenormalized slopes
    m =  [f[0] for f in fits]            
            
    #Get the coordinates for the Inflection point to plot.  
    Inflection= max(fits)              #get the tupple for the inflection point.
    Ix  = Inflection[2]                #get the x for the inflection point and normalize it. 
    Imy = (Inflection[0] - min(m))/(max(m)-min(m))       #get a normalized slope at the infleciton point.    
    Iy  = (y[list(x).index(Ix)] - min(y))/(max(y) -min(y))    #get the normalized y at the inflection point
    
    #Plot the Data
    plt.plot(x2, normalize(m), x, normalize(y))  
    plt.plot(Ix,Iy,'g.', Ix,Imy, 'b.', markersize= 20)
    
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
    '''Nomalize an array s.t. z = (x-min(x))/(max(x)-min(x)'''
    return (x-min(x))/(max(x)-min(x))

class Growth(object):
    '''Import and formats data from a Tecan Plate reader experiment for plotting and analysis
    
    Attributes:
        time (numpy array): Contains the time data. 
        intensity (pandas DataFrame): Contains the Intesnity Data  
        MetaData (pandas Dataframe): Initially a string, but upon creation by the getMetaData function, becomes a Data frame that holds the metadata.  
    
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
            print meta
        else: 
            self.MetaData = self.getMetaData(meta)
        
    def getParameters(self):
        '''Calculate the parameters'''
        pass
        
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
        self.MetaData = read_excel(fin)
    
    def subset(self,Value='', Parameter = 'Condition' ):
        '''Creates a new Growth object subset on a given Parameter and Value.'''
        
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

    
    #g.plot()
    #example slopes

  #  t = g.time
  #  y = g.intensity.iloc[:,13]

    #y = normalize(y)


    #plotNormalizedSlope(t,y)
    #plotInflecionSlopeOnGraph(t,y,45)
   # n = arange(15,45,16)
    #M = [findInflection(t,y,i )[0] for i in n]
    #plt.plot(n,M)
    #plt.figure()
    #[plotInflecionSlopeOnGraph(t,y,i) for i in n]
    #plt.show(True)
    
     


#    m = calculateSlopes(t,y,N)
#    
#    t2 = t[n:len(t)-n]
#    y2  = y[n:len(t)-n]
#    m= normalize(m)
#    y = normalize(y)
#    y2= normalize(y2)
#    
#    plt.plot(t2,m, t,y)
#    m=list(m)
#    i=m.index(max(m))
#    T=t2[i]
#    Y= y2[i]
#    M = m[i]
#    plt.plot(T,Y,'ro', T, M, 'go', markersize = 10)
#    axes = plt.gca()
#    axes.set_ylim([0,1.2])
#    
#    #find the new index
#    i =list(t).index(T)   #this is where the index is. 
#    #we want to estimate a slope around 
#    c= linregress(t[i-n:i+n], y[i-n:i+n])
    

    
    
    
    
#    t4 = arange(0,1200,10)
#    pred =[c[1]+c[0]*tt for tt in t4]
#    plt.plot(t4,pred)
# 

    