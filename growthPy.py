# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:04:27 2014

@author: chrisrichardrivera
"""

from pandas import *
from numpy import *
from matplotlib.pyplot import *
from scipy.stats import *
import seaborn as sns
from os.path import join



def importGrowth(fin, minutes = True):
    '''Imports the growth rate data as a pandas DataFrame object and returns two vectors objects: one for time and one for growth'''
    #this function assumes that the data has been previously formated to include columns only for desired data. 
   
    data= read_csv(fin, delimiter = '\t')  #use the pandas read_csv function to import the data.   
    #return the data, the first column is time, the second is the data
    
    if minutes == True:
        #if minutes == True, then it converts the time to minutes by dividing by zero
        return data.iloc[:,0]/60, data.iloc[:,1:]
    else: 
        return data.iloc[:,0]/60, data.iloc[:,1:]
    
  
def zeroGrowth(growth, N, zero = False):
    '''This zeros the the data by subtracting the average of the N points for the data. if zero == True,then it converts all negative numbers to zero. '''
    g= growth - growth.iloc[0:N, :].mean(axis=0) #take the average the first N points and subtract the average from the original data
    
    if zero ==True:
        g[g<0] =0
        return g +0.00001
    else:
        return g

def averageByNColumns(df, N=2):
    '''Takes the data and averages the data by N columns. '''
    #This assumes that the data is already sorted
    df = df.copy()    
    l = len(df.columns)/N   #calculate the number of bins
    groups = [range(i*N, N*(i+1)) for i in range(l)]   #make groups
    Names = [df.columns[i*N] for i in range(l)]    #get the names
    
    df2 =DataFrame({i: df.iloc[:,groups[i] ].mean(axis=1) for i in range(len(groups))})
    df2.columns = Names
    
    return df2
        
def getRates(x,y, N= 5):
    '''This calculates a moving slope for a 2 vectors. x, y'''
    #this assumes that the length of x and y are the same. 
    return  [linregress(x[i:i+N], y[i:i+N])[0] for i in  range(len(x)-N+1)]
    

def getRatesDF(x,y, N= 5):
    '''Calculates moving rates for each colummn of a data frame.    '''
    #This assumes that x is one vector (time) and Y is an data frame array. 
    #this returns as a data frame
    slopes = {i:getRates(x, y.iloc[:,i], N) for i in range(len(y.columns))} 
    return DataFrame(slopes).sort(axis=1)

def getMaxRatesDF(x,y, N=5):
    '''Returns the maximum slope and the time at the maxium slope'''
    t= array(x)
    #get the slopes and the maximum
    slopes = getRatesDF(x,y,N)
    MaxSlopes = slopes.max()    
    n = int(floor(N/2))    #this is value for indexing. we want to return 
    x = x[n: len(x)-n]
    
    #use a for comprehenstion to loop through and find the time for the max slope
    #This is a complicated comprehension
    #It first subindexes the slopes data frame to look at one colum at a time
    #then it converts the vector into a list. 
    #then it uses the index method to find the index for the maximum value. 
    # it then finds subindexes on that index. 
    #it uses the comprehsion to make a list 
    #It returns the list as an array
    k = len(slopes.columns)   #the number of columns
    t = [t[list(slopes.iloc[:, i]).index(MaxSlopes[i])] for i in range(k)  ]
    
    #return numpy arrays
    return array(MaxSlopes), array(t)

def getPlateau(x, y, N= 5):
    '''This returns the maximum plateau and and array at which that occureds. '''
    
    #First calculate a rolling mean. 
    Rolling= y.apply(rolling_mean, axis = 0, args = [N])
    Plateau = list(Rolling.max())   #get max rolling values which are the plateau. 
    k = len(y.columns)   #the number of columns
#    
#    #This uses a comprehensions to determine the plateau values. 
    L = [list(Rolling.iloc[:, i]) for i in range(k)]    #Convert every vector in the DataFrame into a list.
    indices = [L[i].index(Plateau[i]) for i in range(k)]  #Get the corresponding index for the Plateau values. 

    return array(x[indices]),array( Plateau)


def getTimeToReach(x, y, Value=0.1):
    '''calculates the first time to reach a value. (we will consider an average value later). '''
    #Calculate a data frame with containg the absolute deviation from the desired location    
    y2 = y-Value
    y2 = y2.abs()
    y2min = list(y2.min()) #this must converted to a list. 
    
    #use the above paradigm to find the desired index. 
    k = len(y.columns)   #ket the length of the columns
    index= [list(y2.iloc[:, i]).index(y2min[i]) for i in range(k) ]
     
    #use comprehension to determine the y value for the givne index. 
    Yvalue = [y.iloc[index[i],i] for i in range(k)]
    x = [x.iloc[index[i]] for i in range(k)]
    return array(x), array(Yvalue)
    #return index

def PlotTimeToReach(x, y,Con,  Value = 0.02):
    '''Calculates the time to reach and plots the values''' 
    #x is a series of times, y is a dataframe, z is a vector of concentrations ideally.
    #Call the time to reach function to get the times and the Y values at the posttion 
    tr,Y = getTimeToReach(x,y, Value )   #tr is the time to reach, Y are the values. 
    #Create a new figure and draw to subplots    
    df = DataFrame({'uM': Con, 'Time': tr} )
    
    

    subplot(1, 2, 1)
    y.plot(x=x)
    scatter(tr,Y)
    xlabel('Time [min]')
    ylabel('OD 600nm')
    subplot(1,2,2)
    df.plot(x= 'uM', y = 'Time')
    ylabel('Time [min] to Reach OD of' + str(Value))


def zeroGrowthByMin(growth, N, zero = True):
    '''This zeros the the data by subtracting the average of the minimun N points for the data. if zero == True,then it converts all negative numbers to zero. '''
    g = growth.apply(lambda x: x - mean(sort(x)[:N]))   #usese the apply function to call a lamda function column by column
    
 
    if zero ==True:
        g[g<0] =0
        return g +0.00001
    else:
        return g



if __name__ == '__main__':
  pass