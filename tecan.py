# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:11:49 2015

@author: christopherrivera

These function are for importing,parsing and analyzing tecan plate reader data. 
"""
#%% import the necessary modules
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ylabel, xlabel,gca
import seaborn as sb
from os.path import join

from copy import copy
from scipy.stats import linregress

#%%
class Growth(object):
    '''Reads in and parses a Tecan Excel file to generates a Growth Object. The Growth 
    object contains the data from the tecan experiment in a dictionary that holds each 
    "data" type (i.e. Absorbance, Flourescence ) as a value. It also has several other 
    attributes including as a dataframe with metadata.
    This assumes that the object contains at least one measurment for asborbances at 
    600 nm (growth) and allows for multiple flourescence readings.
        
        Attributes:
            time (pandas series): Contains the time data. \n
            dataDict (dictionary): Dictionary that contains the data. \n
            MetaData (pandas Dataframe): DataFrame that contains metadata. \n
            Parameters (pandas Dataframe): Holds the parameters for fits. \n
        Args:
            fin (str): Absolute path or handle to the excel file. \n
            meta (str): If not equal to '', path to to the MetaData excel file. \n
            minutes (bool): If True, the time data is converted from seconds to minutes. \n
            fitData (bool): If True use several methods to fit the data and extract parameters.\n 
    
        Returns: 
            A Growth Object'''
            
    def __init__(self, fin, meta ='', minutes = True, fitData = False):
        #Initializes  the growth object
        
        #import the raw data from the Tecan Excel file as a data frame         
        self.raw = pd.read_excel(fin, skiprows = 22)   
        
        ########################################################################
        #Parse the excel file by 1) creating a searchable serires object from the 
        #first column, 2) Searching for keywords 3) Using the keywords to find 
        #appropriate index location. 4) Extracting the infomration and attaching 
        #it a good data structure. 
        
        
        #####get the first column of the data frame;
        #this column will be searched to identify tags (eg. Mode and Part of plate 
        #to tell the program the location of the data. )
        self.firstColumn = self.raw.iloc[:,0].copy()
        
        #convert the NA values to an empty string for easier searching. 
        self.firstColumn[pd.isnull(self.firstColumn)] = ''   
        
        #Identify the rows that contain the mode information
        self.modesLocation=self.getRowIndexForWord('Mode')
        
        #get the names for the mode and place in a list. 
        self.modes = [self.raw.iloc[i, 4] for i in self.modesLocation]
        
        ###### Identify the locations of the raw data that hold the actual data. 
        #Search for the row that contains the Part of Plate
        PartOfPlate = self.getRowIndexForWord('Part of Plate')[0]  #only one value is expected
        
        #Get the string that contains the information for part of plate and split it into component parts. 
        PartOfPlate = self.raw.iat[PartOfPlate,4].split('-')   #iat method returns the value at the specific location. 
        
        ###### Now get the locations for the start and end of the data values. 
        #these should include a list of values.  
       
        #identify the rows that contain the raw intensity values. 
        self.firstRows = self.getRowIndexForWord(PartOfPlate[0])
        self.lastRows =  self.getRowIndexForWord(PartOfPlate[1])
        
        #create a dictionary that contains the data as data frames
        self.createDataFrameDict()  #this functoin creates a data fram with the rows
        
        #get the time and attach it as its own variable
        timeIndex = self.firstColumn[self.firstColumn =='Time [s]'].index[0] #we only want the first value
        #print timeIndex
        self.time = self.raw.iloc[timeIndex, 1:] /60  #skip the first value and covnert to minutes
        self.time.name = 'Time [Min]'   #rename it    
        
        #If the metadata is provided call the importMetaData function to import the meta Data.
        if meta !=  '':
            self.importMetaData(meta)
            self.Parameters = self.estimateExponentialParameters()
            
    def createDataFrameDict(self):
        #get the data for each mode and place it to a dictionary with Mode: df pair 
        dataDict = {}
        for i in range(len(self.modes)):
            
            #get the data, copy it and transpose it   
            first = self.firstRows[i]
            last = self.lastRows[i] + 1  #Add one to include
            data = self.raw.iloc[first:last,1:].copy().T  #T transposes
            data.columns = self.raw.iloc[first:last,0] #rename the columns
            
            #zero the data
            data=data.apply(lambda(x):x-np.min(x))            
            
            dataDict[self.modes[i]] = data   #attach to the dictionary
            
        self.dataDict = dataDict
        
    def getRowIndexForWord(self, word):
        '''Helper function that searches the firstColumn index for key words.
        Args: 
            word (str)
        Returns: 
            idicies (int or list of indexes. '''
        index = self.firstColumn.str.contains(word).astype('bool')
        #return a list of the indicess
        indices =  list(self.firstColumn.index[index])
        #if len(indices )==1:
         #   return indices[0]
        return indices
        
    def plotData(self, mode = 'Absorbance',title=''):
        '''Plots the data'''
        data = self.dataDict[mode]   #get the data
        data.plot(x= self.time)
        plt.xlabel('Time [min]')
        plt.ylabel('Intensity (Au)')
        if title == '':
            title = mode
        plt.title(mode + ' Versus Time' )
               
    def estimateExponentialParameters(self, minOD = 0.05, maxOD = 0.30):
        #Estimate several parmetes based on a exponential fit. And place in a data frame
        
        #Get the Absorbance data and then compute the parameters. 
        data =self.averageData['Absorbance']
      
        def estimatefit(t,x, minOD, maxOD):
            #perform linear regression on data points within  a series.
            #convert to log
            y = np.log(x+.0001)
            #subset to the points that are with in an interval of the minOD and the MaxOD
            included_points = np.logical_and(x>minOD, x<maxOD)
            T = t[included_points]
            #T = T-np.min(T)
            Y =  y[included_points]
            #plt.plot(T, Y )#, '.' ,markersize = 10)
            return list(linregress(T, Y)) + [len(T)]
            
        #Create a data frame with the parameters. 
        fits = pd.DataFrame({data.columns[i]: estimatefit(self.time, data.iloc[:,i],minOD, maxOD) for i in range(len(data.columns))})
        lagtime=-fits.iloc[1,:]/fits.iloc[0,:]
        
        #append the lag time (which is estimated from the fit. ) and the maximum value fo the series
        lagtime=-fits.iloc[1,:]/fits.iloc[0,:]
        fits = fits.append(lagtime, ignore_index=True)
        fits = fits.append(data.apply(np.max), ignore_index=True)
        fits.index = ['slope', 'intercept', 'r-value', 'p-value', 'stderr','Number of Points','Lag Time', 'Max']
        return fits
           
    def importMetaData(self, metadata):
        '''Imports metadata, as a data frame and reformats the columns.
        This assumes that the metadata is formated with 3 to 4 columns and is labeled
        Well, Condition, Concentration 1, Concentration 2'''
        
        #import the metadata
        self.meta=  pd.read_csv(metadata)
        index = self.createMultiIndexFromDataFrame(self.meta)  #convert the meta df into a muliindex object
        #format the metadata to a multindex object. 
        
        #create a new dataDict and call it averageData
        self.averageData = {}        
        
        #in a for loop: 
        #attach the metadata to the columns of each index for every dataframe in the datadictionary
        for mode in self.dataDict:
            self.dataDict[mode].columns = index #convert the metadata
            
            #use the group apply combine idiom to calculate the means. 
            #transpose the dataframe and then group by the appropriate levels. 
            
            grouped = self.dataDict[mode].T.groupby(level =range(1,len(self.meta.columns)))
            #compute the average then transpose and attach to the average dataFrame 
            self.averageData[mode] = grouped.apply(np.mean).T
            
    def createMultiIndexFromDataFrame(self,df):
        '''Transforms a dataframe (df) into a list of tupples and then into a MultiIndex object.
        Args:
            df ( pd.DataFrame)
        Returns:
            MultiIndex'''
        #Use apply to convert the df to a Series of tupples
        index = df.apply(tuple,axis = 1)
                    
        #Convert this to a multiIndex object and return it
        return pd.MultiIndex.from_tuples(index, names = df.columns)
      
#%%        
#test code 
if __name__ == '__main__':   
    
    fin ='testTecan.xlsx'
    
    metadata = 'MetaData.csv'
    g = Growth(fin, metadata)
    metadata = pd.read_csv('MetaData.csv')
    meta = metadata.apply(lambda x: x.astype(str))
    
    
    
    ''' 
    def plotParameters(Parameters):
        #plots the parameters 
        #1. Group by the meta data level 0
        grouped = Parameters.groupby(level = 0,axis = 1) #group by the first level
        for name, group in grouped: 
            f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)   #create a suplot
            #plot growth rate (Mu), lag time, Maximum
            ax1.plot()
            
            
            
            
            
            
    data =g.Parameters 
    plotParameters(data)'''
    
#%%%





