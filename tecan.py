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
def createMultiIndexFromDataFrame(df):
    '''Transforms a dataframe (df) into a list of tupples and then into a MultiIndex object.
    Args:
        df ( pd.DataFrame)
    Returns:
        MultiIndex'''
    #Use apply to convert the df to a Series of tupples
    index = df.apply(tuple,axis = 1)
    
    #Convert this to a multiIndex object and return it
    return pd.MultiIndex.from_tuples(index)

class Growth(object):
    '''Read in and parses a Tecan Excel file, and generates a Growth Object. The growth 
    object contains the data from the tecan experiment in a dictionary that holds each 
    "data" type (i.e. Absorbance, Flourescence ) as a value. It also has several other 
    attributes. as well as a dataframe with metadata
    This assumes that the object contains at least one measurment for asborbances at 
    600 nm (growth) and allows for multiple flourescence readings.
        
        Attributes:
            time (pandas series): Contains the time data. \n
            dataDict (dictionary): Dictionary that contains the data with metadat 
            MetaData (pandas Dataframe): Initially a string, but upon creation by the getMetaData function, becomes a Data frame that holds the metadata.  \n
            Parameters (pandas Dataframe): Contains the parameters for the fit. 
        Args:
            fin (str): Absolute path or handle to the excel file. \n
            meta (str): If not equal to '', Path to to the MetaData excel file. \n
            minutes (bool): If True, the time data is converted from seconds to minutes. \n
            fitData (bool): If True use several methods to fit the data and extract parameters. 
    
        Returns: 
            A Growth Object'''
            
    def __init__(self, fin, meta ='', Minutes = True, fitData = False):
        #Initializes  the growth object
        
        #import the raw data from the Tecan Excel file as a data frame         
        self.raw = pd.read_excel(fin, skiprows = 22)   
        
        #get the first column of the data frame;
        #this column will be searched to identify tags (eg. Mode and Part of plate 
        #to tell the program the location of the data. )
        self.firstColumn = self.raw.iloc[:,0].copy()
        
        #convert the na values to empty string for easier searching. 
        self.firstColumn[pd.isnull(self.firstColumn)] = ''   
        
        #Identify the rows that contain the mode information
        self.modesLocation=self.getRowIndexForWord('Mode')
        
        #get the names for the modes. 
        self.modes = [self.raw.iloc[i, 4] for i in self.modesLocation]
        
        ###### Identify the locations of the raw data that hold the actual data. 
        #Search for the row that contains the Part of Plate
        PartOfPlate = self.getRowIndexForWord('Part of Plate')[0]  #only one value is expected
        
        #Get the string that contains the information for part of plate and split it into component parts. 
        PartOfPlate = self.raw.iat[PartOfPlate,4].split('-')   #iat method returns the value at the specific location. 
        print PartOfPlate
        #now get the locations for the start and end of the data values. 
        #these should include a list of values.  
       
        #identify the rows that contain the actual data using a method
        self.firstRows = self.getRowIndexForWord(PartOfPlate[0])
        self.lastRows =  self.getRowIndexForWord(PartOfPlate[1])
        
        #create a dictionary that contains the data as data frames
        self.createDataFrameDict()
        
        #get the time and attach it as its own variable
        timeIndex = self.firstColumn[self.firstColumn =='Time [s]'].index[0] #we only want the first value
        #print timeIndex
        self.time = self.raw.iloc[timeIndex, 1:] /60  #skip the first value and covnert to minutes
        self.time.name = 'Time [Min]'   #rename it    
        
        #If the metadata is provided call the importMetaData function to import the meta Data.
        #if metadata != '':
         #   self.importMetaData(meta)
            
    def createDataFrameDict(self):
        #get the data for each mode and place it to a dictionary with Mode: df pair 
        dataDict = {}
        for i in range(len(self.modes)):
            
            #get the data, copy it and transpose it   
            first = self.firstRows[i]
            last = self.lastRows[i] + 1  #Add one to include
            data = self.raw.iloc[first:last,1:].copy().T  #T transposes
            data.columns = self.raw.iloc[first:last,0] #rename the columns
            
            dataDict[self.modes[i]] = data   #attach to the diction
            
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
        
    def plot(self, mode = 'Absorbance',title=''):
        '''Plots the data'''
        data = self.dataDict[mode]   #get the data
        data.plot(x= self.time)
        plt.xlabel('Time [min]')
        plt.ylabel('Intensity (Au)')
        if title == '':
            title = mode
        plt.title(mode + ' Versus Time' )
        
    def importMetaData(self, metadata):
        '''Imports metadata, as a data frame and reformats the columns.
        This assumes that the metadata is formated with 3 to 4 columns and is labeled
        Well, Condition, Concentration 1, Concentration 2'''
        
        #import the metadata        
        meta =  pd.read_csv(metadata)

        #atttach it as an attribute that can be called later.         
        self.metadata = meta
        
        #go through the each key in the dataDict and assign the metadata to the columns
        #for the attached DataFrame 
        for mode in self.dataDict:
            self.dataDict[mode].columns = meta

        #create a new DataDict with the average values. 
        
        
        
        
        
        #create a 
#%%        
#test code 
if __name__ == '__main__':   
    
    fin ='/Users/christopherrivera/Desktop/testTecan.xlsx'
    g = Growth(fin)
    #g.plot(g.modes[0])
    metadata = '/Users/christopherrivera/Desktop/metadata.csv'
    metadata = pd.read_csv(metadata)
   
    

   
m = createMultiIndexFromDataFrame(metadata.iloc[1:, :])


d = g.dataDict.values()[0]
d.columns = m

#now try to group by


g = d.groupby(by =m, axis =1)
