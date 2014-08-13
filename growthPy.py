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
    close('all')
#    
    #load the data
    fin = '/Users/chrisrichardrivera/Documents/Darpa/Data/GrowthRateExperiments/growth.txt'
    t, g =importGrowth(fin)  #imprt the growth
    g = zeroGrowth(g, 5, zero=True)   #zero the growth. 
    g = g+0.000001
    sp = '/Users/chrisrichardrivera/Documents/Darpa/Data/GrowthRateExperiments/figures/'  #this is the save path
    sns.set_style("white")




    #load the data
    fin = '/Users/chrisrichardrivera/Documents/Darpa/Data/GrowthRateExperiments/growth.txt'
    t, g =importGrowth(fin)  #imprt the growth
    g = zeroGrowth(g, 5, zero=True)   #zero the growth. 
    g = g+0.000001
    sp = '/Users/chrisrichardrivera/Documents/Darpa/Data/GrowthRateExperiments/figures/'  #this is the save path
    sns.set_style("white")
    
    
#########################################################
    #### Mercury
    Mercury =g.iloc[:, 36:]
    
        #partition the data frame and rename the columns
    Hg = array([10.0/3**i for i in range(9)]*2)
    Mercury.columns = Hg
    Mercury = Mercury.sort(axis= 1)
    Hg = sort(Hg)
    
    #rename the Hg
    Mercury.columns =Hg
    Mercury = Mercury.copy()   #its acting wierd. Hgping is fixing the problem
        
##plot the Mercury data
    Mercury.plot(x= t)
    ylim(-.1, 1)
    ylabel('OD 600')
    title('OD verus time for Mercury')
    savefig(join(sp, 'MercuryODTime.pdf'))
    
    
    #Time to Reach
    t2, Y= getTimeToReach(t, Mercury, Value=0.02)
    
   #plot the Mercury data with log and the time to reach
    Mercury.plot(x= t, logy=True)
    ylim(0, 0.1)
    scatter(t2, Y, s = 60)
    

    ylabel('OD 600')
    title('OD verus time for Mercury')
    savefig(join(sp, 'MercuryODTimeLogWithTime.pdf')) 
    
    
     #For mercury we need to plot only a subset. 
    Mercury = Mercury.iloc[:, 0:10]
    
#    #plot the time to reach
    figure()
    t2 = t2[0:10]
    Hg = Hg[0:10]
    scatter(Hg, t2 ,s = 60, alpha = 0.5)
    title('Time to reach 0.02')
    xlabel('Mercury [uM]')
    ylabel('Time (m)')
    xlim(-0.1,0.5)
    savefig(join(sp, 'MercuryTimeToReach.pdf')) 
    
    
    #Remove the first points. 
    t3 = t[t>max(t2)]
    l = len(Mercury) - sum(t>max(t2))
    Mercury = Mercury.iloc[l:, :]
    Mercury.plot(x= t3)
    #Max growth rate
    
    rate,t4 = getMaxRatesDF(t3,Mercury.apply(log, axis = 0), N=11)
    figure()
    scatter(Hg, rate, s= 60, alpha = 0.5)
    xlim(-0.1,0.5)
    xlabel('uM Mercury')
    ylabel('Max Rate')
    title('Max Rate versus Mercury Hgncentration')
    savefig(join(sp, 'MercuryRate.pdf'))    
    
    
##########################################################
#    #### Cobalt
#    Cobalt =g.iloc[:, 18:36]
#    
#        #partition the data frame and rename the columns
#    Co = array([20.0/3**i for i in range(9)]*2)
#    Cobalt.columns = Co
#    Cobalt = Cobalt.sort(axis= 1)
#    Co = sort(Co)
#    
#    #rename the Co
#    Cobalt.columns =Co
#    Cobalt = Cobalt.copy()   #its acting wierd. Coping is fixing the problem
#        
##plot the Cobalt data
#    Cobalt.plot(x= t)
#    ylim(-.1, 1)
#    ylabel('OD 600')
#    title('OD verus time for Cobalt')
#    savefig(join(sp, 'CobaltODTime.pdf'))
#    
#    
#    #Time to Reach
#    t2, Y= getTimeToReach(t, Cobalt, Value=0.02)
#    
#   #plot the Cobalt data with log and the time to reach
#    Cobalt.plot(x= t, logy=True)
#    ylim(0, 0.1)
#    scatter(t2, Y, s = 60)
#
#    ylabel('OD 600')
#    title('OD verus time for Cobalt')
#    savefig(join(sp, 'CobaltODTimeLogWithTime.pdf')) 
#    
#    #plot the time to reach
#    figure()
#    scatter(Co, t2, s = 60, alpha = 0.5)
#    title('Time to reach 0.02')
#    xlabel('Cobalt [uM]')
#    ylabel('Time (m)')
#    xlim(-0.1,25)
#    savefig(join(sp, 'CobaltTimeToReach.pdf')) 
#    
#    
#    #Remove the first points. 
#    t3 = t[t>max(t2)]
#    l = len(Cobalt) - sum(t>max(t2))
#    Cobalt = Cobalt.iloc[l:, :]
#    Cobalt.plot(x= t3)
#    #Max growth rate
#    
#    rate,t4 = getMaxRatesDF(t3,Cobalt.apply(log, axis = 0), N=11)
#    figure()
#    scatter(Co, rate, s= 60, alpha = 0.5)
#    xlim(-0.1,25)
#    xlabel('uM Cobalt')
#    ylabel('Max Rate')
#    title('Max Rate versus Cobalt Concentration')
#    savefig(join(sp, 'CobaltRate.pdf'))    
    
    
    #########################################################
#    #### Copper
#    Copper =g.iloc[:, 0:18]
#    
#        #partition the data frame and rename the columns
#    #Copper    
#    Copper = g.iloc[:, 0:18]
#    Cu = array([20.0/3**i for i in range(9)]*2)
#    Copper.columns = Cu
#    Copper = Copper.sort(axis= 1)
#    Cu = sort(Cu)
#    
#    #rename the Cu
#    Copper.columns =Cu
#    Copper = Copper.copy()   #its acting wierd. Coping is fixing the problem
#        
##plot the Copper data
#    Copper.plot(x= t)
#    ylim(-.1, 1)
#    ylabel('OD 600')
#    title('OD verus time for Copper')
#    savefig(join(sp, 'CopperODTime.pdf'))
#    
#    
#    #Time to Reach
#    t2, Y= getTimeToReach(t, Copper, Value=0.02)
#    
#   #plot the Copper data with log and the time to reach
#    Copper.plot(x= t, logy=True)
#    ylim(0, 0.1)
#    scatter(t2, Y, s = 60)
#
#    ylabel('OD 600')
#    title('OD verus time for Copper')
#    savefig(join(sp, 'CopperODTimeLogWithTime.pdf')) 
#    
#    #plot the time to reach
#    figure()
#    scatter(Cu, t2, s = 60, alpha = 0.5)
#    title('Time to reach 0.02')
#    xlabel('Copper [uM]')
#    ylabel('Time (m)')
#    xlim(-0.1,12)
#    savefig(join(sp, 'CopperTimeToReach.pdf')) 
#    
#    
#    #Remove the first points. 
#    t3 = t[t>max(t2)]
#    l = len(Copper) - sum(t>max(t2))
#    Copper = Copper.iloc[l:, :]
#    Copper.plot(x= t3)
#    #Max growth rate
#    
#    rate,t4 = getMaxRatesDF(t3,Copper.apply(log, axis = 0), N=11)
#    figure()
#    scatter(Cu, rate, s= 60, alpha = 0.5)
#    xlim(-0.1,12)
#    xlabel('uM Copper')
#    ylabel('Max Rate')
#    title('Max Rate versus Copper Concentration')
#    savefig(join(sp, 'CopperRate.pdf'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
############cadmium    
#    
#    #partition the data frame and rename the columns
#    #Cadmium    
#    Cadmium = g.iloc[:, 0:20]
#    Cd = array([4.0/3**i for i in range(10)]*2)
#    Cadmium.columns = Cd
#    Cadmium = Cadmium.sort(axis= 1)
#    Cd = sort(Cd)
#    Cd[0]= 0
#    Cd[1] = 0
#    #rename the Cd
#    Cadmium.columns =Cd
#    Cadmium = Cadmium.copy()   #its acting wierd. Coping is fixing the problem
#    
#
################################################################################    
#    #plot the Cadmium data
#    Cadmium.plot(x= t)
#    ylim(-.1, 1)
#    ylabel('OD 600')
#    title('OD verus time for Cadmium')
#    savefig(join(sp, 'CadmiumODTime.pdf'))
#    
#    
#    #Time to Reach
#    t2, Y= getTimeToReach(t, Cadmium, Value=0.02)
#    
#   #plot the Cadmium data with log and the time to reach
#    Cadmium.plot(x= t, logy=True)
#    ylim(0, 0.1)
#    scatter(t2, Y, s = 60)
#
#    ylabel('OD 600')
#    title('OD verus time for Cadmium')
#    savefig(join(sp, 'CadmiumODTimeLogWithTime.pdf')) 
#    
#    #plot the time to reach
#    figure()
#    scatter(Cd, t2, s = 60, alpha = 0.5)
#    title('Time to reach 0.2')
#    xlabel('Cadmium [uM]')
#    ylabel('Time (m)')
#    savefig(join(sp, 'CadmiumTimeToReach.pdf')) 
#    
#    
#    #Remove the first points. 
#    t3 = t[t>max(t2)]
#    l = len(Cadmium) - sum(t>max(t2))
#    Cadmium = Cadmium.iloc[l:, :]
#    Cadmium.plot(x= t3)
#    #Max growth rate
#    
#    rate,t4 = getMaxRatesDF(t3,Cadmium.apply(log, axis = 0), N=11)
#    figure()
#    scatter(Cd, rate, s= 60, alpha = 0.5)
#    xlim(-0.1, 5)
#    xlabel('uM Cadmium')
#    ylabel('Max Rate')
#    title('Max Rate versus Cadmium Concentration')
#    savefig(join(sp, 'CadmiumRate.pdf'))
#    
# 
######################################################################################
# ### Arsenic Data
#    
#    Arsenic = g.iloc[:, 20:40]
#    As = array([10.0/3**i for i in range(10)]*2)
#    Arsenic.columns = As
#    Arsenic = Arsenic.sort(axis = 1)
#    As = sort(As)
#    #rename the As
#    As[0] = 0
#    As[1] = 0 
#    Arsenic.columns = As
#    Arsenic = Arsenic.copy()
#    
#    
################################################################################    
#    #plot the Arsenic data
#    Arsenic.plot(x= t)
#    ylim(-.1, 1)
#    ylabel('OD 600')
#    title('OD verus time for Arsenic')
#    savefig(join(sp, 'ArsenicODTime.pdf'))
#    
#    
#    #Time to Reach
#    t2, Y= getTimeToReach(t, Arsenic, Value=0.02)
#    
#   #plot the Arsenic data with log and the time to reach
#    Arsenic.plot(x= t, logy=True)
#    ylim(0, 0.1)
#    scatter(t2, Y, s = 60)
#
#    ylabel('OD 600')
#    title('OD verus time for Arsenic')
#    savefig(join(sp, 'ArsenicODTimeLogWithTime.pdf')) 
#    
#    #plot the time to reach
#    figure()
#    scatter(As, t2, s = 60, alpha = 0.5)
#    title('Time to reach 0.2')
#    xlabel('Arsenic [uM]')
#    ylabel('Time (m)')
#    savefig(join(sp, 'ArsenicTimeToReach.pdf')) 
#    
#    
#    #Remove the first points. 
#    t3 = t[t>max(t2)]
#    l = len(Arsenic) - sum(t>max(t2))
#    Arsenic = Arsenic.iloc[l:, :]
#    Arsenic.plot(x= t3)
#    #Max growth rate
#    
#    rate,t4 = getMaxRatesDF(t3,Arsenic.apply(log, axis = 0), N=11)
#    figure()
#    scatter(As, rate, s= 60, alpha = 0.5)
#    xlim(-0.1, 10)
#    xlabel('uM Arsenic')
#    ylabel('Max Rate')
#    title('Max Rate versus Arsenic Concentration')
#    savefig(join(sp, 'ArsenicRate.pdf'))
#        
################################################################################    
#   
##    #Chromium
#    Chromium = g.iloc[:, 40:]
#    Cr= array([20.0/3**i for i in range(10)]*2)
#    Chromium.columns = Cr
#    Chromium = Chromium.sort(axis = 1)
#    Cr = sort(Cr)
#    Cr[0]=0
#    Cr[1]=0
#    Chromium.columns = Cr
#    Chromium = Chromium.copy()
#    
#    
################################################################################    
#    #plot the Chromium data
#    Chromium.plot(x= t)
#    ylim(-.1, 1)
#    ylabel('OD 600')
#    title('OD verus time for Chromium')
#    savefig(join(sp, 'ChromiumODTime.pdf'))
#    
#    
#    #Time to Reach
#    t2, Y= getTimeToReach(t, Chromium, Value=0.02)
#    
#   #plot the Chromium data with log and the time to reach
#    Chromium.plot(x= t, logy=True)
#    ylim(0, 0.1)
#    scatter(t2, Y, s = 60)
#
#    ylabel('OD 600')
#    title('OD verus time for Chromium')
#    savefig(join(sp, 'ChromiumODTimeLogWithTime.pdf')) 
#    
#    #plot the time to reach
#    figure()
#    scatter(Cr, t2, s = 60, alpha = 0.5)
#    title('Time to reach 0.2')
#    xlabel('Chromium [uM]')
#    ylabel('Time (m)')
#    savefig(join(sp, 'ChromiumTimeToReach.pdf')) 
#    
#    
#    #Remove the first points.
#    #we have to subset chromium
#    t2 = t2[0:18]
#    t3 = t[t>max(t2)]
#    l = len(Chromium) - sum(t>max(t2))
#    Chromium = Chromium.iloc[l:, 0:18]
#    Chromium.plot(x= t3)
#    #Max growth rate
#    
#    
#    #We have to subset Chromium. 
#
#    rate,t4 = getMaxRatesDF(t3,Chromium.apply(log, axis = 0), N=11)
#    figure()
#    scatter(Cr[0:18], rate, s= 60, alpha = 0.5)
#    xlim(-0.1, 10)
#    xlabel('uM Chromium')
#    ylabel('Max Rate')
#    title('Max Rate versus Chromium Concentration')
#    savefig(join(sp, 'ChromiumRate.pdf'))
        

    
    
    
    
    
    
    
    
    
    
    
#   
#    
#    #Max growth rate
#    rate,t2 = getMaxRatesDF(t,Cadmium.apply(log, axis = 0), N=11)
#    figure()
#    scatter(Cd, rate, s= 60, alpha = 0.5)
#    xlim(-0.1, 5)
#    xlabel('uM Cadmium')
#    ylabel('Max Rate')
#    title('Max Rate versus Cadmium Concentration')
#    savefig(join(sp, 'CadmiumRate.pdf'))
#    
#    
#    #plot the point of the max rate
#    Cadmium.plot(x= t, logy=True)
#    ylim(0, 0.1)
#    scatter(t2, Y, s = 60)
#
#    ylabel('OD 600')
#    title('OD verus time for Cadmium')
#    savefig(join(sp, 'CadmiumODTimeLogWithRate.pdf'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    #Do these one at a time. Get the cooper. 
#    Copper =g.iloc[:, 0:18]
#
#    #plot the copper
#    Cu = array([20.0/3**i for i in range(9)]*2)   #this is the concentration 
#    Copper.columns = Cu
#    Copper = Copper.sort(axis = 1)  #sort 
#    
##plot the copper 
#    Copper.plot(x=t)
#    ylabel('OD 600')
#    title('OD verus time for Copper')
#    ylim(-0.1, 1.2)
#    savefig(join(sp, 'CopperOD.pdf'))
#    
#    #figure()
#    Copper.plot(x=t, logy = True)
#    ylabel('OD 600')
#    title('log(OD) verus time for Copper')
#    ylim(0.00001, 0.1)
#    xlim(0, 600)
#    savefig(join(sp, 'CopperODLog.pdf'))
#    
#    
#    #get the max rates
#    MaxSlopes,t2 = getMaxRatesDF(t,Copper.apply(log, axis = 0), N=11)
#    figure()
#    
#    scatter(Cu, MaxSlopes)
#    xlim(-0.1, 25)
#    title('Maximum Growth Rate as function of Copper')
#    xlabel('Copper Concentration (uM)')
#    ylabel('Max Rate')
#    savefig(join(sp, 'CopperRate.pdf'))
#        
#    
#    
#    t2, Yvalue=  getTimeToReach(t,Copper, 0.04)
#    #plot above
#    
#    Copper.plot(x=t)
#    scatter(t2, Yvalue, s= 60)
#    
#    ylim(-0.05, 0.1)
#    ylabel('OD 600')
#    title('log(OD) verus time for Copper')
#    savefig(join(sp, 'CopperODWithTimeToReach.pdf'))
#    
#    #plot the time to reach as a funciton of copper concetration
#    figure()
#    scatter(Cu, t2, s= 60)
#    xlim(-0.5, 25)
#    xlabel('Copper Concentration (uM)')
#    ylabel('Time At which OD 0.4 Reached')
#    savefig(join(sp, 'CopperTimeToReach.pdf'))
#    
#    
#    
#    #now Do the Cobalt
#    Cobalt = g.iloc[:,18:36 ]
#    Co = copy(Cu)
#    Cobalt.columns = Co
#    Colbalt = Cobalt.sort(axis = 1)
#    
#    #plot the Max rate and the time to reach
#    
#    MaxSlopes,t2 = getMaxRatesDF(t,Cobalt.apply(log, axis = 0), N=11)
#    figure()
#    
#    scatter(Co, MaxSlopes, s= 60)
#    xlim(-0.1, 25)
#    title('Maximum Growth Rate as function of Cobalt')
#    xlabel('Cobalt Concentration (uM)')
#    ylabel('Max Rate')
#    savefig(join(sp, 'CobaltRate.pdf'))
#    
#    t2, Yvalue=  getTimeToReach(t,Cobalt, 0.04)
#    #plot above
#    figure()
#    scatter(Co, t2, s = 60)
#    xlim(-0.1, 25)
#    title('Time to reach OD 0.04 as function of Cobalt')
#    xlabel('Cobalt Concentration (uM)')
#    ylabel('Time to reach')
#    savefig(join(sp, 'CobaltTime.pdf'))
#    
#    
#    #now Do the Mercury
#    Mercury = g.iloc[:, 36: ]
#    Hg= copy(Cu)/2
#    Mercury.columns = Hg
#    Mercury = Mercury.sort(axis = 1)
#    
#    #plot the Max rate and the time to reach
#    
#    MaxSlopes,t2 = getMaxRatesDF(t,Mercury.apply(log, axis = 0), N=11)
#    figure()
#    
#    scatter(Hg, MaxSlopes, s= 60)
#    xlim(-0.1, 25)
#    title('Maximum Growth Rate as function of Mercury')
#    xlabel('Mercury Concentration (uM)')
#    ylabel('Max Rate')
#    savefig(join(sp, 'MercuryRate.pdf'))
#    
#    t2, Yvalue=  getTimeToReach(t,Mercury, 0.04)
#    #plot above
#    figure()
#    scatter(Hg, t2, s = 60)
#    xlim(-0.1, 25)
#    title('Time to reach OD 0.04 as function of Mercury')
#    xlabel('Mercury Concentration (uM)')
#    ylabel('Time to reach')
#    savefig(join(sp, 'MercuryTime.pdf'))
#    
#    ### plot the mercury with time to reach
#    Mercury.plot(x= t)
#    scatter(t2,Yvalue,  s = 60)
#    ylabel('OD 600')
#    title('log(OD) verus time for Mercury')
#    savefig(join(sp, 'MercuryODWithTimeToReach.pdf'))
#    
#    #okay plot a more limited set for mercury.   #we only want to look at the first 10 points. 
#    Mercury= Mercury.iloc[:, 0:10]
#    Hg = Hg[0:10]
#    MaxSlopes,t2 = getMaxRatesDF(t,Mercury.apply(log, axis = 0), N=11)
#    figure()
#    
#    scatter(Hg, MaxSlopes, s= 60)
#    xlim(-0.1, 25)
#    title('Maximum Growth Rate as function of Mercury')
#    xlabel('Mercury Concentration (uM)')
#    ylabel('Max Rate')
#    savefig(join(sp, 'MercuryRate.pdf'))
#    
#    t2, Yvalue=  getTimeToReach(t,Mercury, 0.04)
#    #plot above
#    figure()
#    scatter(Hg, t2, s = 60)
#    xlim(-0.1, 12)
#    title('Time to reach OD 0.04 as function of Mercury')
#    xlabel('Mercury Concentration (uM)')
#    ylabel('Time to reach')
#
#    savefig(join(sp, 'MercuryTime2.pdf'))
#    
#    ### plot the mercury with time to reach
#    Mercury.plot(x= t)
#    scatter(t2,Yvalue,  s = 60)
#    ylabel('OD 600')
#    title('log(OD) verus time for Mercury')
#    ylim(0, 0.1)
#    savefig(join(sp, 'MercuryODWithTimeToReach2.pdf'))
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    '''
#    g2 = averageByNColumns(g2,2)
#    g2 = g2.iloc[:, 0:4]
#    #g2  =g2.iloc[:,0:1]
#    
#    g2.plot(x=t)
#    ylim(0, 0.1)
#    t2, yvalue = getTimeToReach(t, g2,0.04)
#    scatter(t2,yvalue)
#    
#    
#    
#    
#    
#    
#    #convert the g2 to log (first add small number) 
#    glog = g2 + 0.00000001
#    glog = g2.apply(log, axis = 0)
#    
#    #slopes = getRates(x,y)
#    
#    #MaxSlopes,t2 = getMaxRatesDF(t, g2, N=11)
#    #plot(t2, MaxSlopes)
#    #Plateau,t2 = getPlateau(t, g2)
#    
#    #plot(t, g2)
#    #scatter(t2, Plateau)
#    #plot(t2, slopes)
#    #t2 = t[2:len(t)-2]
#    #plot(t2, slopesDf)
#    #figure()
#    #plot(t, g2)
#    #slopes,t2 = getMaxRatesDF(t, glog)
#    #g2.plot(x=t, logy=True, ylim=(10**-5,1))
#    #scatter(t2, exp(slopes))
#    
#    '''