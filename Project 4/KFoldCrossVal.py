from random import seed, randrange, shuffle
import pandas as pd
import numpy as np
import copy
from itertools import combinations
seed(.7)

def KFoldCrossVal(dataset,folds=5):
    """
    Here is a function for us to seperate our dataframe into KFolds. The function returns
    a list containg subsets of our datframe. Each subset in the list will contain 
    rowsInDataFrame/KFold number of entires.
    
    To test accuracy with this method we need to take the accuracy of our model on each subset and find the mean.
    
    Parameters
    ----------
    dataset : dataframe
        This is our dataframe
    folds : Integer (defaults to 5)
        DESCRIPTION. This is our number of folds (subsets) of the data that we will create.
        Big data should use 10 folds, small data use 3. We will use 5 because it makes it easiest for us to get the 80%-20% ratio of train to test

    Returns
    -------
    dataset_split : List containing each fold
        This is a list. Each entry in the list represents a subset of our data.
        If we had K=5, we would expect this list to have 5 entries.
        The size of each entry is totalRows/KFolds
    """
    
    #This is so that we can handle pandas dataframes
    dataset = dataset.values.tolist()
    #create somewhere to store all the subsets
    dataset_split = []
    #randomly shuffle the dataset. changing our random seed outside of the function will
        #change the way the dat gets shuffled and ultimately our results
    shuffle(dataset)
    #get a list of the dataset
    dataset_copy = list(dataset)
    #Calc the num of rows in each subset
    fold_size = int(len(dataset)/folds)
    #for each fold
    for i in range(folds):
        #create somewhere to save the current fold
        fold = []
        #While the fold is not bigger than the num of rows needed in each subset
        while len(fold) < fold_size:
            #append a random row to the current subset
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        #before moving to the next fold, append this fold to our list of all folds
        dataset_split.append(fold)
    return dataset_split
# Method modeled off of https://machinelearningmastery.com/implement-resampling-methods-scratch-python/

def foldsToTrainAndTest(df,f=5):
    """
    This function takes our folds created from KFoldCrossVal and returns a list that contains 5 entries. Note that this function assumes we are using 5 folds. Each entry in this list is a tuple. Each tuple contains two things. The first being the the 80% of data we will use to train this iteration to create the weights. The second entry of the tuple contains the 20% of the data we will use to test the current iteration.
    Input: dataframe
    Output: 
        a list that contains 5 tuples, each of which contain: the 4 folds that will be used in training the data for the current iteration AND the 1 fold that will be sued for testing for the current iteration 
        AND
        20% of the data that will not be used in K fold cross Val. THis 20% will be used at the very end so that we can see the accuracy of our average weights after the 5 iterations using K Fold
    """
    folds = KFoldCrossVal(df,f)
    dfcombo = list(combinations(folds, f-1))
    Mastercombo = []
    for combo in dfcombo:
        tempfold = copy.deepcopy(folds)
        for df in combo:
            tempfold.remove(df)
        Mastercombo.append((np.vstack(combo),np.array(tempfold[0])))
    
    return Mastercombo