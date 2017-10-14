# --- update in 2017/10/14 --- #

import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris 
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

def readfile(filename):
    dataset = []
    label = []
    with open(filename) as f:
        for lines in f.readlines()[1:]:
            counter = 1
            temp = []
            for data in lines.strip().split(',')[0:]:
                if counter <= 8:
                    temp.append((float)(data))
                else:
                    label.append((int)(data))
                counter += 1
            dataset.append(temp)
    return dataset, label

# ETC means extra_tree_classification
def ETC(train_X, train_y, test_X, test_y, feature_num, feature_names):
    print "''' ------------- use extratreesclassifier to select the best features ---------------- '''"
    forest = ExtraTreesClassifier(max_features = (int)(m.sqrt(feature_num)), n_estimators=1000, random_state=0, max_depth = None)
    forest.fit(train_X, train_y)
    
    print np.sum(test_y)
    
    #print forest.predict([[1,2,3,4,5,6,7,8]])[0]
    #predict_result = []
    counter = 0
    wrong_cases = []
    for i in range(len(test_X)):
        predict_result = forest.predict([test_X[i]])[0]
        print 'The predict result of the', i + 1, 'th vector is:', predict_result
        if predict_result == test_y[i]:
            print 'The predict result is right!'
            counter += 1
        else:
            print 'The predict result is wrong! The right answer is', test_y[i]
            wrong_cases.append(i)
        #predict_result.append(result)
    
    print 'The accuracy is %f:', counter / len(test_X)
    print 'The wrong cases are', wrong_cases
    
    # cross-validation
    #folds_num = 10
    #print 'The average accuracy of', folds_num, 'folds CV is:', cross_val_score(forest, X, y, cv=folds_num, scoring='accuracy').mean()
    
    
    # --- calculate and output the importance ranking result --- # 
    
    # Calculate the importance of the feature
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    # feature list ranking by importance
    feature_list = []
    for i in range(len(feature_names)):
        feature_list.append(feature_names[indices[i]])
    
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(feature_num):
        print "%d. feature %d %s (%f)" % (f + 1, indices[f], feature_list[f], importances[indices[f]])
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(feature_num), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(feature_num), feature_list)
    plt.xlim([-1, feature_num])
    plt.show()


# GBDT
def GBDT(X, y, feature_num, feature_name):
    print '\n'
    print "''' ------------- use GBDT to select the best features ---------------- '''" 
    gbdt = GradientBoostingClassifier(  
        init = None,  
        learning_rate = 0.1,  
        loss = 'deviance',  
        max_depth = None,  
        max_features = (int)(m.sqrt(feature_num)),  
        max_leaf_nodes = None,  
        min_samples_leaf = 1,  
        min_samples_split = 2,  
        min_weight_fraction_leaf = 0.0,  
        n_estimators = 600,  
        random_state = None,  
        subsample = 1.0,  
        verbose = 0,  
        warm_start = False)  
    
    # Output the X which has already been refined using selectfrommodel
    #print SelectFromModel(gbdt).fit_transform(X, y) # Default: delete the importance-lower-than-mean feature
    
    # Analysis the detailed importance value 
    print "fit start!"  
    gbdt.fit(X, y)  
    print "fit success!"  
      
    score = gbdt.feature_importances_  
    score = preprocessing.MinMaxScaler(np.array(score))
    print "The importance of each feature is as follows:", score
    
    # cross-validation
    folds_num = 10
    print 'The average accuracy of', folds_num, 'folds CV is:', cross_val_score(gbdt, X, y, cv=folds_num, scoring='accuracy').mean()

if __name__ == '__main__':
    # dataset 'iris' from sklearn
    iris = load_iris()
    X = Normalizer().fit_transform(iris.data) # Normalize the data, accuracy 0.95/0.96 ---> 0.98, nice!
    y = iris.target
    feature_num = len(iris.feature_names)
    
    train_X, train_y = readfile("D:\\lifu\\pycode\\MRNPC\\trainingdata\\0.csv")
    train_X = Normalizer().fit_transform(train_X) 
    test_X, test_y = readfile("D:\\lifu\\pycode\\MRNPC\\trainingdata\\1.csv")
    test_X = Normalizer().fit_transform(test_X) 
    feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']

    ETC (train_X, train_y, test_X, test_y, 8, feature_names)
    #GBDT(X, y, feature_num)




