import random
import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import operator
from matplotlib import pylab as plt

##most of this code is similar to trial.py

##read the data
train  = pd.read_csv('train.csv', index_col=0)
test  = pd.read_csv('test.csv', index_col=0)

Y=train.RESIGNED
##in case RESIGNED value is missing, assume he hasn't resigned
whereisnan=Y.isnull()
Y[whereisnan]=0

##drop these columns since they don't exist in test set
train.drop('RESIGNED', axis=1, inplace=True)
train.drop('RESIGNATION_MTH', axis=1, inplace=True)
train.drop('RESIGNATION_QTR', axis=1, inplace=True)
train.drop('RESIGNATION_YEAR', axis=1, inplace=True)
train.drop('RESIGN_DATE', axis=1, inplace=True)
train.drop('STATUS', axis=1, inplace=True)

##for character columns, replace all missing values with the string 'blank'
train.loc[:, train.dtypes == object] = train.loc[:, train.dtypes == object].fillna('blank')
test.loc[:, test.dtypes == object] = test.loc[:, test.dtypes == object].fillna('blank')

##for numeric columns, replace all missing values with 0
train=train.fillna(0)
test=test.fillna(0)

columns = train.columns
test_ind = test.index

##by inspecting the feature importance png figure, i found that these columns contribute very little, & can be eliminated from our data.
##column 2 is not seen in the figure, bcoz it probably has 0 or negative importance
##after this, we've reduced from 50 columns to 36.
droplist=[1,2,5,24,25,26,27,28,32,33,41,42,43,47]
for i in train.columns[droplist]:
    train.drop(i,axis=1,inplace=True)
    test.drop(i,axis=1,inplace=True)

##convert to numpy arrays
train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]) )
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

##splitting training data & corresponding class labels using train-test-split function
Xtrain,Xcv,Ytrain,Ycv=train_test_split(train,Y,test_size=0.33,random_state=619)

##train the xgboost model
##tune the parameters by trial & error to get the least logloss on the CV set
##info of parameters is found here - https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.05
params["gamma"] = 1
params["min_child_weight"] = 2
params["subsample"] = 1
params["colsample_bytree"] = 0.5
params["seed"] = 619
params["scale_pos_weight"] = 2
params["max_depth"] = 10
params["eval_metric"] = "logloss"
        
plst = list(params.items())

num_rounds = 10000
xgtest = xgb.DMatrix(test)
xgtrain = xgb.DMatrix(Xtrain,label=Ytrain)
xgcv=xgb.DMatrix(Xcv,label=Ycv)
xgtrain_full=xgb.DMatrix(train,label=Y)
watchlist = [(xgtrain, 'train'),(xgcv, 'val')]
model1 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

##keep tuning parameters till u seem satisfied with model1.
##apply the same parameters & train again on the full training set
xgb1= xgb.train(plst,xgtrain_full,num_boost_round=model1.best_iteration)

##use xgb1 to predict on test set
preds1=xgb1.predict(xgtest)

##create output file (.csv) for submission        
output = pd.DataFrame({"PERID": test_ind, "RESIGNED": preds1})
output = output.set_index('PERID')
output.to_csv('submission1.csv')

