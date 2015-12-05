import random
import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import operator
from matplotlib import pylab as plt

##function for plotting the importance of the variables. found this code through kaggle scripts
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

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

a=range(train.shape[1])
create_feature_map(a)

##splitting training data & corresponding class labels using train-test-split function
Xtrain,Xcv,Ytrain,Ycv=train_test_split(train,Y,test_size=0.33,random_state=619)

##train a simple xgboost model with basic(mostly default) parameters
##this is just a preliminary model. I won't use this for making predictions
##it just serves to explore the data & find useful (or useless) variables
## P.S - You'll need to do some googling or reading thru Kaggle forums for xgboost installation instructions.
params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.1
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["seed"] = 619
params["max_depth"] = 6
params["eval_metric"] = "logloss"
        
plst = list(params.items())

num_rounds = 10000
xgtrain = xgb.DMatrix(Xtrain,label=Ytrain)
xgcv=xgb.DMatrix(Xcv,label=Ycv)
watchlist = [(xgtrain, 'train'),(xgcv, 'val')]

model1 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

importance = model1.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

##make the variable importance plot, save it as png file. inspecting this file can help us remove irrelevant variables.
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')
