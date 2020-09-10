import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

def preprocess(dataset, test):
    dataset = dataset.fillna(0)
    test = test.fillna(0)
    return dataset, test

def get_result(dataset, test):    #dataset must be received in pandas DataFrame
    dataset, test = preprocess(dataset, test)
    x=dataset.iloc[:, 0:5].values #taking item condition, shipping, review, social_impact, brand value as features
    y=dataset.iloc[:, 5].values   #price is the label
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    onehot = OneHotEncoder()
    onehot.fit_transform(x)
    regressor = xgb.XGBRegressor(n_estimators = 100,
                             max_depth = 5)
    pipeline = make_pipeline(regressor)
    pipeline.fit(x, y)
    for i in range(len(test)):
        pred = np.round(pipeline.predict(test))

    return pred

def rename_cols(test):
    test = pd.DataFrame(test)
    test = test.rename(columns={'0':'Item_Condition', '1':'Shipping', '2':'Review', '3':'Social_Impact', '4':'Brand_Value'})
    return test
    
def better_params(test):
    test = pd.DataFrame(test)
    test[test == 0] = 1
    return test

def predict_price(dataset, test):
    test = pd.DataFrame(test)
    result = get_result(dataset, test)
    return result
    
def show_result(dataset, test, price, df1, better_price):
    test = rename_cols(test)
    plt.plot(test.iloc[0], color='blue', linewidth = 3, 
         marker='o', markerfacecolor='red', markersize=12)
    plt.title('Suggested Price: {}'.format(price))
    plt.show()
    df1 = rename_cols(df1)
    plt.plot(df1.iloc[0], color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)
    plt.title('Suggested Price: {}'.format(better_price))
    plt.show()
