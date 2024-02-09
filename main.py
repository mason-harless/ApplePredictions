import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

#load data into frame
df = pd.read_csv('apple_quality.csv')
#This is a good way to see all of the cols in run window
pd.options.display.width = 0

#take a first look at the data
#print(df.head())
print(df.info())

#note that acidity should be a float, check
def check_obj_num(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


print(df[df['Acidity'].apply(lambda x: not check_obj_num(x))])
#clearly an attribution line
#Brute force method
#for value in df['Acidity']:
    #if (not check_obj_num(value)):
        #print(df[df['Acidity'] == value])


#remove problematic row which happens to be the last row
df.drop(df.tail(1).index, inplace=True)
print(df.info())



