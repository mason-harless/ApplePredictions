import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats
#from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

#IMPORT AND VIEW DATA

#load data into frame
df = pd.read_csv('apple_quality.csv')
#This is a good way to see all of the cols in run window
pd.options.display.width = 0

#take a first look at the data
#print(df.head())
#note that data is already scaled
print(df.info())

#CLEAN DATA

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

#remove problematic row which happens to be the last row, convert Acidity to float
df.drop(df.tail(1).index, inplace=True)
df['Acidity'] = df['Acidity'].astype(float)


#print(df.head())

#make the quality column binary (see the multiple methods in scratch)
#use a pd.Series method, is the fastest
df['Quality'] = pd.Series(np.where(df['Quality'].values == 'good', 1, 0), df.index)
#print(df.head())
print(df.info())

#get rid of the 'A_id' column
df.drop('A_id', inplace=True, axis=1)
#print(df.head())

#EXPLORATORY

#view initial metrics
print(df.describe().T)

#show bar charts for each, don't need quality
cols = cols = df.columns.tolist()[: -1]
plt.figure(figsize=(10, 10))
plt.style.use('ggplot')

def univar_histograms(column):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=column, kde=True, bins=30, stat='density')

for i, column in enumerate(cols, 1):
    univar_histograms(column)

plt.tight_layout()
plt.show()

#view boxplots, probably not necessary since already scaled
plt.figure(figsize=(10, 10))

def univar_box(column):
    plt.subplot(3,3, i)
    sns.boxplot(y=df[column])

for i, column in enumerate(cols, 1):
    univar_box(column)
plt.tight_layout()
plt.show()

#view qqplots for normality
#this is new for me, never used library before
#need to learn more about how to format more usefully,did the redneck way to be able to see anything

def qqplots(column):
    #plt.subplot(3, 3, i)
    pg.qqplot(df[column], dist='norm')
    plt.title(column)

for i, column in enumerate(cols, 1):
    qqplots(column)
    plt.tight_layout()
    plt.show()

#skewness
print(F'\nSkewness using Fisher-Pearson Coefficient (should be near zero for noraml)')
for column in cols:
    print(F"{column} Skewness: {round(stats.skew(df[column]), 5)}")

#kurtosis
print(F'\nKurtosis using Pearson (vs. normal), should be near 3 for normal')
for column in cols:
    print(F"{column} Kurtosis: {round(stats.kurtosis(df[column], fisher=False), 5)}")

#normality test
print(F'\np-value for normality (> .05 to be normal')
for column in cols:
    pval = round(stats.normaltest(df[column])[1], 5)
    if pval > .05:
        print(F'Is normal, pvalue is: {pval}')
    else:
        print(F'Is not normal, pvalue is: {pval}')

#look at imbalance of target
#is there a reason this would be a bad way to do this?
#maybe if i made a mistake and there were columns that weren't 0 or 1 it could be bad idk
#good_quality = len(df[df['Quality'] == 1])
#bad_quality = len(df[df['Quality'] == 0])

#probably better, could also use the .eq(0) and eq(1)
print(df['Quality'].value_counts())
good_q =(df['Quality'] == 1).sum()
bad_q = (df['Quality'] == 0).sum()

#no need to manage, is clearly balanced enough, could use just accuracy, but will use all majors just for practice
print(F'The ratio of Good Quality to Bad quality is: {round(float(good_q / bad_q), 5)}')







#PREPROCESSING


















#IPMPLEMENTATION AND TESTING



#POSTPROCESSING / ADJUSTING

print("\nEND")