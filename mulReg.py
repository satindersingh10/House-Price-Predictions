# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# Loading the data
from sklearn.datasets import load_boston
boston_data = load_boston()

# Creating the dataframe
data = pd.DataFrame(data = boston_data.data, columns = boston_data.feature_names)
data['PRICE'] = boston_data.target

# Visualizing the data with MatplotLib
plt.hist(data['PRICE'], bins = 10, color = 'blue')
plt.xlabel('Price of houses in 000s')
plt.ylabel('Number of houses')
plt.show()

# Visualizing the data with Seaborn
#sns.distplot(data['PRICE'])
#plt.show()

# Visualizing acces to highway index
frequency = data['RAD'].value_counts()
plt.bar(frequency.index, height= frequency)
plt.xlabel('Access to Highways')
plt.ylabel('NO of Houses')
plt.show()

# Finding correlation between price and no. of rooms in houses
print(data['PRICE'].corr(data['RM']))
mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
sns.heatmap(data.corr(), mask = mask, annot =True)
plt.show()

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['TAX'], y= data['RAD'], color = 'indigo', joint_kws{'alpha':0.5})
plt.show()

sns.pairplot(data, kind = 'reg')
plt.show()

# Splitting in training and test dataSET
prices = data['PRICE']
features = data.drop('PRICE', axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 0)

# Fitting the model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train, y_train)


# Printing R square values of training and test dataset
print("Training Data R-Square value: ", regr.score(X_train, y_train))
print("Test Data R-Square value: ", regr.score(X_test, y_test))

print('Intercept', regr.intercept_)
pd.DataFrame(data = regr.coef_, index = X_train.columns)

# Data Transformation
y_log = np.log(data['PRICE'])
y_log.skew()
sns.distplot(y_log)
plt.show()

transferred_data = features
transferred_data['LOG PRICES'] = y_log
sns.lmplot(x = 'LSTAT', y= 'PRICE', data = data, scatter_kws={'alpha':0.6}, line_kws={'color': 'cyan'})
plt.show()

# Regression using log prices

prices = np.log(data['PRICE'])
features = data.drop('PRICE', axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train, y_train)
print('Intercept', regr.intercept_)
pd.DataFrame(data = regr.coef_, index = X_train.columns)

# P value & Evaluating coefficiants

import statsmodels.api as sm
X_incl = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl)
results = model.fit()
pd.DataFrame({'coef': results.params, 'P value': results.pvalues})

# Testing for multi colinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor


























