# Data Preparation for Linear Regression Modeling

In this section, we discuss the data preparation required for training and testing in linear regression modeling.

## Overview

The source data contains independent and dependent variables. To train and test the model, we need to split the data into four sets, namely `X_train`, `y_train`, `X_test`, and `y_test`. 

- The `X_train` and `X_test` arrays represent independent variables.
- The `y_train` and `y_test` arrays represent dependent variables.

We will use column "Avg. Area Income" from the `USA_Housing.csv` dataset as an independent variable and column "Price" as the dependent variable. 

## Data Splitting

We can split the data into these arrays using the code provided or by using an alternative implementation that involves creating a list of predictors and using the `pandas` library to obtain the X and y arrays.

After creating these arrays, we will split them into training and testing datasets. We will use the `train_test_split` method from the `sklearn` library to split the data, with 40% of the data placed in the testing dataset and 60% in the training dataset. A different value for "random_state" can be used to split the data randomly.

```python
import pandas as pd

df_2 = pd.read_csv(r"/C/USA_Housing.csv")
X = df_2[['Avg. Area Income']]
y = df_2['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```

### Linear Regression
We will use Linear Regression modeling to illustrate the process. By employing the sklearn library, we can create a Linear Regression model and fit it on the training data. The predict() function can then be used over the testing dataset to predict quantities. By comparing the predictions with y_test, we can determine the accuracy of the model.

```python
from sklearn.linear_model import LinearRegression

df_2 = LinearRegression()
df_2.fit(X_train, y_train) 
predictions = df_2.predict(X_test)

print('coefficient of determination:', df_2.score(X_train,y_train))
print('intercept:', df_2.intercept_)
print('slope:', df_2.coef_)
```

### Visualization
We can generate a graphical illustration of the training set along with the regression line.

```python
import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color='g')
plt.plot(X_train, df_2.predict(X_train), color='k')
```
