#!/usr/bin/env python
# coding: utf-8

# # Final project of Scientific computation course
# ## Using California Housing DataSet
# #### by Mojtaba Zolfaghari
# ##### STD_ID: 95143017

# ### Import all libraries

# In[75]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer


# ##### Turn on xkcd sketch-style drawing mode. This will only have effect on things drawn after this function is called. For best results, the "Humor Sans" font should be installed: it is not included with matplotlib.

# In[71]:


plt.xkcd()
plt.show()


# ### Preprocessing and preparing the data

# In[3]:


def preprocessing(data):

    # Extract input (X) and output (y) data from the datase

    X = data.iloc[:, :-1].values
    y = data.iloc[:, [-1]].values

    # Handle missing values: 
            # Fill the missing values with the mean of the respective column

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[:, :-1] = imputer.fit_transform(X[:, :-1])
    y = imputer.fit_transform(y)

    # Encode categorical data: 
        # Convert categorical column in the dataset to numerical data

    from sklearn.preprocessing import LabelEncoder
    X_labelencoder = LabelEncoder()
    X[:, -1] = X_labelencoder.fit_transform(X[:, -1])

    return X, y


# ### Plot learning curve

# In[4]:


def plot_learning_curve(model,X_train,y_train, X_val, y_val, test_size=0.2, step_length = 100):

    train_err, val_err = [], []
    for m in range(1, len(X_train), step_length):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_err.append(mean_squared_error(y_train[:m], y_train_predict))
        val_err.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_err), "r-+", label="Training")
    plt.plot(np.sqrt(train_err), "b-", label="Validation")
    plt.legend(loc="upper right")
    plt.xlabel("Size of the training set")
    plt.ylabel("RMSE")
    plt.show()


# ### Execute model hyperparameter tuning and crossvalidation

# In[5]:


def model(pipeline, parameters, X_train, y_train, X, y):
    
    grid_obj = GridSearchCV(estimator=pipeline,
                            param_grid=parameters,
                            cv=3,
                            scoring='r2',
                            verbose=2,
                            n_jobs=1,
                            refit=True)
    grid_obj.fit(X_train, y_train)

    '''Results'''

    results = pd.DataFrame(pd.DataFrame(grid_obj.cv_results_))
    results_sorted = results.sort_values(by=['mean_test_score'], ascending=False)

    print("##### Results")
    print(results_sorted)

    print("best_index", grid_obj.best_index_)
    print("best_score", grid_obj.best_score_)
    print("best_params", grid_obj.best_params_)

    '''Cross Validation'''

    estimator = grid_obj.best_estimator_
    '''
    if estimator.named_steps['scl'] == True:
        X = (X - X.mean()) / (X.std())
        y = (y - y.mean()) / (y.std())
    '''
    shuffle = KFold(n_splits=5,
                    shuffle=True,
                    random_state=0)
    cv_scores = cross_val_score(estimator,
                                X,
                                y.ravel(),
                                cv=shuffle,
                                scoring='r2')
    print("##### CV Results")
    print("mean_score", cv_scores.mean())

    '''Show model coefficients or feature importances'''

    try:
        print("Model coefficients: ", list(zip(list(X), estimator.named_steps['clf'].coef_)))
    except:
        print("Model does not support model coefficients")

    try:
        print("Feature importances: ", list(zip(list(X), estimator.named_steps['clf'].feature_importances_)))
    except:
        print("Model does not support feature importances")

    '''Predict along CV and plot y vs. y_predicted in scatter'''

    y_pred = cross_val_predict(estimator, X, y, cv=shuffle)

    plt.scatter(y, y_pred)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, xmax], [ymin, ymax], "g--", lw=1, alpha=0.4)
    plt.xlabel("True prices")
    plt.ylabel("Predicted prices")
    plt.annotate(' R-squared CV = {}'.format(round(float(cv_scores.mean()), 3)), size=9,
             xy=(xmin,ymax), xytext=(10, -15), textcoords='offset points')
    plt.annotate(grid_obj.best_params_, size=9,
                 xy=(xmin, ymax), xytext=(10, -35), textcoords='offset points', wrap=True)
    plt.title('Predicted prices (EUR) vs. True prices (EUR)')
    plt.show()


# #### Load the data Read the “housing.csv” file from the folder into the program

# In[6]:


data = pd.read_csv('housing.csv')


# #### Print 10 random rows of this data

# In[74]:


data.sample(10)


# In[62]:


plt.hist((data.median_house_value))
plt.show()


# In[9]:


data.hist(bins=50, figsize=(20, 15))
plt.show()


# In[61]:


data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(20,15))
plt.show()


# #### Execute preprocessing

# In[11]:


X, y = preprocessing(data)


# #### Generate polynomial and interaction features.

# In[12]:


poly_features = PolynomialFeatures(degree = 3, include_bias=False)
X_poly = poly_features.fit_transform(X)


# #### Split the dataset: Split the data into 80% training dataset and 20% test dataset

# In[13]:


Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_poly, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


# #### Standardize data: Standardize training and test datasets

# In[14]:


scaler = StandardScaler()
Xp_train = scaler.fit_transform(Xp_train)
Xp_test = scaler.transform(Xp_test)
yp_train = scaler.fit_transform(yp_train)
yp_test = scaler.transform(yp_test)


# ## Building and testing model with different types of Regressors

# ### 1) KNeighborsRegressor 

# In[15]:


knn = KNeighborsRegressor(n_neighbors=1)


# ##### KNeighborsRegressor learning curve

# ##### Pipeline and Parameters for  KNN and  Execute model hyperparameter tuning and crossvalidation

# In[16]:


pipe_knn = Pipeline([('clf', KNeighborsRegressor())])

param_knn = {'clf__n_neighbors':[5, 10, 15, 25, 30]}

model(pipe_knn, param_knn, Xp_train, yp_train, X, y)


# ### 2) SGDRegressor

# In[17]:


sgd = SGDRegressor()


# #### Learning curve for SGDRegressor

# In[18]:


plot_learning_curve(sgd,
                    Xp_train,
                    yp_train.ravel(),
                    Xp_test,
                    yp_test.ravel(),
                    test_size=0.2,
                    step_length = 100)


# ##### Pipeline and Parameters for  SGDRegressor and  Execute model hyperparameter tuning and crossvalidation

# In[19]:


pipe_sgd = Pipeline([('clf', SGDRegressor())])

param_sgd = {'clf__alpha': [0.0001, 0.00001, 0.1, 1, 10]}


model(pipe_sgd, param_sgd, Xp_train, yp_train.ravel(), X, y.ravel())


# ### 3) Polynomial regression

# In[20]:


linear_regression= LinearRegression()


# ##### Learning curve for LinearRegression

# In[21]:


plot_learning_curve(linear_regression,
                    Xp_train,
                    yp_train.ravel(),
                    Xp_test,
                    yp_test.ravel(),
                    test_size=0.2,
                    step_length = 100)


# #### Pipeline and Parameters for Polynomial Regression and Execute model hyperparameter tuning and crossvalidation

# In[22]:


pipe_poly = Pipeline([('clf', LinearRegression())])

param_poly = {}


model(pipe_poly, param_poly, Xp_train, yp_train, X, y)


# ## 4) Ridge Regression

# In[23]:


ridge_regression= Ridge()


# ##### Learning curve for LinearRegression

# In[24]:


plot_learning_curve(ridge_regression,
                    Xp_train,
                    yp_train,
                    Xp_test,
                    yp_test,
                    test_size=0.2,
                    step_length = 100)


# #### Pipeline and Parameters for Ridge and Execute model hyperparameter tuning and crossvalidation

# In[25]:


pipe_ridge = Pipeline([('clf', Ridge())])

param_ridge = {'clf__alpha': [0.01, 0.1, 1, 10]}


model(pipe_ridge, param_ridge, Xp_train, yp_train, X, y)


# ## 5) Lasso Regression

# In[26]:


lasso_regression = Lasso(max_iter=1500)


# ##### Learning curve for Lasso

# In[27]:


plot_learning_curve(lasso_regression,
                    Xp_train,
                    yp_train,
                    Xp_test,
                    yp_test,
                    test_size=0.2,
                    step_length = 100)


# In[28]:


# Pipeline and Parameters - Lasso # Execute model hyperparameter tuning and crossvalidation


# In[29]:


pipe_lasso = Pipeline([('clf', Lasso(max_iter=1500))])

param_lasso = {'clf__alpha': [0.01, 0.1, 1, 10]}


model(pipe_lasso, param_lasso, Xp_train, yp_train, X, y)


# ## 7) Elastic-Net Regression

# In[30]:


elastic_net = ElasticNet()


# ##### Learning curve for ElasticNet

# In[31]:


plot_learning_curve(elastic_net,
                    Xp_train,
                    yp_train,
                    Xp_test,
                    yp_test,
                    test_size=0.2,
                    step_length = 100)


# #### Pipeline and Parameters for Lasso and Execute model hyperparameter tuning and crossvalidation

# In[32]:


pipe_elasticnet = Pipeline([('clf', ElasticNet())])

param_lasso = {'clf__alpha': [0.01, 0.1, 1, 10]}


model(pipe_elasticnet, param_lasso, Xp_train, yp_train, X, y)


# In[ ]:





# ## Using California Housing Dataset in Scikit-learn

# Also scikit-learn comes with wide variety of datasets for regression, classification and other problems. Lets load our data into pandas dataframe and take a look.
# In this part We will work with the California housing dataset in sklearn and perform some different models to predict apartment `prices` based on the ``median income`` in the block.

# #### Load the dataset

# In[33]:


from sklearn.datasets import california_housing

housing_data = california_housing.fetch_california_housing()


# In[34]:


Features = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
Target = pd.DataFrame(housing_data.target, columns=['Target'])
df = Features.join(Target)


# Features as `MedInc` and `Target` were scaled to some degree.

# In[35]:


df.corr()


# ## Preprocessing: Removing Outliers and Scaling

# In[36]:


df[['MedInc', 'Target']].describe()[1:] #.style.highlight_max(axis=0)


# It seems that `Target` has some outliers (as well as `MedInc`), because 75% of the data has price less than 2.65, but maximum price go as high as 5. We're going to remove extremely expensive houses as they will add unnecessary noize to the data.

# In[37]:


df = df[df.Target < 3.5]
df = df[df.MedInc < 8]


# ### Removed Outliers

# In[38]:


df[['MedInc', 'Target']].describe()[1:]


# We will also scale `MedInc` and `Target` variables to [0-1].

# In[39]:


sc = MinMaxScaler()
mid_income = sc.fit_transform(df['MedInc'].values.reshape(df.shape[0],1))
target = sc.fit_transform(df['Target'].values.reshape(df.shape[0],1))


# In[40]:


mid_income.max(), target.max()


# ## Correlation Between Price and Income

# Visually we can determine what kind of accuracy we can expect from the models.

# In[41]:


plt.figure(figsize=(16,6))
plt.rcParams['figure.dpi'] = 227
plt.style.use('seaborn-whitegrid')
plt.scatter(mid_income, target, label='Data', c='#388fd8', s=6)
plt.title('Positive Correlation Between Income and House Price', fontSize=15)
plt.xlabel('Income', fontSize=12)
plt.ylabel('House Price', fontSize=12)
plt.legend(frameon=True, loc=1, fontsize=10, borderpad=.6)
plt.tick_params(direction='out', length=6, color='#a0a0a0', width=1, grid_alpha=.6)
plt.show()


# In[42]:


def plot_regression(X,  y, X_test, y_pred, title, log=None):
    plt.xkcd()
    plt.figure(figsize=(16,6))
    plt.rcParams['figure.dpi'] = 227
    plt.scatter(X, y, label='Data', c='#388fd8', s=6)
    if log != None:
        for i in range(len(log)):
            plt.plot(X, log[i][0]*X + log[i][1], lw=1, c='#caa727', alpha=0.35)
    plt.plot(X_test, y_pred, c='#ff7702', lw=3, label='Regression')
    plt.title(title, fontSize=14)
    plt.xlabel('Income', fontSize=11)
    plt.ylabel('Price', fontSize=11)
    plt.legend(frameon=True, loc=1, fontsize=10, borderpad=.6)
    plt.tick_params(direction='out', length=6, color='#a0a0a0', width=1, grid_alpha=.6)
    plt.show()


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(mid_income, target, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


# ## K-Nearest Neighbors

# In[44]:


knn = KNeighborsRegressor()

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[45]:


print("MSE:",mean_squared_error(y_test, y_pred))
plot_regression(mid_income, target, X_test, y_pred, title="Stochastic gradein descent")


# ## Stochastic gradient descent 

# In[46]:


sgd = SGDRegressor()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)


# In[47]:


print("MSE:",mean_squared_error(y_test, y_pred))
plot_regression(mid_income, target, X_test, y_pred, title="Stochastic gradein descent")


# ## Linear Regression 

# In[48]:


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[49]:


print("MSE:",mean_squared_error(y_test, y_pred))
plot_regression(mid_income, target, X_test, y_pred, title="Linear Regression")


# Result of our model is the regression line. Just by looking at the graph we can tell that data points go well above and beyond our line, making predictions approximate.

# ## Ridge Regression 

# In[50]:


ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)


# In[51]:


print("MSE:",mean_squared_error(y_test, y_pred))
plot_regression( mid_income, target, X_test, y_pred, title="Ridge Regression")


# ## LASSO Regression 

# In[52]:


lasso = Lasso()

lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)


# In[53]:


print("MSE:",mean_squared_error(y_test, y_pred))
plot_regression( mid_income, target, X_test, y_pred, title="LASSO Regression")


# ## ElasticNet Regression

# In[54]:


elastic_net = ElasticNet()

elastic_net.fit(X_train, y_train)
y_pred = elastic_net.predict(X_test)


# In[55]:


print("MSE:",mean_squared_error(y_test, y_pred))
plot_regression( mid_income, target, X_test, y_pred, title="ElasticNet Regression")


# In[60]:


scores = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'F1-score': f1_score
}
for name, scorer in scores.items():
    print(f'The {name} of Bernoulli Naive Bayes is {scorer(y_test, y_pred)}')


# In[64]:


regr = svm.SVR()
regr.fit(X_train, y_train.ravel())
y_pred = regr.predict(X_test)


# In[65]:


print("MSE:",mean_squared_error(y_test, y_pred))
plot_regression( mid_income, target, X_test, y_pred, title="ElasticNet Regression")


# In[ ]:




