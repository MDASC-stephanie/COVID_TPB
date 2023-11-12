import numpy as np
import matplotlib.pyplot as plt
  
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)
  
def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
  
    # predicted response vector
    y_pred = b[0] + b[1]*x
  
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
  
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
  
    # function to show plot
    plt.show()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
def regression(df):
    X1 = pd.DataFrame(np.c_[df['s1'], df['s2'],df['s3'], df['s4'],df['s5'], df['s6'], df['s7']], columns=['s1','s2','s3','s4','s5','s6','s7'])
    Y1 = df['y1']
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size = 0.2, random_state=9)
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    
    lin_reg_mod1 = LinearRegression()
    lin_reg_mod1.fit(X_train, y_train)
    pred1 = lin_reg_mod1.predict(X_test)
    from sklearn.metrics import r2_score
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred1)))

    test_set_r2 = r2_score(y_test, pred1)
    Adj_r2 = 1 - (1-test_set_r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    print(test_set_rmse)
    print(test_set_r2)
    print(Adj_r2)
    X2 = pd.DataFrame(np.c_[df['s1_I'], df['s2_I'],df['s3_I'], df['s4_I'],df['s5_I'], df['s6_I'], df['s7_I']], columns=['s1_I','s2_I','s3_I','s4_I','s5_I','s6_I','s7_I'])
    Y1 = df['y1']
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y1, test_size = 0.2, random_state=9)
    X_train_fs2, X_test_fs2, fs2 = select_features(X_train2, y_train2, X_test2)
    for i in range(len(fs2.scores_)):
        print('Feature %d: %f' % (i, fs2.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs2.scores_))], fs2.scores_)
    pyplot.show()
    lin_reg_mod2 = LinearRegression()
    lin_reg_mod2.fit(X_train2, y_train2)
    pred2 = lin_reg_mod2.predict(X_test2)
    test_set_rmse2 = (np.sqrt(mean_squared_error(y_test2, pred2)))
    test_set_r22 = r2_score(y_test2, pred2)
    Adj_r22 = 1 - (1-test_set_r22) * (len(y_test2)-1)/(len(y_test2)-X_test2.shape[1]-1)
    print(test_set_rmse2)
    print(test_set_r22)
    print(Adj_r22)
    X3 = pd.DataFrame(np.c_[df['s1_I'], df['s2_I'],df['s3_I'], df['s4_I'],df['s5_I'], df['s6_I'], df['s7_I'],df['s1'], df['s2'],df['s3'], df['s4'],df['s5'], df['s6'], df['s7']], columns=['s1_I','s2_I','s3_I','s4_I','s5_I','s6_I','s7_I','s1','s2','s3','s4','s5','s6','s7'])
    Y1 = df['y1']
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, Y1, test_size = 0.2, random_state=9)
    X_train_fs3, X_test_fs3, fs3 = select_features(X_train3, y_train3, X_test3)
    for i in range(len(fs3.scores_)):
        print('Feature %d: %f' % (i, fs3.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs3.scores_))], fs3.scores_)
    pyplot.show()
    lin_reg_mod3 = LinearRegression()
    lin_reg_mod3.fit(X_train3, y_train3)
    pred3 = lin_reg_mod3.predict(X_test3)
    test_set_rmse3 = (np.sqrt(mean_squared_error(y_test3, pred3)))
    test_set_r23 = r2_score(y_test3, pred3)
    Adj_r23 = 1 - (1-test_set_r23) * (len(y_test3)-1)/(len(y_test3)-X_test3.shape[1]-1)
    print(test_set_rmse3)
    print(test_set_r23)
    print(Adj_r23)
df = pd.read_csv ('C:/Users/mancl/Downloads/regressionInputwithTPBRebuilt_8.csv')
df_uk=df[df['country']=='United Kingdom']
df_uk.head()
regression(df_uk)

df_usa=df[df['country']=='United States']
df_usa.head()
regression(df_usa)

df_can=df[df['country']=='Canada']
df_can.head()
regression(df_can)

df_aus=df[df['country']=='Australia']
df_aus.head()
regression(df_aus)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
def regression_selection(df):
    # load the dataset
    X= pd.DataFrame(np.c_[df['I1'], df['I2'],df['I3'], df['I4'],df['I5'], df['I6'], df['I7'],df['I8'], df['I9'],df['I10'], df['I11'],df['I12'], df['I13'], df['I14'],df['I15'], df['I16'],df['I17'], df['I18'],df['I19'], df['I20'], df['I21'],df['I22'], df['I23'], df['I24']], columns=['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10',
    'I11','I12','I13','I14','I15','I16','I17','I18','I19','I20','I21','I22','I23','I24'])
    Y= df['y']
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, yhat)
    test_set_r = r2_score(y_test, yhat)
    Adj_r2 = 1 - (1-test_set_r) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    print(test_set_r)
    print(Adj_r2)
    print('MAE: %.3f' % mae)
    
dfusa = pd.read_csv ('C:/Users/mancl/Downloads/DASC7600 data/BigMacIndexUSA.csv')
dfusa.head()
regression_selection(dfusa)

dfcan = pd.read_csv ('C:/Users/mancl/Downloads/DASC7600 data/BigMacIndexCAN.csv')
dfcan.head()
regression_selection(dfcan)

dfgbp = pd.read_csv ('C:/Users/mancl/Downloads/DASC7600 data/BigMacIndexGBP.csv')
dfgbp.head()
regression_selection(dfgbp)

dfaus = pd.read_csv ('C:/Users/mancl/Downloads/DASC7600 data/BigMacIndexAUS.csv')
dfaus.head()
regression_selection(dfaus)

import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

import statsmodels.formula.api as smf
import seaborn as sns
from math import sqrt
def LinearMixed(df):
    model = smf.mixedlm("y1 ~ SN + A + PBC+ O+C(s1_I)+C(s2_I)+C(s3_I)+C(s4_I)+C(s5_I)+C(s6_I)+C(s7_I)",df,groups=df["country"],re_formula="~1+SN + A + PBC+ O+C(s1_I)+C(s2_I)+C(s3_I)+C(s4_I)+C(s5_I)+C(s6_I)+C(s7_I)")
    mdf = model.fit(method=['bfgs'])
    print(mdf.summary())
    
    model2 = smf.mixedlm("y1 ~ C(s1_I)+C(s2_I)+C(s3_I)+C(s4_I)+C(s5_I)+C(s6_I)+C(s7_I)+s1_SN+s2_SN+s3_SN+s4_SN+s5_SN+s6_SN+s7_SN+s1_A+s2_A+s3_A+s4_A+s5_A+s6_A+s7_A+s1_PBC+s2_PBC+s3_PBC+s4_PBC+s5_PBC+s6_PBC+s7_PBC+s1_O+s2_O+s3_O+s4_O+s5_O+s6_O+s7_O", df,groups=df["country"]).fit()

    print(model2.summary())
    model3 = smf.mixedlm("y1 ~ SN + A + PBC+ O+C(s1_I)+C(s2_I)+C(s3_I)+C(s4_I)+C(s5_I)+C(s6_I)+C(s7_I)",df,groups=df["country"])
    mdf2 = model3.fit(method=['bfgs'])
    print(mdf2.summary())
    
    model4 = smf.mixedlm("y1 ~ C(s1_I)+C(s2_I)+C(s3_I)+C(s4_I)+C(s5_I)+C(s6_I)+C(s7_I)+s1_SN+s2_SN+s3_SN+s4_SN+s5_SN+s6_SN+s7_SN+s1_A+s2_A+s3_A+s4_A+s5_A+s6_A+s7_A+s1_PBC+s2_PBC+s3_PBC+s4_PBC+s5_PBC+s6_PBC+s7_PBC+s1_O+s2_O+s3_O+s4_O+s5_O+s6_O+s7_O", df,groups=df["country"],re_formula="~1+C(s1_I)+C(s2_I)+C(s3_I)+C(s4_I)+C(s5_I)+C(s6_I)+C(s7_I)+s1_SN+s2_SN+s3_SN+s4_SN+s5_SN+s6_SN+s7_SN+s1_A+s2_A+s3_A+s4_A+s5_A+s6_A+s7_A+s1_PBC+s2_PBC+s3_PBC+s4_PBC+s5_PBC+s6_PBC+s7_PBC+s1_O+s2_O+s3_O+s4_O+s5_O+s6_O+s7_O").fit()

    print(model4.summary())
    
    LinearMixed(df)

    from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def CNN(df):
    X1 = pd.DataFrame(np.c_[df['s1'], df['s2'],df['s3'], df['s4'],df['s5'], df['s6'], df['s7']], columns=['s1','s2','s3','s4','s5','s6','s7']).to_numpy()
    Y1 = df['y1'].to_numpy() 
    x, y = X1,Y1
    print(x.shape)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    print(x.shape)

    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(7,1)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)

    ypred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    test_set_r = r2_score(ytest, ypred)
    Adj_r2 = 1 - (1-test_set_r) * (len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)
    print("MSE: %.4f" % mean_squared_error(ytest, ypred))
    print(test_set_r)
    x_ax = range(len(ypred))
    plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()
def CNN2(df):
    X1 = pd.DataFrame(np.c_[df['s1_I'], df['s2_I'],df['s3_I'], df['s4_I'],df['s5_I'], df['s6_I'], df['s7_I']], columns=['s1_I','s2_I','s3_I','s4_I','s5_I','s6_I','s7_I']).to_numpy()
    Y1 = df['y1'].to_numpy() 
    x, y = X1,Y1
    print(x.shape)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    print(x.shape)

    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(7,1)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)

    ypred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    test_set_r = r2_score(ytest, ypred)
    Adj_r2 = 1 - (1-test_set_r) * (len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)
    print("MSE: %.4f" % mean_squared_error(ytest, ypred))
    print(test_set_r)

    x_ax = range(len(ypred))
    plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()
def CNN3(df):
    X1 = pd.DataFrame(np.c_[df['s1_I'], df['s2_I'],df['s3_I'], df['s4_I'],df['s5_I'], df['s6_I'], df['s7_I'],df['s1'], df['s2'],df['s3'], df['s4'],df['s5'], df['s6'], df['s7']], columns=['s1_I','s2_I','s3_I','s4_I','s5_I','s6_I','s7_I','s1','s2','s3','s4','s5','s6','s7']).to_numpy()
    Y1 = df['y1'].to_numpy() 
    x, y = X1,Y1
    print(x.shape)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    print(x.shape)

    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(14,1)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)

    ypred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    test_set_r = r2_score(ytest, ypred)
    Adj_r2 = 1 - (1-test_set_r) * (len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)
    print("MSE: %.4f" % mean_squared_error(ytest, ypred))
    print(test_set_r)
    x_ax = range(len(ypred))
    plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()
def CNN4(df):
    X1 = pd.DataFrame(np.c_[df['s1_I'], df['s2_I'],df['s3_I'], df['s4_I'],df['s5_I'], df['s6_I'], df['s7_I'],df['SN'], df['PBC'],df['A'], df['O']], columns=['s1_I','s2_I','s3_I','s4_I','s5_I','s6_I','s7_I','SN','PBC','A','O']).to_numpy()
    Y1 = df['y1'].to_numpy() 
    x, y = X1,Y1
    print(x.shape)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    print(x.shape)

    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(11,1)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)

    ypred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    test_set_r = r2_score(ytest, ypred)
    Adj_r2 = 1 - (1-test_set_r) * (len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)
    print("MSE: %.4f" % mean_squared_error(ytest, ypred))
    print(test_set_r)

    x_ax = range(len(ypred))
    plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()
def CNN5(df):
    X1 = pd.DataFrame(np.c_[df['s1_I'], df['s2_I'],df['s3_I'], df['s4_I'],df['s5_I'], df['s6_I'], df['s7_I'],df['s1_SN'], df['s2_SN'],df['s3_SN'], df['s4_SN'],df['s5_SN'],df['s6_SN'],df['s7_SN'],df['s1_PBC'], df['s2_PBC'],df['s3_PBC'], df['s4_PBC'],df['s5_PBC'],df['s6_PBC'],df['s7_PBC'],df['s1_A'], df['s2_A'],df['s3_A'], df['s4_A'],df['s5_A'],df['s6_A'],df['s7_A'],df['s1_O'], df['s2_O'],df['s3_O'], df['s4_O'],df['s5_O'],df['s6_O'],df['s7_O']], columns=['s1_I','s2_I','s3_I','s4_I','s5_I','s6_I','s7_I','s1_SN','s2_SN','s3_SN','s4_SN', 's5_SN','s6_SN','s7_SN', 's1_PBC','s2_PBC','s3_PBC','s4_PBC', 's5_PBC','s6_PBC','s7_PBC', 's1_A','s2_A','s3_A','s4_A', 's5_A','s6_A','s7_A', 's1_O','s2_O','s3_O','s4_O', 's5_O','s6_O','s7_O']).to_numpy()
    Y1 = df['y1'].to_numpy() 
    x, y = X1,Y1
    print(x.shape)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    print(x.shape)

    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(35,1)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)

    ypred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    test_set_r = r2_score(ytest, ypred)
    Adj_r2 = 1 - (1-test_set_r) * (len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)
    print("MSE: %.4f" % mean_squared_error(ytest, ypred))
    print(test_set_r)

    x_ax = range(len(ypred))
    plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()
 
