import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import statsmodels.api as sm

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("Loading datasets...  ", end = "")
gdf = pd.read_csv("../data/Kaggle Files/12k_unimputed.csv")
tdf = pd.read_csv("../data/Kaggle Files/quantitative_traits.csv")
gdf = gdf.rename(columns = {gdf.columns[0]: "ID"}).set_index("ID").fillna(method = "ffill")
tdf = tdf.rename(columns = {tdf.columns[0]: "ID"}).set_index("ID").fillna(method = "ffill")
gdf = gdf[gdf.index.isin(tdf.index.values)]
print("Done")

take = lambda arr, i: [val[i] for val in arr]

mdf = gdf.merge(tdf, on = "ID")
column_count = len(tdf.columns)
features = mdf.iloc[:, :-column_count]
labels = mdf.iloc[:, -column_count:]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import math

np.seterr(all = "ignore")

def model_p(model, X, y, n_jobs = 1):
    model = model.fit(X, y, n_jobs)
    sse = np.sum((model.predict(X) - y)**2, axis = 0)/float(X.shape[0] - X.shape[1])
    se = np.array([np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X)))) for i in range(sse.shape[0])])
    model.t = model.coef_/se
    model.p = 2*(1 - stats.t.cdf(np.abs(model.t), y.shape[0] - X.shape[1]))
    return model

def manhattan_weights(trait, nweights = 1000):
    x_vals = labels[trait].values.reshape(-1, 1)
    y_vals = features.values
    
    model = LinearRegression()
    model = model_p(model, x_vals, y_vals)

    plt.title(trait)
    plt.ylabel("R^2")
    plt.xlabel("Chromosome")
    coefs = model.coef_.squeeze()
    for i, pval in enumerate(tqdm(coefs)):
        lp = pval**2
        plt.plot(i/1000, lp, color = ["gray", "blue"][math.floor(i/1000) % 2], marker = "o", markersize = 2)
    plt.show()
    
    weights = np.array([features.columns[i] for i in np.argsort(coefs)[-nweights:]])
    return weights

def weighted_regression(trait, sample_weights = features.columns, ntest = 60):
    x_vals = features[features.columns.intersection(sample_weights)].values
    y_vals = labels[trait].values.reshape(-1, 1)
    
    x_train = x_vals[:-ntest]
    y_train = y_vals[:-ntest]
    x_test = x_vals[-ntest:]
    y_test = y_vals[-ntest:]
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    plt.title(trait)
    plt.scatter(predictions, y_test)
    plt.show()
    print("MSE: ", mean_absolute_error(predictions, y_test))
    print("Correlation: ", pd.Series(predictions.squeeze()).corr(pd.Series(y_test.squeeze())))

for trait in tdf.columns:
    sample_weights = manhattan_weights(trait)
    weighted_regression(trait)

model.summary()
