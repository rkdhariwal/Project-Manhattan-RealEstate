import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error, f1_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#import data and create dummy variables
df = pd.read_csv("/Users/raman/Desktop/manhattan.csv") #update to user directory
df = df.drop(['rental_id','borough'], axis=1)
df = pd.get_dummies(df)

#regression summary results
x = df.drop('rent', axis=1)
x = sm.add_constant(x)
y = df['rent']

lm = sm.OLS(y, x).fit()
print(lm.summary())

#perform linear regression
#separating x and y  
x = df.drop(['rent', "rental_id"], axis=1)
y = df['rent']

#test/train the dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

#initialize model
model = LinearRegression()
model.fit(x_train,y_train)

#evaluate the model
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
rmse=mse**0.5

print("R squared is",r2_score(y_test, y_pred))
'''R squared is 0.8056968247998944'''

print("rmse is", rmse)
'''rmse is 1344.552642371391'''

#random forest
rfr=RandomForestRegressor(n_estimators=500)
rfr.fit(x_train, y_train)

#make predictions
y_pred_rfr=rfr.predict(x_test)

#evaluate the model
mse=mean_squared_error(y_test,y_pred_rfr)
rmse=mse**0.5

print("R squared is",r2_score(y_test, y_pred_rfr))
'''R squared is 0.8240863102892507'''

print("rmse score is:", rmse)
'''rmse score is: 1279.3450131933355'''

#grid search to optimize the random forest
parameter_grid = {"max_depth": range (2,16), "min_samples_split": range(2,6)}

grid=GridSearchCV(rfr, parameter_grid, verbose=3, scoring="neg_mean_squared_error")

#grid search 
grid.fit(x_train,y_train)

#best parameters 
grid.best_params_

#random forest with optimized parameters
rfr=RandomForestRegressor(n_estimators=500,max_depth=15,min_samples_split=3)
rfr.fit(x_train,y_train)

#make predictions
y_pred_rfr=rfr.predict(x_test)

#evaluate the model
mse=mean_squared_error(y_test,y_pred_rfr)
rmse=mse**0.5

print("R squared is",r2_score(y_test, y_pred_rfr))
'''R squared is 0.824129259526885'''

print("rmse score is:", rmse)
'''rmse score is: 1279.1888279433365'''

""""
Note: SelectKBest was not included in    
our final report because it did not 
perform as well as random forest.  
"""

#with SelectKBest features          

for i in range(1,47):
    bestfeatures = SelectKBest(score_func=f_regression, k=i)
    new_x = bestfeatures.fit_transform(x,y)
    x_train, x_test, y_train, y_test = train_test_split(new_x,y,test_size=0.3, 
                                                        random_state=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("R squared is", i, r2_score(y_test, predictions))

lm_columns = pd.DataFrame(best, columns = ['best_feature'])
lm_columns['variable'] = col
lm_columns = lm_columns.sort_values(['best_feature','variable'], ascending=[False,True])
lm_columns.index = np.arange(1, 47)
lm_columns[:28]

#perform KMeans cluster
df = pd.read_csv(r'C:\Users\keith\Downloads\manhattan.csv')
df = df.drop(['rental_id','borough', 'neighborhood', 'no_fee', 
              'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 
              'has_patio', 'has_gym', 'bedrooms','bathrooms'], axis=1)
              
km = KMeans(n_clusters=3, random_state=1)
km.fit(df)

df['labels'] = km.labels_

#change cluster labels from 0,1,2 to 1,2,3
df.loc[df['labels'] == 0, 'cluster'] = 1
df.loc[df['labels'] == 1, 'cluster'] = 2
df.loc[df['labels'] == 2, 'cluster'] = 3

df['cluster'] = df['cluster'].astype(int)

sns.scatterplot(x="rent",y="size_sqft", data=df, hue="cluster", palette="Set1")

#elbow method
wcv = []
silk_score = []

for i in range(2,11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(df, km.labels_,))
    
plt.plot(range(2,11), wcv)
plt.xlabel('no. of clusters')
plt.ylabel('within cluster variation')
plt.show()

#pair plot for cluster analysis
sns.pairplot(df,hue='cluster', palette="Set1")

#getting and plotting cluster centers
centroids = km.cluster_centers_

sns.scatterplot(x="rent",y="size_sqft", data=df, hue="cluster", palette="Set1") 
plt.scatter(centroids[:,0] , centroids[:,1] , s = 60, color = 'black')
plt.legend()
plt.show()

center = km.cluster_centers_

df_center = pd.DataFrame(center)
df_center