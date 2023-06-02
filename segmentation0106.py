""" Imports and information about the variables:
Variable description including type, range and full description:

Variable	Data Type	Range	Description
ID	numerical	Integer	Shows a unique identificator of a customer.
Sex	categorical	{0,1}	Biological sex (gender) of a customer. 0 = male / 1 = female
Marital status	categorical	{0,1}	Marital status of a customer. 0 = single / 1 = non-single
Age	numerical	Integer	The age of the customer in years, calculated as current year minus the year of birth of the customer at the time of creation of the dataset (Min. age = 18 / Max. age = 78)
Education	categorical	{0,1,2,3}	Level of education of the customer. 0=no education / 1=high-school / 2=university / 3=graduate
Income	numerical	Real	Self-reported annual income in US dollars of the customer.
Occupation	categorical	{0,1,2}	Category of occupation of the customer. 0=unemployed / 1=employee/oficial / 2=management or self-employed
Settlement size	categorical	{0,1,2}	The size of the city that the customer lives in. 0=small / 1=mid-size / 2=big"""


# import all necessary libraries
import pandas as pd # for dat manipulation data
import numpy as np # for numerical calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # advance data visualization
import scipy.stats as stats # for statistical analysis
from sklearn.preprocessing import MinMaxScaler # to scale the data
from sklearn.cluster import KMeans # to apply kmeans clustering 
#%matplotlib inline
from sqlalchemy import create_engine # for connecting the data base
pd.set_option('display.max_columns', None)
# Loading the dataset

customer = pd.read_csv(r"C:\Users\DELL\Downloads\segmentation data.csv")
pwd = '9963973155' # password
user = 'root' # user name
db = 'customer' # data base
# create the database connection 
conn = create_engine(f'mysql+pymysql://{user}:{pwd}@localhost/{db}' ) # create connection to connect the data base
customer.to_sql('customer_info', conn, if_exists = 'replace', chunksize = 1000) # push the data into mysql database
sql = 'select* from customer_info' # query to retrive the data from data base
customer1 = pd.read_sql_query(sql, con = conn) # import data form mysql data base

customer.sample(10) # shows sample of records form the data
customer.info() # shows the summary of the data
num_columns = ['ID', 'Age','Income'] # numeric columns 
customer.columns # shows the all the column in the data
cat_columns = [  'Sex', 'Marital status',  'Education', 
       'Occupation', 'Settlement size'] # separate the categorical columns
customer[cat_columns] = customer[cat_columns].astype('str') # convert the intiger data type into the objective
customer.describe(include = 'object').T # statistical summary of the categorical columns 
customer.describe().T # statistical summary of the numerical columns
customer.isna().sum() # check the null values in data set 

# Exploratory data analysis 
import sweetviz # Auto EDA
report = sweetviz.analyze(customer)
report.show_html('sweetviz_report.html')

# histogram for shows the distribution of the data
for numerical in num_columns:
    plt.figure(figsize = (8, 4))
    sns.histplot(data = customer, x = numerical)
    plt.show()

# distribution of categorical varibales
for category in cat_columns:
    plt.figure(figsize = (8, 4))
    sns.countplot(data = customer, x = category)
    plt.show()
#Bivariate Analysis 
sns.scatterplot(x = 'Age', y = 'Income', data = customer)
plt.show()
customer.corr()

import scipy.stats as stats # for statistical analysis 
print(stats.pearsonr(customer['Age'], customer['Income'])) # find the corelation between age and income

# scatter plot for find the linear realation with slope
sns.lmplot(x = 'Age', y = 'Income', data = customer)
plt.show()
# Desity plots 
# Categorical Vs Numerical
for category in cat_columns:
    for numerical in num_columns:
        if numerical != 'ID':
            plt.figure(figsize =(8, 4))
            sns.kdeplot(data = customer, x = numerical, hue = category)
            plt.show()
# categorical Vs Categorical 
for category1 in cat_columns:
    for category in cat_columns:
        if category1 != category:
            plt.figure(figsize = (8, 4))
            sns.countplot(data = customer, x = category1, hue = category)
            plt.show()
#Multivariate Analysis
def bivariate_scatter(x,y, hue, df):
    plt.figure(figsize = (6,6))
    sns.scatterplot(x = x , y = y, data = df, hue = hue, alpha = 0.85 )
    plt.show()
for column in cat_columns:
    bivariate_scatter('Age', 'Income', column, customer)

from scipy import stats  # for statistical analysis
normal_test_result_income = stats.normaltest(customer['Income'])
print(normal_test_result_income) # p_value is less then the 0.05 so the data is not normal 
normaltest_result_age = stats.normaltest(customer['Age'])
print(normaltest_result_age) # p_value is less than 0.05 so the data is not normal

# Transformation
def apply_log(column):
    return np.log(column)
def normality_test(column):
    return stats.normaltest(column)

log_income = apply_log(customer['Income'])
normality_test(log_income) # p_value is less than the 0.05 so the data is not normal

log_age = apply_log(customer['Age'])
normality_test(log_age) # p_value is less than the 0.05 so the data is not normal


# power Transformer
from sklearn.preprocessing import PowerTransformer
# power Transform data
feature = customer['Income'].to_numpy().reshape(-1,1)
feature.shape
power_tr = PowerTransformer()
feature_transf = power_tr.fit_transform(feature)
array_1d = feature_transf.flatten()
feature = pd.Series(data = array_1d, index = list(range(len(array_1d))))
normality_test(feature)
# Create axis form original data plot (ax1) and transformed data (ax2)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,4));

# Plot original data & get metrics
customer['Income'].plot(kind='hist', ax=ax1)
ax1.title.set_text('Original data')
norm_test1 = normality_test(customer['Income'])

# Plot log transformed data & get metrics
log_income.plot(kind='hist', ax=ax2);
ax2.title.set_text('Log Transformed data')
norm_test2 = normality_test(log_income)

# Plot power transformed data & get metrics
feature.plot(kind='hist', ax=ax3);
ax3.title.set_text('PowerTransformed data')
norm_test3 = normality_test(feature)
plt.show()
print(norm_test1, norm_test2, norm_test3)

# Create a DataFrame that shows normality test results form each transformation
norm_results = [norm_test1, norm_test2, norm_test3]
metrics = pd.DataFrame(norm_results, index = ['Orginal data', 'Log transform', 'PowerTransform'])
metrics
# Power Transform data on Age 
feature2 = customer['Age'].to_numpy().reshape(-1,1)
power_tr = PowerTransformer()
feature_transf = power_tr.fit_transform(feature2)
array_1d = feature_transf.flatten()
feature2 = pd.Series(data = array_1d , index = list(range(len(array_1d))))
# Log Transform data
log_trans_age = apply_log(customer['Age'])
# Create axis for original data plot (ax1) and transfromed data (ax2)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 4))
# plot orginal data & get mertrics 
customer['Age'].plot(kind = 'hist', ax = ax1)
ax1.title.set_text('Original data')
norm_test1 = normality_test(customer['Age'])

# Plot log transformed data & get metrics
log_trans_age.plot(kind='hist', ax=ax2);
ax2.title.set_text('Log Transformed data')
norm_test2 = normality_test(log_trans_age)

# Plot power transformed data & get metrics
feature2.plot(kind='hist', ax=ax3);
ax3.title.set_text('PowerTransformed data')
norm_test3 = normality_test(feature2)
plt.show()

# Create DataFrame that shows normality test results form each transformation
norm_results = [norm_test1, norm_test2, norm_test3]
metrics = pd.DataFrame(norm_results, index = ['Orginal data', 'Log Transform', 'PowerTransform'] )
metrics
"""After running the tests, we notice that the data isn't normally distributed yet, 
so neither the log transformation or the PowerTransformer were able to get it to a full normal distribution. 
Even though we get data that still isn't normally distributed, it has improved significantly from the initial tests.
 This means that our transformed data is a better approximation to normally distributed data than the original data,
 so we will use this transformed data instead.

For the 'Age' feature, we transform the data using the Log Transform method with our function
For the 'Income' feature, we use the PowerTransformer from scikit learn"""

# Take the Transformed feature Drop the orginal feature create Transformed data Frame
customer['transf_income'] = feature
customer['trans_age'] = log_trans_age
customer
customer_transformed = customer.drop(['Income', 'Age', 'ID'],axis = 1)
customer_transformed
# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(customer_transformed)
print(X)

# Select the number of clusters using Elbow method
from sklearn.cluster import KMeans
clusters_range = list(range(2,19))
inertias = []
for c in clusters_range:
    kmeans = KMeans(n_clusters = c, random_state = 0).fit(X)
    inertias.append(kmeans.inertia_)
inertias
# elbow curve for selecting the clustes
plt.figure(figsize = (7, 7))
plt.plot(clusters_range, inertias, marker = 'o')
plt.show()
"""When running the loop shown above, we notice that the elbow happens around 6-7 clusters,
 which would be a good approximation. The curve isn't very clear and you could also say that 12 is a good number as well,
   but you should understand that 12 clusters is generally too much, so we would rather lose some information about the groups our customers belong to,
     than gaining more accuracy in the clustering used.

Even though we have a somewhat convincing result above, we will use the Silhouette scores to see if we can gain more insight on how many clusters should we use. 
See the procedure below:"""

from sklearn.metrics import silhouette_samples, silhouette_score
clusters_range = range(2, 20)
random_range = range(0, 20)
results = []
for c in clusters_range:
    for r in random_range:
        clusterer = KMeans(n_clusters = c, random_state = r)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        results.append([c, r, silhouette_avg])
result = pd.DataFrame(results, columns = ['n_clusters', 'seed', 'silhouette_score'])
pivot_km = pd.pivot_table(result, index = 'n_clusters', columns = 'seed', values = 'silhouette_score')

plt.figure(figsize = (15, 6))
sns.heatmap(pivot_km, annot = True, linewidths = 0.5, fmt = '.3f', cmap = sns.cm.rocket_r)
plt.show()
"We decide to create 6 and 7 clusters and use our business understanding to determine which classification provides more insights about the customers"

# Create a 3D data set with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3, random_state = 3)
X_pca = pca.fit_transform(X)
X_pca_df = pd.DataFrame(data = X_pca, columns =['X1', 'X2', 'X3'])
# applying K_means with no of clusters 6 and visualizing the results with PCA decompostion
kmeans = KMeans(n_clusters = 6, random_state = 0).fit(X)
labels = kmeans.labels_
X_pca_df['Labels'] = labels

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
X_pca_df.head()

X_pca_df['Labels'] = X_pca_df['Labels'].astype(str)
import plotly.express as px
fig = px.scatter_3d(X_pca_df, x = 'X1', y = 'X2', z ='X3', color = X_pca_df['Labels'])
plt.show()

sns.scatterplot(data = X_pca_df, x = 'X1', y = 'X2', hue = X_pca_df['Labels'] )
plt.show()

sns.scatterplot(data = X_pca_df, x = 'X2', y = 'X3', hue = X_pca_df['Labels'] )
plt.show()


sns.scatterplot(data = X_pca_df, x = 'X1', y = 'X3', hue = X_pca_df['Labels'] )
plt.show()
"""We should first notice that the clusters are quite separated and the algorithm seems to be doing the cluster separation well,
 as the frontiers between clusters seem quite clear.

Now we should create a Results DataFrame that includes the labels and apply filtering methods to infer information about the clusters provided by the K-Means model.
 Our objective is to define what type of customer is reflected in each cluster!"""

results_df = customer.drop(['ID', 'transf_income', 'trans_age'],axis = 1 )
results_df['Labels'] = kmeans.labels_
results_df = results_df.astype({'Sex': 'int32', 'Marital status':'int32','Education':'int32', 'Occupation':'int32', 'Settlement size':'int32'})
results_df.info()
customer.columns
results_df

# Summary Satistics of each cluster
summary ={}
for index in range(6):
    summary[index] = results_df[results_df['Labels'] == index].describe().T
summary[0] # Summary of cluster zero(0)
 
results_df[results_df['Labels'] == 0].hist(figsize = (15, 15)  )
plt.show()


results_df[results_df['Labels'] == 1].hist(figsize = (15, 15)  )
plt.show()
summary[1]


results_df[results_df['Labels'] == 2].hist(figsize = (15, 15)  )
plt.show()


results_df[results_df['Labels'] == 3].hist(figsize = (15, 15)  )
plt.show()


results_df[results_df['Labels'] == 4].hist(figsize = (15, 15)  )
plt.show()


results_df[results_df['Labels'] == 5].hist(figsize = (15, 15)  )
plt.show()


