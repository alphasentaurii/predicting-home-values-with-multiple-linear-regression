#!/usr/bin/env python
# coding: utf-8

# ---
# **Module 1 Final Project Submission**
# * Student name: **Ru Keïn**
# * Student pace: **Full-Time**
# * Project review date/time: **November 4, 2019 at 2:00 PM PST**
# * Instructor name: **James Irving, PhD**
# ---
# > Blog post URL: 
# http://www.hakkeray.com/projects/datascience/2019/11/06/predicting-home-values-with-multiple-linear-regression.html
# 
# > Link to video: 
# https://vimeo.com/rukein/datascience-project-1
# 
# > Link to tableau public: https://public.tableau.com/views/HousePricesbyZipCodeinKingCountyWA/KingCounty?:display_count=y&:origin=viz_share_link

# **GOAL**
# * Identify best combination of variable(s) for predicting property values in King County, Washington, USA. 
# 
# **OBJECTIVES**
# * Address null, missing, duplicate, and unreliable values in the data.
# * Determine best approach for analyzing each feature: continuous vs. categorical values
# * Identify which combination of features (X) are the best predictors of the target (y). 
# 
# **QUESTIONS TO EXPLORE**
# * *Scrub*
#     * 1. How should we address each feature to prepare it for EDA?
#  
# * *Explore*
#     * 2. Which predictors are closely related (and should be dropped)?
#     * 3. Is there an overlap in square-footage measurements?
#     * 4. Can we combine two features into one to achieve a higher correlation?
#     * 5. Does geography (location) have any relationship with the values of each categorical variable?
#  
# * *Model*
#     * 6. Which features are the best candidates for predicting property values?
#     * 7. Does removing outliers improve the distribution?
#     * 8. Does scaling/transforming variables improve the regression algorithm?

# **TABLE OF CONTENTS**
# 
# **[1  OBTAIN]**
# Import libraries, packages, data set
# * 1.1 Import libraries and packages
# * 1.2 Import custom functions
# * 1.3 Import dataset and review columns, variables
# 
# **[2  SCRUB]**
# Clean and organize the data.
# * 2.1 Find and replace missing values (nulls)
# * 2.2 Identify/Address characteristics of each variable (numeric vs categorical) 
# * 2.3 Check for and drop any duplicate observations (rows)
# * 2.4 Decide which variables to keep for EDA
# 
# **[3  EXPLORE]**
# Preliminary analysis and visualizations.
# * 3.1 Linearity: Scatterplots, scattermatrix
# * 3.2 Multicollinearity: Heatmaps, scatterplots
# * 3.3 Distribution: Histograms, Kernel Density Estimates (KDE), LMplots, Boxplots
# * 3.4 Regression: regression plots
# 
# **[4  MODEL]**
# Iterate through linreg models to find best fit predictors
# * 4.1 Model 1: OLS Linear Regression
# * 4.2 Model 2: One-Hot Encoding
# * 4.3 Model 3: Error terms
# * 4.4 Model 4: QQ Plots
# * 4.5 Model 5: Outliers
# * 4.6 Model 6: Robust Scaler
# 
# **[5  VALIDATION]**
# Validate the results.
# * 5.1 K-Fold Cross Validation
# 
# **[6  INTERPRET]**
# Summarize the findings and make recommendations.
# * 6.1 Briefly summarize the results of analysis
# * 6.2 Make recommendations
# * 6.3 Describe possible future directions
# 
# **[7  Additional Research]**
# Extracting median home values based on zipcodes

# ---
# # OBTAIN

# ## Import libraries + packaes

# In[1]:


# Import libraries and packages

# import PyPi package for cohort libraries using shortcut
#!pip install -U fsds_100719 # comment out after install so it won't run again
# Import packages
import fsds_100719 as fs
from fsds_100719.imports import *
plt.style.use('fivethirtyeight')
#inline_rc = dict(mpl.rcParams)
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import normaltest as normtest # D'Agostino and Pearson's omnibus test
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
#!pip install uszipcode


#ignore pink warnings
import warnings
warnings.filterwarnings('ignore')

# Allow for large # columns
pd.set_option('display.max_columns', 0)
# pd.set_option('display.max_rows','')


# ## Import custom functions

# In[2]:


# HOT_STATS() function: display statistical summaries of a feature column
def hot_stats(data, column, verbose=False, t=None):
    """
    Scans the values of a column within a dataframe and displays its datatype, 
    nulls (incl. pct of total), unique values, non-null value counts, and 
    statistical info (if the datatype is numeric).
    
    ---------------------------------------------
    
    Parameters:
    
    **args:
    
        data: accepts dataframe
    
        column: accepts name of column within dataframe (should be inside quotes '')
    
    **kwargs:
    
        verbose: (optional) accepts a boolean (default=False); verbose=True will display all 
        unique values found.   
    
        t: (optional) accepts column name as target to calculate correlation coefficient against 
        using pandas data.corr() function. 
    
    -------------
    
    Examples: 
    
    hot_stats(df, 'str_column') --> where df = data, 'string_column' = column you want to scan
    
    hot_stats(df, 'numeric_column', t='target') --> where 'target' = column to check correlation value
    
    -----------------
    Developer notes: additional features to add in the future:
    -get mode(s)
    -functionality for string objects
    -pass multiple columns at once and display all
    -----------------
    SAMPLE OUTPUT: 
    ****************************************
    
    -------->
    HOT!STATS
    <--------

    CONDITION
    Data Type: int64

    count    21597.000000
    mean         3.409825
    std          0.650546
    min          1.000000
    25%          3.000000
    50%          3.000000
    75%          4.000000
    max          5.000000
    Name: condition, dtype: float64 

    à-la-Mode: 
    0    3
    dtype: int64


    No Nulls Found!

    Non-Null Value Counts:
    3    14020
    4     5677
    5     1701
    2      170
    1       29
    Name: condition, dtype: int64

    # Unique Values: 5
    
    """
    # assigns variables to call later as shortcuts 
    feature = data[column]
    rdash = "-------->"
    ldash = "<--------"
    
    # figure out which hot_stats to display based on dtype 
    if feature.dtype == 'float':
        hot_stats = feature.describe().round(2)
    elif feature.dtype == 'int':
        hot_stats = feature.describe()
    elif feature.dtype == 'object' or 'category' or 'datetime64[ns]':
        hot_stats = feature.agg(['min','median','max'])
        t = None # ignores corr check for non-numeric dtypes by resetting t
    else:
        hot_stats = None

    # display statistics (returns different info depending on datatype)
    print(rdash)
    print("HOT!STATS")
    print(ldash)
    
    # display column name formatted with underline
    print(f"\n{feature.name.upper()}")
    
    # display the data type
    print(f"Data Type: {feature.dtype}\n")
    
    # display the mode
    print(hot_stats,"\n")
    print(f"à-la-Mode: \n{feature.mode()}\n")
    
    # find nulls and display total count and percentage
    if feature.isna().sum() > 0:  
        print(f"Found\n{feature.isna().sum()} Nulls out of {len(feature)}({round(feature.isna().sum()/len(feature)*100,2)}%)\n")
    else:
        print("\nNo Nulls Found!\n")
    
    # display value counts (non-nulls)
    print(f"Non-Null Value Counts:\n{feature.value_counts()}\n")
    
    # display count of unique values
    print(f"# Unique Values: {len(feature.unique())}\n")
    # displays all unique values found if verbose set to true
    if verbose == True:
        print(f"Unique Values:\n {feature.unique()}\n")
        
    # display correlation coefficient with target for numeric columns:
    if t != None:
        corr = feature.corr(data[t]).round(4)
        print(f"Correlation with {t.upper()}: {corr}")


# In[3]:


# NULL_HUNTER() function: display Null counts per column/feature
def null_hunter(df):
    print(f"Columns with Null Values")
    print("------------------------")
    for column in df:
        if df[column].isna().sum() > 0:
            print(f"{df[column].name}: \n{df[column].isna().sum()} out of {len(df[column])} ({round(df[column].isna().sum()/len(df[column])*100,2)}%)\n")


# In[4]:


# CORRCOEF_DICT() function: calculates correlation coefficients assoc. with features and stores in a dictionary
def corr_dict(X, y):
    corr_coefs = []
    for x in X:
        corr = df[x].corr(df[y])
        corr_coefs.append(corr)
    
    corr_dict = {}
    
    for x, c in zip(X, corr_coefs):
        corr_dict[x] = c
    return corr_dict


# In[5]:


# SUB_SCATTER() function: pass list of features (x_cols) and compare against target (or another feature)
def sub_scatter(data, x_cols, y, color=None, nrows=None, ncols=None):
    """
    Desc: displays set of scatterplots for multiple columns or features of a dataframe.
    pass in list of column names (x_cols) to plot against y-target (or another feature for 
    multicollinearity analysis)
    
    args: data, x_cols, y
    
    kwargs: color (default is magenta (#C839C5))
    
    example:
    
    x_cols = ['col1', 'col2', 'col3']
    y = 'col4'
    
    sub_scatter(df, x_cols, y)
    
    example with color kwarg:
    sub_scatter(df, x_cols, y, color=#)
    
    alternatively you can pass the column list and target directly:
    sub_scatter(df, ['col1', 'col2', 'col3'], 'price')

    """   
    if nrows == None:
        nrows = 1
    if ncols == None:
        ncols = 3
    if color == None:
        color = '#C839C5'
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,4))
    for x_col, ax in zip(x_cols, axes):
        data.plot(kind='scatter', x=x_col, y=y, ax=ax, color=color)
        ax.set_title(x_col.capitalize() + " vs. " + y.capitalize())


# In[6]:


# SUB_HISTS() function: plot histogram subplots
def sub_hists(data):
    plt.style.use('fivethirtyeight')
    for column in data.describe():
        fig = plt.figure(figsize=(12, 5))
        
        ax = fig.add_subplot(121)
        ax.hist(data[column], density=True, label = column+' histogram', bins=20)
        ax.set_title(column.capitalize())

        ax.legend()
        
        fig.tight_layout()


# In[7]:


# PLOT_REG() function: plot regression
def plot_reg(data, feature, target):
    sns.regplot(x=feature, y=target, data=data)
    plt.show()


# ## Import Data

# In[8]:


# import dataset and review data types, columns, variables
df = pd.read_csv('kc_house_data.csv') 
df.head()


# ---
# # SCRUB 
# 
# Clean and organize the data.

# **FIRST GLANCE - Items to note**
#     * There are 2 object datatypes that contain numeric values : 'date', 'sqft_basement'
#     * The total value count is 21597. Some columns appear to be missing a substantial amount of data 
#     (waterfront and yr_renovated).

# In[9]:


# Display information about the variables, columns and datatypes
df.info()


# Before going further, a little house-keeping is in order. Let's breakdown the columns into groups based on feature-type as they relate to a real estate market context:
# 
# *Dependent Variable:*
# 
# TARGET
# **price**
# 
# *Independent Variables:*
# 
# INTERIOR
# **bedrooms, bathrooms, floors**
# 
# SIZE (SQUARE FOOTAGE)
# **sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15**
# 
# LOCATION
# **zipcode, lat, long, waterfront**
# 
# QUALITY
# **condition, grade, yr_built, yr_renovated**
# 
# ANALYTICS
# **date, id, view**

# ## Missing Values
# Find and replace missing values using null_hunter() function.

# In[10]:


# hunt for nulls
null_hunter(df)            


# Before deciding how to handle nulls in the 3 columns above, let's take a closer look at each one and go from there.

# ## Data Casting
# 
# Identify/Address characteristics of each variable (numeric vs categorical)

# ### ['waterfront']

# In[11]:


hot_stats(df, 'waterfront')


# In[12]:


# Fill nulls with most common value (0.0) # float value
df['waterfront'].fillna(0.0, inplace=True)
#  verify changes
df['waterfront'].isna().sum()


# In[13]:


# Convert datatype to boolean (values can be either 0 (not waterfront) or 1(is waterfront)
df['is_wf'] = df['waterfront'].astype('bool')
# verify
df['is_wf'].value_counts()


# ### ['yr_renovated']

# In[14]:


hot_stats(df, 'yr_renovated')


# In[15]:


# This feature is also heavily skewed with zero values. 
# It should also be treated as a boolean since a property is either renovated or it's not).

# fill nulls with most common value (0)
df['yr_renovated'].fillna(0.0, inplace=True) # use float value to match current dtype

# verify change
df['yr_renovated'].isna().sum()


# In[16]:


# Use numpy arrays to create binarized column 'is_renovated'
is_renovated = np.array(df['yr_renovated'])
is_renovated[is_renovated >= 1] = 1
df['is_ren'] = is_renovated
df['is_ren'].value_counts()


# In[17]:


# Convert to boolean
df['is_ren'] = df['is_ren'].astype('bool')

# verify
df['is_ren'].value_counts()


# ### ['view']

# In[18]:


hot_stats(df, 'view')


# In[19]:


# Once again, almost all values are 0 .0

# replace nulls with most common value (0). 
df['view'].fillna(0, inplace=True)

#verify
df['view'].isna().sum()


# Since view has a finite set of values (0 to 4) we could assign category codes. However, considering the high number of zeros, it makes more sense to binarize the values into a new column representing whether or not the property was viewed.

# In[20]:


# create new boolean column for view:
df['viewed'] = df['view'].astype('bool')

# verify
df['viewed'].value_counts()


# ### ['sqft_basement']

# In[21]:


hot_stats(df, 'sqft_basement')


# In[22]:


# Note the majority of the values are zero...we could bin this as a binary 
# where the property either has a basement or does not...

# First replace '?'s with string value '0.0'
df['sqft_basement'].replace(to_replace='?', value='0.0', inplace=True)


# In[23]:


# and change datatype to float.
df['sqft_basement'] = df['sqft_basement'].astype('float')


# In[24]:


hot_stats(df, 'sqft_basement', t='price')


# In[25]:


df['basement'] = df['sqft_basement'].astype('bool')


# In[26]:


df['basement'].value_counts()


# In[27]:


corrs = ['is_wf', 'is_ren', 'viewed', 'basement']

# check correlation coefficients
corr_dict(corrs, 'price')


# None of these correlation values look strong enough to be predictive of price (min threshold > 0.5, ideally 0.7)

# ### ['floors']

# In[28]:


hot_stats(df, 'floors', t='price')


# Bathrooms appears to have a very linear relationship with price. Bedrooms is somewhat linear up to a certain point. Let's look at the hot stats for both.

# ### ['bedrooms']

# In[29]:


hot_stats(df, 'bedrooms', t='price')


# ### ['bathrooms']

# In[30]:


hot_stats(df, 'bathrooms', t='price')


# Bathrooms is the only feature showing correlation over the 0.5 threshold.

# The column-like distributions of these features in the scatterplots below indicate the values are categorical.

# In[31]:


# sub_scatter() creates scatter plots for multiple features side by side.
y = 'price'
x_cols = ['floors','bedrooms', 'bathrooms']

sub_scatter(df, x_cols, y)


# Looking at each one more closely using seaborn's catplot:

# In[32]:


for col in x_cols:
    sns.catplot(x=col, y='price', height=10, legend=True, data=df)


# In[33]:


# save correlation coefficients higher than 0.5 in a dict
corr_thresh_dict = {}
corrs = ['bathrooms']
corr_thresh_dict = corr_dict(corrs, 'price')
corr_thresh_dict


# ### ['condition']

# In[34]:


hot_stats(df, 'condition', t='price')


# In[35]:


sns.catplot(x='condition', y='price', data=df, height=8)


# Positive linear correlation between price and condition up to a point, but with diminishing returns.

# ### ['grade']

# In[36]:


# View grade stats
hot_stats(df, 'grade', t='price')


# In[37]:


x_cols = ['condition', 'grade']
for col in x_cols:
    sns.catplot(x=col, y='price', height=10, legend=True, data=df)


# Grade shows a relatively strong positive correlation with price.

# ### ['yr_built'] 

# In[38]:


hot_stats(df, 'yr_built', t='price')


# In[39]:


# Let's look at the data distribution of yr_built values 

fig, ax = plt.subplots()
df['yr_built'].hist(bins=10, color='#68FDFE', edgecolor='black', grid=True, alpha=0.6)
xticks = (1900, 1920, 1940, 1960, 1980, 2000, 2015)
yticks = (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
plt.xticks(xticks);
plt.yticks(yticks);
ax.set_title('Year Built Histogram', fontsize=16)
ax.set_xlabel('yr_built', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12);


# Most houses were built during the second half of the century (after 1950). We'll use adaptive binning based on quantiles for yr_built in order to create a more normal distribution.

# In[40]:


# define a binning scheme with custom ranges based on quantiles
quantile_list = [0, .25, .5, .75, 1.]

quantiles = df['yr_built'].quantile(quantile_list)

quantiles # 1900, 1951, 1975, 1997, 2015


# In[41]:


# Bin the years in to ranges based on the quantiles.
yb_bins = [1900, 1951, 1975, 1997, 2015]

# label the bins for each value 
yb_labels = [1, 2, 3, 4]

# store the yr_range and its corresponding yr_label as new columns in df

# create a new column for the category range values
df['yb_range'] = pd.cut(df['yr_built'], bins=yb_bins)

# create a new column for the category labels
df['yb_cat'] = pd.cut(df['yr_built'], bins=yb_bins, labels=yb_labels)


# In[42]:


# view the binned features corresponding to each yr_built 
df[['yr_built','yb_cat', 'yb_range']].iloc[9003:9007] # picking a random index location


# In[43]:


# Let’s look at the original distribution histogram again with the quantiles added:

fig, ax = plt.subplots()

df['yr_built'].hist(bins=10, color='#68FDFE', edgecolor='black', grid=True, alpha=0.6)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color='b')
    ax.legend([qvl], ['Quantiles'], fontsize=10)
    xticks = quantiles
    yticks = (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
    plt.xticks(xticks);
    plt.yticks(yticks);
    ax.set_title('Year Built Histogram with Quantiles',fontsize=16)
    ax.set_xlabel('Year Built', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)


# In[44]:


# values look much more normally distributed between the new categories
df.yb_cat.value_counts()


# In[45]:


# visualize the distribution of the binned values

fig, ax = plt.subplots()
df['yb_cat'].hist(bins=4, color='#68FDFE', edgecolor='black', grid=True, alpha=0.6)
ax.set_title('Year Built Categories Histogram', fontsize=12)
ax.set_xlabel('Year Built Binned Categories', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)


# In[46]:


sns.catplot(x='yb_cat', y='price', data=df, height=8)


# ###  ['zipcode']

# In[47]:


hot_stats(df, 'zipcode')


# In[48]:


# Let's look at the data distribution of the 70 unique zipcode values 
fig, ax = plt.subplots()
df['zipcode'].hist(bins=7, color='#67F86F',
edgecolor='black', grid=True)
ax.set_title('Zipcode Histogram', fontsize=16)
ax.set_xlabel('Zipcodes', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)


# In[49]:


# Let’s define a binning scheme with custom ranges for the zipcode values 
# The bins will be created based on quantiles

quantile_list = [0, .25, .5, .75, 1.]

quantiles = df['zipcode'].quantile(quantile_list)

quantiles # 98001, 98033, 98065, 98118, 98199


# In[50]:


# Now we can label the bins for each value and store both the bin range 
# and its corresponding label.

zip_bins = [98000, 98033, 98065, 98118, 98200]

zip_labels = [1, 2, 3, 4]

df['zip_range'] = pd.cut(df['zipcode'], bins=zip_bins)

df['zip_cat'] = pd.cut(df['zipcode'], bins=zip_bins, labels=zip_labels)

# view the binned features 
df[['zipcode','zip_cat', 'zip_range']].iloc[9000:9005] # pick a random index


# In[51]:


# visualize the quantiles in the original distribution histogram

fig, ax = plt.subplots()

df['zipcode'].hist(bins=7, color='#67F86F', edgecolor='black', grid=True)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color='black')
    ax.legend([qvl], ['Quantiles'], fontsize=10)
    ax.set_title('Zipcode Histogram with Quantiles',fontsize=12)
    ax.set_xlabel('Zipcodes', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)


# In[52]:


sns.catplot(x='zipcode', y='price', data=df, height=10)


# > Some zip codes may have higher priced homes than others, so it's hard to determine from the catplot how this could be used as a predictor. We'll have to explore this variable using geographic plots to see how the distributions trend on a map (i.e. proximity).

# ### ['lat']  ['long']

# > The coordinates for latitude and longitude are not going to be useful to us as far as regression models since we already have zipcodes as a geographic identifier. However we can put them to use for our geographic plotting.

# ### ['date'] 
# convert to datetime

# In[53]:


df['date'] = pd.to_datetime(df['date'])
df['date'].dtype


# In[54]:


hot_stats(df, 'date', t='price')


# ### ['sqft_above']

# In[55]:


hot_stats(df, 'sqft_above', t='price')


#     Some correlation with price here!

# ### ['sqft_living']

# In[56]:


hot_stats(df, 'sqft_living', t='price')


# sqft_living shows correlation value of 0.7 with price -- our highest coefficient yet!

# ### ['sqft_lot']

# In[57]:


hot_stats(df, 'sqft_lot', t='price')


# ### ['sqft_living15']

# In[58]:


hot_stats(df, 'sqft_living15', t='price')


# We've identified another coefficient over the 0.5 correlation threshold.

# In[59]:


hot_stats(df, 'sqft_lot15', t='price')


# ## Duplicates

# The primary key we'd use as an index for this data set would be 'id'. Our assumption therefore is that the 'id' for each observation (row) is unique. Let's do a quick scan for duplicate entries to confirm this is true.

# ### ['id']

# In[60]:


hot_stats(df, 'id')


# In[61]:


# check for duplicate id's
df['id'].duplicated().value_counts() 


# In[62]:


# Looks like there are in fact some duplicate ID's! Not many, but worth investigating.

# Let's flag the duplicate id's by creating a new column 'is_dupe':
df.loc[df.duplicated(subset='id', keep=False), 'is_dupe'] = 1 # mark all duplicates 

# verify all duplicates were flagged
df.is_dupe.value_counts() # 353


# In[63]:


# the non-duplicate rows show as null in our new column
df.is_dupe.isna().sum()


# In[64]:


# Replace 'nan' rows in is_dupe with 0.0
df.loc[df['is_dupe'].isna(), 'is_dupe'] = 0

# verify
df['is_dupe'].unique()


# In[65]:


# convert column to boolean data type
df['is_dupe'] = df['is_dupe'].astype('bool')
# verify
df['is_dupe'].value_counts()


# In[66]:


# Let's now copy the duplicates into a dataframe subset for closer inspection
# It's possible the pairs contain data missing from the other which 
# we can use to fill nulls identified previously.

df_dupes = df.loc[df['is_dupe'] == True]

# check out the data discrepancies between duplicates (first 3 pairs)
df_dupes.head(6)


# In[67]:


# Looks like the only discrepancies might occur between 'date' and 'price' values
# Some of the prices nearly double, even when the re-sale is just a few months later!

df_dupes.loc[df_dupes['id'] == 6021501535]


# In[68]:


# Set index of df_dupes to 'id'
df_dupes.set_index('id')
# Set index of df to 'id'
df.set_index('id')


# In[69]:


# Before we drop the duplicates, let's save a backup copy of the current df using pickle.
import pickle
# create pickle data_object
df_predrops = df


# In[70]:


with open('data.pickle', 'wb') as f:
    pickle.dump(df_predrops, f, pickle.HIGHEST_PROTOCOL)


# In[71]:


#import df (pre-drops) with pickle
#with open('data.pickle', 'rb') as f:
#    df = pickle.load(f)


# In[72]:


# let's drop the first occurring duplicate rows and keep the last ones 
# (since those more accurately reflect latest market data)

# save original df.shape for comparison after dropping duplicate rows
predrop = df.shape # (21597, 28)

# first occurrence, keep last
df.drop_duplicates(subset='id', keep ='last', inplace = True) 

# verify dropped rows by comparing df.shape before and after values
print(f"predrop: {predrop}")
print(f"postdrop: {df.shape}")


# ## Target

# #### ['price']

# In[73]:


# Let's take a quick look at the statistical data for our dependent variable (price):
hot_stats(df, 'price')


# > Keeping the below numbers in mind could be helpful as we start exploring the data:
# 
# * range: 78,000 to 7,700,000
# * mean value: 540,296
# * median value: 450,000

# In[74]:


# long tails in price and the median is lower than the mean - distribution is skewed to the right
sns.distplot(df.price)


# At this point we can begin exploring the data. Let's first review our current feature list and get rid of any columns we no longer need. As we go through our analysis we'll decide which additional columns to drop, transform, scale, normalize, etc.

# In[75]:


df.info()


# In[76]:


# cols to drop bc irrelevant to linreg model or using new versions instead:
hot_drop = ['date','id','waterfront', 'yr_renovated', 'view', 'yr_built', 'yb_range', 'zip_range']


# In[77]:


# store hot_drop columns in separate df
df_drops = df[hot_drop].copy()


# In[78]:


# set index of df_drops to 'id'
df_drops.set_index('id')
# verify
df_drops.info()


# In[79]:


# drop it like its hot >> df.drop(hot_drop, axis=1, inplace=True)
df.drop(hot_drop, axis=1, inplace=True)

# verify dropped columns
df.info()


# # EXPLORE:
#     
#     EDA CHECKLIST:
#     linearity (scatter matrices)
#     multicollinearity (heatmaps)
#     distributions (histograms, KDEs)
#     regression (regplot)

# **QUESTION: Which features are the best candidates for predicting property values?**

# In[80]:


df.describe()


# ## Linearity
# During the scrub process we made some assumptions and guesses based on correlation coefficients and other values. Let's see what the visualizations tell us by creating some scatter plots.

# In[81]:


# Visualize the relationship between square-footages and price
sqft_int = ['sqft_living', 'sqft_above', 'sqft_basement']
sub_scatter(df, sqft_int, 'price', color='#0A2FC4') #20C5C6


# In[82]:


# visualize relationship between sqft_lot, sqft_lot15, sqft_living15 and price.
y = 'price'
sqft_ext = ['sqft_living15', 'sqft_lot', 'sqft_lot15']

sub_scatter(df, sqft_ext, y, color='#6A76FB')


# Linear relationships with price show up clearly for sqft_living, sqft_above, sqft_living15.

# ## Multicollinearity
# **QUESTION: which predictors are closely related (and should be dropped)?**

#     + multicollinearity: remove variable having most corr with largest # of variables

# In[83]:


#correlation values to check

corr = df.corr()

# Checking multicollinearity with a heatmap
def multiplot(corr,figsize=(20,20)):
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.zeros_like(corr, dtype=np.bool)
    idx = np.triu_indices_from(mask)
    mask[idx] = True

    sns.heatmap(np.abs(corr),square=True,mask=mask,annot=True,cmap="Greens",ax=ax)
    
    ax.set_ylim(len(corr), -.5, .5)
    
    
    return fig, ax

multiplot(np.abs(corr.round(3)))


# The square footages probably overlap. (In other words sqft_above and sqft_basement could be part of the total sqft_living measurement).

# In[84]:


# Visualize multicollinearity between interior square-footages
x_cols = ['sqft_above', 'sqft_basement']
sub_scatter(df, x_cols, 'sqft_living', ncols = 2, color='#FD6F6B')  # lightred


#     Yikes. These are extremely linear. Just for fun, let's crunch the numbers...
# 
# **QUESTION: Is there any overlap in square-footage measurements?**

# In[116]:


# create new col containing sum of above and basement
df['sqft_sums'] = df['sqft_above'] + df['sqft_basement']
df['sqft_diffs'] = df['sqft_living'] - df['sqft_above']


sqft_cols = ['sqft_sums', 'sqft_living', 'sqft_above','sqft_diffs', 'sqft_basement']
df_sqft = df[sqft_cols]
df_sqft


# > Looks like the 0.0 values in sqft_basement do in fact mean there is no basement, and for those houses the sqft_above is exactly the same as sqft_living. With this now confirmed, we can be confident that sqft_living is the only measurement worth keeping for analysis.

# In[117]:


# check random location in the index
print(df['sqft_living'].iloc[0]) #1180
print(df['sqft_above'].iloc[0] + df['sqft_basement'].iloc[0]) #1180

print(df['sqft_living'].iloc[1]) #2570
print(df['sqft_above'].iloc[1] + df['sqft_basement'].iloc[1]) #2570


# In[118]:


# sqft_living == sqft_basement + sqft_above ?
# sqft_lot - sqft_living == sqft_above ?

sqft_lv = np.array(df['sqft_living'])
sqft_ab = np.array(df['sqft_above'])
sqft_bs = np.array(df['sqft_basement'])

sqft_ab + sqft_bs == sqft_lv #array([ True,  True,  True, ...,  True,  True,  True])


# In[119]:


# check them all at once
if sqft_ab.all() + sqft_bs.all() == sqft_lv.all():
    print("True")


# **ANSWER: Yes. Sqft_living is the sum of sqft_above and sqft_basement.**

# ## Distributions

# In[93]:


# group cols kept for EDA into lists for easy extraction

# binned:
bins = ['is_wf', 'is_ren', 'viewed','basement', 'yb_cat', 'zip_cat', 'is_dupe']

# categorical:
cats = ['grade', 'condition', 'bathrooms', 'bedrooms', 'floors']

#numeric:
nums = ['sqft_living', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot', 'sqft_lot15']

# geographic:
geo = ['lat', 'long', 'zipcode']

# target variable
t = ['price']


# ### Histograms

# In[94]:


# hist plot
sub_hists(df)


# Although sqft_living15 didn't have as much linearity with price as other candidates, it appears to have the most normal distribution out of all of them.

# ### KDEs

# In[95]:


# Kernel Density Estimates (distplots) for square-footage variables
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(12,12))
sns.distplot(df['sqft_living'], ax=ax[0][0])
sns.distplot(df['sqft_living15'], ax=ax[0][1])
sns.distplot(df['sqft_lot'], ax=ax[1][0])
sns.distplot(df['sqft_lot15'], ax=ax[1][1])
sns.distplot(df['sqft_above'], ax=ax[2][0])
sns.distplot(df['sqft_basement'], ax=ax[2][1])


# In[96]:


fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(12,12))
sns.distplot(df['bathrooms'], ax=ax[0])
sns.distplot(df['bedrooms'], ax=ax[1])
sns.distplot(df['floors'], ax=ax[2])


# In[97]:


fig, ax = plt.subplots(ncols=3, figsize=(12,3))
sns.distplot(df['condition'], ax=ax[0])
sns.distplot(df['grade'], ax=ax[1])
sns.distplot(df['zipcode'], ax=ax[2]) # look at actual zipcode value dist instead of category


# > Diminishing returns for condition (highest scores = min 3 out of 5) and grade (score of 7 out of 13)
# Zip Codes are all over the map...literally.

# ### Geographic

# **QUESTION: Does geography (location) have any relationship with the values of each categorical variable?**

# In[98]:


cats


# In[99]:


# lmplot geographic distribution by iterating over list of cat feats
for col in cats:
    sns.lmplot(data=df, x="long", y="lat", fit_reg=False, hue=col, height=10)
plt.show()


# > The highest graded properties appear to be most dense in the upper left (NW) quadrant. Since we already know that grade has a strong correlation with price, we can posit more confidently that grade, location, and price are strongly related.
# 
# > Homes with highest number of floors tend to be located in the NW as well. If we were to look at an actual map, we'd see this is Seattle. 

# In[100]:


bins


# In[101]:


# run binned features through lmplot as a forloop to plot geographic distribution visual
for col in bins:
    sns.lmplot(data=df, x="long", y="lat", fit_reg=False, hue=col, height=10)
plt.show()


# > Some obvious but also some interesting things to observe in the above lmplots:
# 
# * waterfront properties do indeed show up as being on the water. Unfortunately as we saw earlier, this doesn't seem to correlate much with price. This is odd (at least to me) because I'd expect those homes to be more expensive. If this were Los Angeles (where I live) that's a well-known fact...
#     
# * 'is_dupe' (which represents properties that sold twice in the 2 year period of this dataset) tells us pretty much nothing about anything. They look evenly distributed geographically - we can eliminate this from the model. 
#     
# * Probably the most surprising observation is 'viewed'. They almost all line up with the coastline, or very close to the water. This may not mean anything but it is worth noting.
#     
# * Lastly, is_renovated is pretty densely clumped up in the northwest quadrant (again, Seattle). We can assume therefore that a lot of renovations are taking place in the city. Not entirely useful but worth mentioning neverthless.

# ### Box Plots

# In[102]:


x = df['grade']
y = df['price']

plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=1,figsize=(12,6))
sns.boxplot(x=x, y=y, ax=ax, showfliers=False) # outliers removed
# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='x-large',   
                  rotation=45)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='Grade Boxplot - No Outliers'
ax.set_title(title.title())
ax.set_xlabel('grade')
ax.set_ylabel('price')
fig.tight_layout()


# In[103]:


x = df['bathrooms']
y = df['price']

plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots(ncols=1,figsize=(12,6))
sns.boxplot(x=x, y=y, ax=ax, showfliers=False) # outliers removed
# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='large',   
                  rotation=90)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='Bathrooms Boxplot'
ax.set_title(title.title())
ax.set_xlabel('bathrooms')
ax.set_ylabel('price')
fig.tight_layout()


# In[104]:


x = df['bedrooms']
y = df['price']

plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots(ncols=1,figsize=(12,6))
sns.boxplot(x=x, y=y, ax=ax, showfliers=False) #outliers removed
# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='large',   
                  rotation=90)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='Bedrooms Boxplot'
ax.set_title(title.title())
ax.set_xlabel('bedrooms')
ax.set_ylabel('price')
fig.tight_layout()


# In[105]:


x = df['floors']
y = df['price']

plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots(ncols=1,figsize=(12,6))
sns.boxplot(x=x, y=y, ax=ax, showfliers=False) #outliers removed
# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='large',   
                  rotation=90)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='Floors Boxplot'
ax.set_title(title.title())
ax.set_xlabel('floors')
ax.set_ylabel('price')
fig.tight_layout()


# In[106]:


x = df['zipcode']
y = df['price']

plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots(ncols=1,figsize=(20,20))
sns.boxplot(x=x, y=y, ax=ax, showfliers=False) #outliers removed

# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='x-large',   
                  rotation=45)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='zipcode boxplot'
ax.set_title(title.title())
ax.set_xlabel('zipcode')
ax.set_ylabel('price')
fig.tight_layout()


# Certain zipcodes certainly contain higher home prices (mean and median) than others. This is definitely worth exploring further.

# In[107]:


x = df['yb_cat']
y = df['price']

plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots(ncols=1,figsize=(12,6))
sns.boxplot(x=x, y=y, ax=ax, showfliers=False)
# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='x-large',   
                  rotation=45)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='Year Built Categories Boxplot'
ax.set_title(title.title())
ax.set_xlabel('yb_cat')
ax.set_ylabel('price')
fig.tight_layout()


# To a certain degree, there is some linearity in year built, with newer homes (category4) falling within higher price ranges than older homes.

# # MODEL:

# Let's run the first model using features that have the highest correlation with price. (Min threshold > 0.5)

# In[108]:


# BINNED VARS
corr_dict(bins, 'price') # none are over 0.5


# In[109]:


# GEOGRAPHIC VARS
corr_dict(geo, 'price') # none are over 0.5


# In[110]:


# CATEGORICAL VARS
corr_dict(cats, 'price') # grade and bathrooms are over 0.5


# In[111]:


# NUMERIC VARS
corr_dict(nums, 'price') #sqft_living, sqft_above, sqft_living15 are over 0.5


# NOTES: 
# > The coefficients above are based on the raw values. It's possible that some of the variables will produce a higher correlation with price after scaling / transformation. We'll test this out in the second model iteration.
# 
# > We also need to take covariance/multicollinearity into consideration. As we saw when we created the multiplot heatmap (as well as scatterplots), the sqft variables have covariance. To make things more difficult, they're also collinear with grade and bathrooms. This could cause our model to overfit.

# ## Model 1

# In[113]:


# highest corr coefs with price - initial model using raw values
pred1 = ['grade', 'sqft_living', 'bathrooms']


# In[114]:


corr_dict(pred1, 'price')


# In[115]:


f1 = '+'.join(pred1)
f1


# In[116]:


f ='price~'+f1
model = smf.ols(formula=f, data=df).fit()
model.summary()


# > P-values look good, but the R-squared (0.536) could be much higher. 
# 
# > Skew and Kurtosis are not bad (Skew:	3.293, Kurtosis:	35.828)
# 
# > Let's do a second iteration after one-hot-encoding condition and run OLS again with zipcode included and the condition dummies.

# In[117]:


# save key regression values in a dict for making quick comparisons between models
reg_mods = dict()
reg_mods['model1'] = {'vars':f1,'r2':0.536, 's': 3.293, 'k': 35.828}

reg_mods


# ## Model 2

# ### One-Hot Encoding
# 
# * Create Dummies for Condition

# In[118]:


# apply one-hot encoding to condition
df2 = pd.get_dummies(df, columns=['condition'], drop_first=True)
df2.head()


# In[119]:


df2.columns


# In[120]:


# iterate through list of dummy cols to plot distributions where val > 0

# create list of dummy cols
cols = ['condition_2', 'condition_3', 'condition_4']

# create empty dict
groups={}

# iterate over dummy cols and grouby into dict for vals > 0 
for col in cols:
    groups[col]= df2.groupby(col)[col,'price'].get_group(1.0)

# check vals
groups.keys()
groups['condition_2']


# In[121]:


# show diffs between subcats of condition using histograms
for k, v in groups.items():
    plt.figure()
    plt.hist(v['price'], label=k)
    plt.legend()


# In[122]:


# As we saw before, there are diminishing returns as far as condition goes...
# visualize another way with distplots
for k, v in groups.items():
    plt.figure()
    sns.distplot(v['price'], label=k)
    plt.legend()


# > NOTE condition is skewed by tails/outliers)

# In[123]:


# use list comp to grab condition dummies
c_bins = [col for col in df2.columns if 'condition' in col]
c_bins


# In[124]:


pred2 = ['C(zipcode)', 'grade', 'sqft_living', 'sqft_living15']


# In[125]:


pred2.extend(c_bins)


# In[126]:


pred2


# In[127]:


f2 = '+'.join(pred2)
f2


# In[128]:


f ='price~'+f2
f


# In[129]:


model = smf.ols(formula=f, data=df2).fit()
model.summary()


# In[130]:


# store model2 values to dict
reg_mods['model2'] = {'vars':f2, 'r2':0.750, 's': 5.170, 'k': 75.467}

reg_mods


# > Much higher R-squared but there are some fatal issues with this model:
# 1. There are a handful of zipcodes with very high P-values. 
# 2. Condition dummies have very high p-values.
# 3. Skew: 5.170 (increased)
# 4. Kurtosis: 75.467	(almost doubled from model1)
# 
# > Let's drop condition and try running it again.

# ## Model 3

# In[131]:


# create list of selected predictors
pred3 = ['C(zipcode)','grade', 'sqft_living']

# convert to string with + added
f3 = '+'.join(pred3)

# append target
f ='price~'+f3 

# Run model and show sumamry
model = smf.ols(formula=f, data=df).fit()
model.summary()


# In[132]:


reg_mods['model3'] = {'vars':f3, 'r2':0.744, 's': 4.927, 'k':69.417 }
reg_mods


# > R-squared value slightly decreased 
# 
# > P-values look good
# 
# > Skew and Kurtosis decreased slightly (compared to Model2 at least)

# ### Error Terms ('grade')

# In[151]:


# Visualize Error Terms for Grade
f = 'price~grade'
model = ols(formula=f, data=df).fit()
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, 'grade', fig=fig)
plt.show()


# ## Model 4

# In[142]:


# checking same model with GRADE as cat.
pred4 = ['C(zipcode)', 'C(grade)', 'sqft_living']

# convert to string with + added
f4 = '+'.join(pred4)

# append target
f ='price~'+f4 

# Run model and show sumamry
model = smf.ols(formula=f, data=df).fit()
model.summary()


# In[143]:


reg_mods['model4'] = {'vars': f4, 'r2':0.783, 's':4.032,'k':55.222 }
reg_mods


# > R-squared value increased from 0.75 to 0.78 - better.  
# 
# > P-values for Grade as a categorical are horrible except for scores of 11, 12, and 13. This could mean we recommend Grade as a factor still, with the requirement that the home score above 10 in order to have an impact on price. 
# 
# > Kurtosis and Skew both decreased to levels lower than models 2 and 3. However, the model would most likely benefit further from scaling/normalization. 

# ### QQ Plots
# Investigate high p-values

# In[160]:


import scipy.stats as stats
residuals = model.resid
fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
fig.show()


# > This is not what we want to see...Let's take a closer look at the outliers and find out if removing them helps at all. If not, we may need to drop Grade from the model.

# ## Model 5

# ### Outliers
# 
# **QUESTION: Does removing outliers improve the distribution?**

# In[163]:


# Visualize outliers with boxplot for grade
x = 'grade'
y = 'price'

plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=1,figsize=(12,6))

# iterate over categorical vars to build boxplots of price distributions

sns.boxplot(data=df, x=x, y=y, ax=ax)
# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='x-large',   
                  rotation=45)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='Grade Boxplot with Outliers'
ax.set_title(title.title())
ax.set_xlabel('grade')
ax.set_ylabel('price')
fig.tight_layout()


# In[136]:


# visualize outliers with boxplot for zipcode
x = 'zipcode'
y = 'price'

plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=1,figsize=(20,20))

# iterate over categorical vars to build boxplots of price distributions

sns.boxplot(data=df, x=x, y=y, ax=ax)
# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='small',   
                  rotation=45)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='Zipcode Boxplot with Outliers'
ax.set_title(title.title())
ax.set_xlabel('sqft_living')
ax.set_ylabel('price')
fig.tight_layout()


# > NOTE: If the assumption about zipcode (i.e. location) being a critical factor for home price is correct, we could identify from this a list of zipcodes with the highest prices of homes based on median home values -- the assumption for this being that people will pay more for a house located in a certain area than they would for a house in other parts of the county (even if that house is much bigger, has a higher grade, etc). 

# In[221]:


# Detect actual number of outliers for our predictors

out_vars = ['sqft_living', 'zipcode', 'grade', 'price']

df_outs = df[out_vars]
df_outs


# In[222]:


# Get IQR scores
Q1 = df_outs.quantile(0.25)
Q3 = df_outs.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[223]:



# True indicates outliers present
outliers = (df_outs < (Q1 - 1.5 * IQR)) |(df_outs > (Q3 + 1.5 * IQR))

for col in outliers:
    print(outliers[col].value_counts(normalize=True))


# > 8% of the values in grade and 5% in price are outliers.

# In[224]:


# Remove outliers 
df_zero_outs = df_outs[~((df_outs < (Q1 - 1.5 * IQR)) |(df_outs > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_outs.shape, df_zero_outs.shape)


# In[225]:


# number of outliers removed
df_outs.shape[0] - df_zero_outs.shape[0] # 2388


# In[226]:


# rerun OLS with outliers removed
pred5 = ['C(zipcode)', 'C(grade)', 'sqft_living']

# convert to string with + added
f5 = '+'.join(pred5)

# append target
f ='price~'+f5 

# Run model and show sumamry
model = smf.ols(formula=f, data=df_zero_outs).fit()
model.summary()


# In[227]:


reg_mods['model5'] = {'vars': f5, 'r2':0.780, 's':0.766, 'k':6.241}
reg_mods


# > Removing outliers drastically improved the skew and kurtosis values while maintaining R-squared at 0.78. However, this was at the cost of losing the majority of the grade score levels, leaving us with only 7,8,9 in the model. 
# 
# > We could use this to recommend aiming for a minimum grade score between 7 and 9. 

# In[228]:


# check QQ Plot for Model5
residuals = model.resid
fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
fig.show()


# > Not perfect, but definitely a significant improvement over Model 4.

# ## Model 6 (FINAL)

# ### Robust Scaler
# 
# Considering sqft_living values are in the 1000's while grade is 1 to 13, the model could most likely be improved further by scaling the square-footages down to a magnitude that aligns more closely with the other variables.

# In[231]:


# ADDING OUTLIER REMOVAL FROM preprocessing.RobuseScaler
# good to use when you have outliers bc uses median 
from sklearn.preprocessing import RobustScaler

robscaler = RobustScaler()
robscaler


# In[232]:


scale_vars = ['sqft_living']


# In[233]:


for col in scale_vars:
    col_data = df[col].values
    res = robscaler.fit_transform(col_data.reshape(-1,1)) # don't scale target
    df['sca_'+col] = res.flatten()


# In[234]:


df.describe().round(3)


# In[235]:


# plot histogram to check normality
df['sca_sqft_living'].hist(figsize=(6,6))


# In[237]:


df_zero_outs['sca_sqft_living'] = df['sca_sqft_living'].copy()
df_zero_outs


# In[256]:


# rerun OLS with outliers removed
pred6 = ['C(zipcode)', 'C(grade)', 'sca_sqft_living']

# convert to string with + added
f6 = '+'.join(pred6)

# append target
f ='price~'+f6 

# Run model and show sumamry
model = smf.ols(formula=f, data=df_zero_outs).fit()
model.summary()


# In[240]:


reg_mods['model6'] = {'vars': f6, 'r2': 0.780, 's': 0.766, 'k': 6.241}
reg_mods


# In[251]:


# save final output
df_fin = df_zero_outs.copy()

with open('data.pickle', 'wb') as f:
    pickle.dump(df_fin, f, pickle.HIGHEST_PROTOCOL)


# In[254]:



df_fin.to_csv('kc_housing_model_df_final_data.csv')


# # VALIDATION

# ## K-Fold Validation with OLS

# In[270]:


# k_fold_val_ols(X,y,k=10):
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 


y = df_fin['price']

X = df_fin.drop('price', axis=1)


# Run 10-fold cross validation
results = [['set#','R_square_train','MSE_train','R_square_test','MSE_test']]


num_coeff = X.shape[1]


list_predictors = [str(x) for x in X.columns]
list_predictors.append('intercept') 


reg_params = [list_predictors]


i=0
k=10
while i <(k+1):
    X_train, X_test, y_train, y_test = train_test_split(X,y) #,stratify=[cat_col_names])


    data = pd.concat([X_train,y_train],axis=1)
    f = 'price~C(zipcode)+C(grade)+sca_sqft_living' 
    model = smf.ols(formula=f, data=data).fit()
    model.summary()
    
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)


    train_residuals = y_hat_train - y_train
    test_residuals = y_hat_test - y_test


        
    train_mse = metrics.mean_squared_error(y_train, y_hat_train)
    test_mse = metrics.mean_squared_error(y_test, y_hat_test)


    R_sqare_train = metrics.r2_score(y_train,y_hat_train)
    R_square_test = metrics.r2_score(y_test,y_hat_test)


    results.append([i,R_sqare_train,train_mse,R_square_test,test_mse])
    i+=1


    
results = pd.DataFrame(results[1:],columns=results[0])
results
model.summary()


# # INTERPRET

# > SUMMARY: We can be confident that 78% of the final model (#6) explains the variation in data. Unfortunately, multicollinearity is a significant issue for linear regression and cannot be completely avoided. 
# 
# 
# > RECOMMENDATIONS: According to our final model, the best predictors of house prices are sqft-living, zipcode, and grade. 
# 
# > * Homes with larger living areas are valued higher than smaller homes. 
# > * Houses in certain zip codes are valued at higher prices than other zip codes.
# > * Homes that score above at least 8 on Grade will sell higher than those below.
# 
# > FUTURE WORK:
# * Identify ranking for zip codes by highest home prices (median home value)
#  

# # Additional Research

# > **Do house prices change over time or depending on season?**
# This data set was limited to a one-year time-frame. I'd be interested in widening the sample size to investigate how property values fluctuate over time as well as how they are affected by market fluctuations.

# > **Can we validate the accuracy of our prediction model by looking specifically at houses that resold for a higher price in a given timeframe?** In other words, try to identify which specific variables changed (e.g. increased grade score after doing renovations) and therefore were determining factors in the increased price of the home when it was resold.

# In[260]:


# pypi package for retrieving information based on us zipcodes
from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode=True) # set simple_zipcode=False to use rich info database

# create array of zipcodes
zipcodes = df['zipcode'].unique()
zipcodes


# In[261]:


# create empty dictionary 
dzip = {}

# search pypi uszipcode library to retreive data for each zipcode
for c in zipcodes:
    z = search.by_zipcode(c)
    dzip[c] = z.to_dict()
    
dzip.keys()


# In[265]:


# check information for one of the zipcodes 
# 98032 had the worst p-value (0.838)
dzip[98032]


# In[267]:


# try retrieving just the median home value for a given zipcode 
dzip[98199]['median_home_value'] #98199 mhv is 3x higher than 98032


# In[268]:


# create empty lists for keys and vals
med_home_vals = []
zips = []

# pull just the median home values from dataset and append to list
for index in dzip:
    med_home_vals.append(dzip[index]['median_home_value'])

# put zipcodes in other list
for index in dzip:
    zips.append(dzip[index]['zipcode'])

# zip both lists into dictionary
dzip_mhv = dict(zip(zips, med_home_vals))


# In[302]:


# we now have a dictionary that matches median home value to zipcode.
dzip_mhv

mills = []
halfmills = []

for k,v in dzip_mhv.items():
    if v > 1000000:
        mills.append([k])
    if v > 500000:
        halfmills.append([k])

print(mills)
print(halfmills)

