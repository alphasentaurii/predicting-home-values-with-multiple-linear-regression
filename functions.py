#         _ __ _   _
#  /\_/\ | '__| | | |
#  [===] | |  | |_| |
#   \./  |_|   \__,_| 
#
# statistical functions by Ru Keïn / @hakkeray

# * hot_stats()
# * zip_stats()

# * note: USZIPCODE pypi library is required to run zip_stats()
# Using pip in the notebook:
# !pip install -U uszipcode

# fsds tool required
# !pip install -U fsds_100719


import fsds_100719 as fs
from fsds_100719.imports import *
plt.style.use('seaborn-bright')
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as ols
import scipy.stats as stats
from scipy.stats import normaltest as normtest # D'Agostino and Pearson's omnibus test
from collections import Counter
from sklearn.preprocessing import RobustScaler
import uszipcode
from uszipcode import SearchEngine

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


# NULL_HUNTER() function: display Null counts per column/feature
def null_hunter(data):
    print(f"Columns with Null Values")
    print("------------------------")
    for column in data:
        if data[column].isna().sum() > 0:
            print(f"{data[column].name}: \n{data[column].isna().sum()} out of {len(data[column])} ({round(data[column].isna().sum()/len(data[column])*100,2)}%)\n")


# CORRCOEF_DICT() function: calculates correlation coefficients assoc. with features and stores in a dictionary
def corr_dict(data, X, y):
    corr_coefs = []
    for x in X:
        corr = data[x].corr(data[y])
        corr_coefs.append(corr)
    
    corr_dict = {}
    
    for x, c in zip(X, corr_coefs):
        corr_dict[x] = c
    return corr_dict

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


# SUB_HISTS() function: plot histogram subplots
def sub_hists(data):
    plt.style.use('seaborn-bright')
    for column in data.describe():
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121)
        ax.hist(data[column], density=True, label = column+' histogram', bins=20)
        ax.set_title(column.capitalize())
        ax.legend()
        fig.tight_layout()

# --------- ZIP_STATS() --------- #

def zip_stats(zipcodes, 
              minimum=0, maximum=5000000,
              simple=True):
    """
    Lookup median home values for zipcodes or return zip codes of a min and max median home value
    #TODO: add input options for city state county
    #TODO: add input options for other keywords besides median home val
    
    *Prerequisites: USZIPCODE() pypi package is a required dependency
    
    **ARGS
    zipcodes: dataframe or array of strings (zipcodes) 
    > Example1: zipcodes=df[zipcode']
    > Example2: zipcodes=['01267','90025']
    
    minimum: integer for dollar amount min threshold (default is 0)
    maximum: integer for dollar amount max threshold (default is 5000000, i.e. no maximum)
    
    **KWARGS
    simple: default=True
    > set simple_zipcode=False to use rich info database (will only apply once TODOs above are added)
    """
    # pypi package for retrieving information based on us zipcodes
    import uszipcode
    from uszipcode import SearchEngine
    
    # set simple_zipcode=False to use rich info database
    if simple:
        search = SearchEngine(simple_zipcode=True)
    else:
        search = SearchEngine(simple_zipcode=False)
        
    # create empty dictionary 
    dzip = {}

    # search pypi uszipcode library to retrieve data for each zipcode
    for code in zipcodes:
        z = search.by_zipcode(code)
        dzip[code] = z.to_dict()
    
    keyword='median_home_value'
    # # pull just the median home values from dataset and append to list
    # create empty lists for keys and vals
    keys = []
    zips = []
    
    for index in dzip:
        keys.append(dzip[index][keyword])

    # put zipcodes in other list
    for index in dzip:
        zips.append(dzip[index]['zipcode'])
        
    # zip both lists into dictionary
    zipkey = dict(zip(zips, keys))

    zipvals = {}
    
    for k,v in zipkey.items():
        if v > minimum and v < maximum:
            zipvals[k]=v
    return zipvals
