
---
**Module 1 Final Project Submission**

* Student name: **Ru Keïn**
* Student pace: **Full-Time**
* Project review date/time: **November 4, 2019 at 2:00 PM PST**
* Instructor name: **James Irving, PhD**
* Blog post URL: www.hakkeray.com/projects/datascience/king-county-housing-data

**GOAL**
* Identify best variable(s) for predicting property values in King County, Washington, USA. 

**OBJECTIVES**
* Address null, missing, duplicate, and unreliable values in the data.
* Determine best approach for analyzing each feature: continuous vs. categorical values
* Identify which combination of features (X) are the best predictors of the target (y). 

**QUESTIONS TO EXPLORE**
* Which predictors are closely related (and should be dropped)?
* Is there an overlap in square-footage measurements?
* Can we combine two features into one to achieve a higher correlation?
* Does geography (location) have any relationship with the values of each categorical variable?
* Which features are the best candidates for predicting property values?
* Does removing outliers improve the distribution?
* Does scaling/transforming variables improve the regression algorithm?

**FUTURE ANALYSES**
* Analyze Duplicates - is there a trend within houses re-sold for higher prices?

---
# OBTAIN
* Import requisite libraries and data
* Inspect columns, dataypes
    * df.head()


```python
# import PyPi package for cohort libraries using shortcut
#!pip install -U fsds_100719 # comment out after install so it won't run again
import fsds_100719 as fs
from fsds_100719.imports import *
plt.style.use('seaborn')
#inline_rc = dict(mpl.rcParams)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import normaltest as normtest # D'Agostino and Pearson's omnibus test
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
```

    fsds_1007219  v0.4.8 loaded.  Read the docs: https://fsds.readthedocs.io/en/latest/ 
    > For convenient loading of standard modules use: `>> from fsds_100719.imports import *`
    



<style  type="text/css" >
</style><table id="T_39eeb58c_ff49_11e9_a102_14109fdfaded" ><caption>Loaded Packages and Handles</caption><thead>    <tr>        <th class="col_heading level0 col0" >Package</th>        <th class="col_heading level0 col1" >Handle</th>        <th class="col_heading level0 col2" >Description</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow0_col0" class="data row0 col0" >IPython.display</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow0_col1" class="data row0 col1" >dp</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow0_col2" class="data row0 col2" >Display modules with helpful display and clearing commands.</td>
            </tr>
            <tr>
                                <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow1_col0" class="data row1 col0" >fsds_100719</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow1_col1" class="data row1 col1" >fs</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow1_col2" class="data row1 col2" >Custom data science bootcamp student package</td>
            </tr>
            <tr>
                                <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow2_col0" class="data row2 col0" >matplotlib</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow2_col1" class="data row2 col1" >mpl</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow2_col2" class="data row2 col2" >Matplotlib's base OOP module with formatting artists</td>
            </tr>
            <tr>
                                <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow3_col0" class="data row3 col0" >matplotlib.pyplot</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow3_col1" class="data row3 col1" >plt</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow3_col2" class="data row3 col2" >Matplotlib's matlab-like plotting module</td>
            </tr>
            <tr>
                                <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow4_col0" class="data row4 col0" >numpy</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow4_col1" class="data row4 col1" >np</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow4_col2" class="data row4 col2" >scientific computing with Python</td>
            </tr>
            <tr>
                                <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow5_col0" class="data row5 col0" >pandas</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow5_col1" class="data row5 col1" >pd</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow5_col2" class="data row5 col2" >High performance data structures and tools</td>
            </tr>
            <tr>
                                <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow6_col0" class="data row6 col0" >seaborn</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow6_col1" class="data row6 col1" >sns</td>
                        <td id="T_39eeb58c_ff49_11e9_a102_14109fdfadedrow6_col2" class="data row6 col2" >High-level data visualization library based on matplotlib</td>
            </tr>
    </tbody></table>



```python
#ignore pink warnings
import warnings
warnings.filterwarnings('ignore')

# Allow for large # columns
pd.set_option('display.max_columns', 0)
# pd.set_option('display.max_rows','')
```


```python
# import dataset and review data types, columns, values
df = pd.read_csv('kc_house_data.csv') 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



---
# SCRUB 

**Q1: How should we address datatype of each feature to prepare it for EDA?**
    + 1. find and replace nulls
    + 2. re-cast datatypes (continuous, binary, categorical)
    + 3. check for duplicate observations (rows)
    + 4. preliminary analysis and visualizations
    + 5. decide which columns and rows to drop before EDA

    Custom Functions: 
    * hot_stats()
    * null_hunter()
    * corr_dict()
    * sub_scatter()


```python
# HOT_STATS() function: display statistical summaries of a feature column
def hot_stats(data, column, verbose=False, target=None):
    """
    v.1.0
    Scans the values of a column within a dataframe
    and displays its datatype, nulls (incl. pct of total), 
    unique values, non-null value counts, as well as 
    statistical info if the datatype is numeric.
    
    Args:
    data: a dataframe
    
    column: a column within the data you want to scan *should be inside quotes ''
    
    KWargs:
    
    verbose: (optional) accepts a boolean (default=False). 
              verbose=True will display all unique values found.   
    
    target: (optional) accepts column name similar to column arg. 
             calculates correlation coefficient between feature and target using pandas data.corr() function. 
    
    example: 
    hot_stats(df, 'price')
    data = df
    column = 'price'
    
    Developer notes: additional features to add to v2 might
    include:
    -mode finder (frequency dict)
    -more functionality for string objects
    -side by side comparison between two features
    -OR ability to pass multiple columns at once and display all
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
    elif feature.dtype == 'object' or 'category':
        hot_stats = feature.agg(['min','median','max'])
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
    print(hot_stats)
    
    
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
        
    # display correlation coefficient with target
    if target != None:
        corr = feature.corr(data[target]).round(4)
        print(f"Correlation with {target.upper()}: {corr}")
```


```python
# NULL_HUNTER() function: display Null counts per column/feature
def null_hunter(df):
    print(f"Columns with Null Values")
    print("------------------------")
    for column in df:
        if df[column].isna().sum() > 0:
            print(f"{df[column].name}: \n{df[column].isna().sum()} out of {len(df[column])} ({round(df[column].isna().sum()/len(df[column])*100,2)}%)\n")
```


```python
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
```


```python
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
```


```python
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
```


```python
# PLOT_REG() function: plot regression
def plot_reg(data, feature, target):
    sns.regplot(x=feature, y=target, data=data)
    plt.show()
```

**FIRST GLANCE - Items to note**
    * There are 2 object datatypes that contain numeric values : 'date', 'sqft_basement'
    * The total value count is 21597. Some columns appear to be missing a substantial amount of data 
    (waterfront and yr_renovated).


```python
# Display information about the variables, columns and datatypes
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21597 non-null int64
    date             21597 non-null object
    price            21597 non-null float64
    bedrooms         21597 non-null int64
    bathrooms        21597 non-null float64
    sqft_living      21597 non-null int64
    sqft_lot         21597 non-null int64
    floors           21597 non-null float64
    waterfront       19221 non-null float64
    view             21534 non-null float64
    condition        21597 non-null int64
    grade            21597 non-null int64
    sqft_above       21597 non-null int64
    sqft_basement    21597 non-null object
    yr_built         21597 non-null int64
    yr_renovated     17755 non-null float64
    zipcode          21597 non-null int64
    lat              21597 non-null float64
    long             21597 non-null float64
    sqft_living15    21597 non-null int64
    sqft_lot15       21597 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB


Before going further, a little house-keeping is in order. Let's breakdown the columns into groups based on feature-type as they relate to a real estate market context:

*Dependent Variable:*

TARGET
**price**

*Independent Variables:*

INTERIOR
**bedrooms, bathrooms, floors**

SIZE (SQUARE FOOTAGE)
**sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15**

LOCATION
**zipcode, lat, long, waterfront**

QUALITY
**condition, grade, yr_built, yr_renovated**

ANALYTICS
**date, id, view**

## Null Hunting

    Let's start by hunting for nulls and then decide how to address each one (remove or replace) 
    in a manner that will not skew our analysis.

    NOTES:
      Drop null rows or columns as appropriate
      * df.isna().sum()
      * df.drop()
      * df.drop(['col1','col2'],axis=1)

      Coarse Binning NUMERICAL Data
      * replace with median or bin/convert to categorical
           * bin yr_built
           * bin sqft_basement
           * bin sqft_above
          
      CATEGORICAL data: 
      * make NaN own category OR replace with most common category
      * Fill in null values and recast variables for EDA
           * zipcode --> coded
           * View --> category
           * Waterfront --> boolean
           * yr_renovated --> is_reno (boolean)


```python
# hunt for nulls
null_hunter(df)            
```

    Columns with Null Values
    ------------------------
    waterfront: 
    2376 out of 21597 (11.0%)
    
    view: 
    63 out of 21597 (0.29%)
    
    yr_renovated: 
    3842 out of 21597 (17.79%)
    


Before deciding how to handle nulls in the 3 columns above, let's take a closer look at each one and go from there.

## Data Casting (hot_stats)

### Binaries (Boolean)

Although waterfront, yr_renovated, and view all contain numeric values, if we consider what each of them represents, they're more likely to be useful to the analysis if we convert them into binaries (is renovated or not, is waterfront or not, was viewed or not).

#### ['waterfront']


```python
hot_stats(df, 'waterfront')
```

    -------->
    HOT!STATS
    <--------
    
    WATERFRONT
    Data Type: float64
    
    count    19221.00
    mean         0.01
    std          0.09
    min          0.00
    25%          0.00
    50%          0.00
    75%          0.00
    max          1.00
    Name: waterfront, dtype: float64
    Found
    2376 Nulls out of 21597(11.0%)
    
    Non-Null Value Counts:
    0.0    19075
    1.0      146
    Name: waterfront, dtype: int64
    
    # Unique Values: 3
    



```python
# This really should be a boolean (property either is waterfront or is not waterfront)

# Fill nulls with most common value (0.0) # float value
df['waterfront'].fillna(0.0, inplace=True)
#  verify changes
df['waterfront'].isna().sum()
```




    0




```python
# Convert datatype to boolean (values can be either 0 or 1)
df['is_waterfront'] = df['waterfront'].astype('bool')
# verify
df['is_waterfront'].value_counts()
```




    False    21451
    True       146
    Name: is_waterfront, dtype: int64



#### ['yr_renovated']


```python
hot_stats(df, 'yr_renovated')
```

    -------->
    HOT!STATS
    <--------
    
    YR_RENOVATED
    Data Type: float64
    
    count    17755.00
    mean        83.64
    std        399.95
    min          0.00
    25%          0.00
    50%          0.00
    75%          0.00
    max       2015.00
    Name: yr_renovated, dtype: float64
    Found
    3842 Nulls out of 21597(17.79%)
    
    Non-Null Value Counts:
    0.0       17011
    2014.0       73
    2003.0       31
    2013.0       31
    2007.0       30
              ...  
    1946.0        1
    1959.0        1
    1971.0        1
    1951.0        1
    1954.0        1
    Name: yr_renovated, Length: 70, dtype: int64
    
    # Unique Values: 71
    



```python
# This feature is also heavily skewed with zero values. 
# It should also be treated as a boolean since a property is either renovated or it's not).

# fill nulls with most common value (0)
df['yr_renovated'].fillna(0.0, inplace=True) # use float value to match current dtype

# verify change
df['yr_renovated'].isna().sum()
```




    0




```python
# Use numpy arrays to create binarized column 'is_renovated'
is_renovated = np.array(df['yr_renovated'])
is_renovated[is_renovated >= 1] = 1
df['is_renovated'] = is_renovated
```


```python
df['is_renovated'].value_counts()
```




    0.0    20853
    1.0      744
    Name: is_renovated, dtype: int64




```python
# Convert to boolean
df['is_renovated'] = df['is_renovated'].astype('bool')

# verify
df['is_renovated'].value_counts()
```




    False    20853
    True       744
    Name: is_renovated, dtype: int64



#### ['view']


```python
hot_stats(df, 'view')
```

    -------->
    HOT!STATS
    <--------
    
    VIEW
    Data Type: float64
    
    count    21534.00
    mean         0.23
    std          0.77
    min          0.00
    25%          0.00
    50%          0.00
    75%          0.00
    max          4.00
    Name: view, dtype: float64
    Found
    63 Nulls out of 21597(0.29%)
    
    Non-Null Value Counts:
    0.0    19422
    2.0      957
    3.0      508
    1.0      330
    4.0      317
    Name: view, dtype: int64
    
    # Unique Values: 6
    



```python
# Once again, almost all values are 0 .0

# replace nulls with most common value (0). 
df['view'].fillna(0, inplace=True)

#verify
df['view'].isna().sum()
```




    0



Since view has a finite set of values (0 to 4) we could assign category codes. However, considering the high number of zeros, it makes more sense to binarize the values into a new column representing whether or not the property was viewed.


```python
# create new boolean column for view:
df['viewed'] = df['view'].astype('bool')

# verify
df['viewed'].dtype
```




    dtype('bool')




```python
binaries = ['is_waterfront', 'is_renovated', 'viewed']

# check correlation coefficients
corr_dict(binaries, 'price')
```




    {'is_waterfront': 0.2643062804831157,
     'is_renovated': 0.11754308700194362,
     'viewed': 0.3562431893938032}



None of these correlation values look strong enough to be predictive of price (min threshold > 0.5, ideally 0.7)

### Categories - Nominal

#### ['floors']


```python
hot_stats(df, 'floors', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    FLOORS
    Data Type: float64
    
    count    21597.00
    mean         1.49
    std          0.54
    min          1.00
    25%          1.00
    50%          1.50
    75%          2.00
    max          3.50
    Name: floors, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    1.0    10673
    2.0     8235
    1.5     1910
    3.0      611
    2.5      161
    3.5        7
    Name: floors, dtype: int64
    
    # Unique Values: 6
    
    Correlation with PRICE: 0.2568


Although you could theoretically have any number of floors, this really should be treated as a category (i.e. contains finite possible values). Assuming this is probably true for the other interior features (bedrooms, bathrooms). Let's look at a scatter plot using the sub_scatter function.


```python
# sub_scatter() creates scatter plots for multiple features side by side.
y = 'price'
x_cols = ['floors','bedrooms', 'bathrooms']

sub_scatter(df, x_cols, y)
```


![png](output_44_0.png)


Bathrooms appears to have a very linear relationship with price. Bedrooms is somewhat linear up to a certain point. Let's look at the hot stats for both.

#### ['bedrooms']


```python
hot_stats(df, 'bedrooms', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    BEDROOMS
    Data Type: int64
    
    count    21597.000000
    mean         3.373200
    std          0.926299
    min          1.000000
    25%          3.000000
    50%          3.000000
    75%          4.000000
    max         33.000000
    Name: bedrooms, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    3     9824
    4     6882
    2     2760
    5     1601
    6      272
    1      196
    7       38
    8       13
    9        6
    10       3
    11       1
    33       1
    Name: bedrooms, dtype: int64
    
    # Unique Values: 12
    
    Correlation with PRICE: 0.3088


Not a significant correlation between bedrooms and price.

#### ['bathrooms']


```python
hot_stats(df, 'bathrooms', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    BATHROOMS
    Data Type: float64
    
    count    21597.00
    mean         2.12
    std          0.77
    min          0.50
    25%          1.75
    50%          2.25
    75%          2.50
    max          8.00
    Name: bathrooms, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    2.50    5377
    1.00    3851
    1.75    3048
    2.25    2047
    2.00    1930
    1.50    1445
    2.75    1185
    3.00     753
    3.50     731
    3.25     589
    3.75     155
    4.00     136
    4.50     100
    4.25      79
    0.75      71
    4.75      23
    5.00      21
    5.25      13
    5.50      10
    1.25       9
    6.00       6
    5.75       4
    0.50       4
    8.00       2
    6.25       2
    6.75       2
    6.50       2
    7.50       1
    7.75       1
    Name: bathrooms, dtype: int64
    
    # Unique Values: 29
    
    Correlation with PRICE: 0.5259


Bathrooms is the only feature showing correlation over the 0.5 threshold. We can probably drop the other two as predictor candidates, especially if they exhbit multicollinearity (we'll explore that in more detail later).


```python
# save correlation coefficients higher than 0.5 in a dict
corr_thresh_dict = {}
corrs = ['bathrooms']
corr_thresh_dict = corr_dict(corrs, 'price')
corr_thresh_dict
```




    {'bathrooms': 0.5259056214532012}




```python
# Create category columns for interior features: floors, bedrooms, and bathrooms
df['floor_cat'] = df['floors'].astype('category')
df['bedroom_cat'] = df['bedrooms'].astype('category')
df['bathroom_cat'] = df['bathrooms'].astype('category')
```

### Categories - Ordinal

#### ['condition']


```python
hot_stats(df, 'condition', target='price')
```

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
    
    No Nulls Found!
    
    Non-Null Value Counts:
    3    14020
    4     5677
    5     1701
    2      170
    1       29
    Name: condition, dtype: int64
    
    # Unique Values: 5
    
    Correlation with PRICE: 0.0361



```python
# Condition should be treated as ordinal since there is a relationship between 
# the values (ranking scale from 1 to 5)
cat_dtype = pd.api.types.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True)

# Recast condition as an ordered category
df['condition_cat'] = df['condition'].astype(cat_dtype)

# verify
df['condition_cat'].dtype
```




    CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True)



#### ['grade']


```python
# View grade stats
hot_stats(df, 'grade', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    GRADE
    Data Type: int64
    
    count    21597.000000
    mean         7.657915
    std          1.173200
    min          3.000000
    25%          7.000000
    50%          7.000000
    75%          8.000000
    max         13.000000
    Name: grade, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    7     8974
    8     6065
    9     2615
    6     2038
    10    1134
    11     399
    5      242
    12      89
    4       27
    13      13
    3        1
    Name: grade, dtype: int64
    
    # Unique Values: 11
    
    Correlation with PRICE: 0.668



```python
corrs.append('grade')

corr_thresh_dict = corr_dict(corrs, 'price')

corr_thresh_dict
```




    {'bathrooms': 0.5259056214532012, 'grade': 0.6679507713876452}



    Condition is useless, but grade is our highest coefficient so far at 0.68! 
    We can also see (below) a clear linear relationship between grade and price. 


```python
# visualize on a scatter plot
sub_scatter(df, ['condition', 'grade'], 'price', nrows=1, ncols=2)
```


![png](output_62_0.png)



```python
# Grade should also be treated as an ordinal category

# create ranking scale from 1 to 13
cat_dtype = pd.api.types.CategoricalDtype(categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ordered=True)

# Create new ordered category column for grade
df['grade_cat'] = df['grade'].astype(cat_dtype)

# verify
df['grade_cat'].dtype
```




    CategoricalDtype(categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ordered=True)



#### ['yr_built'] 


```python
hot_stats(df, 'yr_built', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    YR_BUILT
    Data Type: int64
    
    count    21597.000000
    mean      1970.999676
    std         29.375234
    min       1900.000000
    25%       1951.000000
    50%       1975.000000
    75%       1997.000000
    max       2015.000000
    Name: yr_built, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    2014    559
    2006    453
    2005    450
    2004    433
    2003    420
           ... 
    1933     30
    1901     29
    1902     27
    1935     24
    1934     21
    Name: yr_built, Length: 116, dtype: int64
    
    # Unique Values: 116
    
    Correlation with PRICE: 0.054



```python
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
```


![png](output_66_0.png)


    Most houses were built during the second half of the century (after 1950)

    We'll use adaptive binning based on quantiles for yr_built in order to create
    a more normal distribution.


```python
# define a binning scheme with custom ranges based on quantiles
quantile_list = [0, .25, .5, .75, 1.]

quantiles = df['yr_built'].quantile(quantile_list)

quantiles # 1900, 1951, 1975, 1997, 2015
```




    0.00    1900.0
    0.25    1951.0
    0.50    1975.0
    0.75    1997.0
    1.00    2015.0
    Name: yr_built, dtype: float64




```python
# Bin the years in to ranges based on the quantiles.
# label the bins for each value 
# store the yr_range and its corresponding yr_label as new columns in df

yb_bins = [1900, 1951, 1975, 1997, 2015]

yb_labels = [1, 2, 3, 4]

# create a new column for the category range values
df['yb_range'] = pd.cut(df['yr_built'], bins=yb_bins)

# create a new column for the category labels
df['yb_cat'] = pd.cut(df['yr_built'], bins=yb_bins, labels=yb_labels)
```


```python
# view the binned features corresponding to each yr_built 
df[['yr_built','yb_cat', 'yb_range']].iloc[9003:9007] # picking a random index location
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yr_built</th>
      <th>yb_cat</th>
      <th>yb_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9003</td>
      <td>1996</td>
      <td>3</td>
      <td>(1975, 1997]</td>
    </tr>
    <tr>
      <td>9004</td>
      <td>1959</td>
      <td>2</td>
      <td>(1951, 1975]</td>
    </tr>
    <tr>
      <td>9005</td>
      <td>2003</td>
      <td>4</td>
      <td>(1997, 2015]</td>
    </tr>
    <tr>
      <td>9006</td>
      <td>1902</td>
      <td>1</td>
      <td>(1900, 1951]</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


![png](output_71_0.png)



```python
# values look much more normally distributed between the new categories
df.yb_cat.value_counts()
```




    2    5515
    3    5411
    1    5326
    4    5258
    Name: yb_cat, dtype: int64




```python
# visualize the distribution of the binned values

fig, ax = plt.subplots()
df['yb_cat'].hist(bins=4, color='#68FDFE', edgecolor='black', grid=True, alpha=0.6)
ax.set_title('Year Built Categories Histogram', fontsize=12)
ax.set_xlabel('Year Built Binned Categories', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
```




    Text(0, 0.5, 'Frequency')




![png](output_73_1.png)


#### ['zipcode']


```python
hot_stats(df, 'zipcode')
```

    -------->
    HOT!STATS
    <--------
    
    ZIPCODE
    Data Type: int64
    
    count    21597.000000
    mean     98077.951845
    std         53.513072
    min      98001.000000
    25%      98033.000000
    50%      98065.000000
    75%      98118.000000
    max      98199.000000
    Name: zipcode, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    98103    602
    98038    589
    98115    583
    98052    574
    98117    553
            ... 
    98102    104
    98010    100
    98024     80
    98148     57
    98039     50
    Name: zipcode, Length: 70, dtype: int64
    
    # Unique Values: 70
    



```python
# There are 70 unique zipcode values
# Let's look at the data distribution of zipcode values 
fig, ax = plt.subplots()
df['zipcode'].hist(bins=7, color='#67F86F',
edgecolor='black', grid=True)
ax.set_title('Zipcode Histogram', fontsize=16)
ax.set_xlabel('Zipcodes', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
```




    Text(0, 0.5, 'Frequency')




![png](output_76_1.png)



```python
# Let’s define a binning scheme with custom ranges for the zipcode values 
# The bins will be created based on quantiles

quantile_list = [0, .25, .5, .75, 1.]

quantiles = df['zipcode'].quantile(quantile_list)

quantiles # 98001, 98033, 98065, 98118, 98199
```




    0.00    98001.0
    0.25    98033.0
    0.50    98065.0
    0.75    98118.0
    1.00    98199.0
    Name: zipcode, dtype: float64




```python
# Now we can label the bins for each value and store both the bin range 
# and its corresponding label.

zip_bins = [98000, 98033, 98065, 98118, 98200]

zip_labels = [1, 2, 3, 4]

df['zip_range'] = pd.cut(df['zipcode'], bins=zip_bins)

df['zip_cat'] = pd.cut(df['zipcode'], bins=zip_bins, labels=zip_labels)

# view the binned features 
df[['zipcode','zip_cat', 'zip_range']].iloc[9000:9005] # pick a random index
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zipcode</th>
      <th>zip_cat</th>
      <th>zip_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9000</td>
      <td>98092</td>
      <td>3</td>
      <td>(98065, 98118]</td>
    </tr>
    <tr>
      <td>9001</td>
      <td>98117</td>
      <td>3</td>
      <td>(98065, 98118]</td>
    </tr>
    <tr>
      <td>9002</td>
      <td>98144</td>
      <td>4</td>
      <td>(98118, 98200]</td>
    </tr>
    <tr>
      <td>9003</td>
      <td>98038</td>
      <td>2</td>
      <td>(98033, 98065]</td>
    </tr>
    <tr>
      <td>9004</td>
      <td>98004</td>
      <td>1</td>
      <td>(98000, 98033]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# visualize the quantiles in the original distribution histogram

fig, ax = plt.subplots()

df['zipcode'].hist(bins=7, color='#67F86F', edgecolor='black', grid=True)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color='black')
    ax.legend([qvl], ['Quantiles'], fontsize=10)
    ax.set_title('Zipcode Histogram with Quantiles',fontsize=12)
    ax.set_xlabel('Zipcodes', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
```


![png](output_79_0.png)


#### ['lat']  ['long']

The coordinates for latitude and longitude are not going to be useful to us as far as regression models since we already have zipcodes as a geographic identifier (i.e. they'd be redundant). However, we may want to use them for creating some lmplots when we get to EDA.


```python
#%pip install zipcode
```

    [33mWARNING: The directory '/Users/hakkeray/Library/Caches/pip/http' or its parent directory is not owned by the current user and the cache has been disabled. Please check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.[0m
    [33mWARNING: The directory '/Users/hakkeray/Library/Caches/pip' or its parent directory is not owned by the current user and caching wheels has been disabled. check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.[0m
    Requirement already satisfied: zipcode in /Users/hakkeray/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages (4.0.0)
    Requirement already satisfied: haversine in /Users/hakkeray/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages (from zipcode) (2.1.2)
    Requirement already satisfied: SQLAlchemy in /Users/hakkeray/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages (from zipcode) (1.3.8)
    Note: you may need to restart the kernel to use updated packages.



```python
# pypi package for retrieving information based on us zipcodes
from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode=True) # set simple_zipcode=False to use rich info database
zipcode = search.by_zipcode("98199")
zipcode
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-50-77820cab01a1> in <module>
          1 # pypi package for retrieving information based on us zipcodes
    ----> 2 from uszipcode import SearchEngine
          3 search = SearchEngine(simple_zipcode=True) # set simple_zipcode=False to use rich info database
          4 zipcode = search.by_zipcode("98199")
          5 zipcode


    ModuleNotFoundError: No module named 'uszipcode'



```python

```


```python

SimpleZipcode(zipcode=u'10001', zipcode_type=u'Standard', major_city=u'New York', post_office_city=u'New York, NY', common_city_list=[u'New York'], county=u'New York County', state=u'NY', lat=40.75, lng=-73.99, timezone=u'Eastern', radius_in_miles=0.9090909090909091, area_code_list=[u'718', u'917', u'347', u'646'], population=21102, population_density=33959.0, land_area_in_sqmi=0.62, water_area_in_sqmi=0.0, housing_units=12476, occupied_housing_units=11031, median_home_value=650200, median_household_income=81671, bounds_west=-74.008621, bounds_east=-73.984076, bounds_north=40.759731, bounds_south=40.743451)

>>> zipcode.values() # to list
[u'10001', u'Standard', u'New York', u'New York, NY', [u'New York'], u'New York County', u'NY', 40.75, -73.99, u'Eastern', 0.9090909090909091, [u'718', u'917', u'347', u'646'], 21102, 33959.0, 0.62, 0.0, 12476, 11031, 650200, 81671, -74.008621, -73.984076, 40.759731, 40.743451]

>>> zipcode.to_dict() # to dict
{'housing_units': 12476, 'post_office_city': u'New York, NY', 'bounds_east': -73.984076, 'county': u'New York County', 'population_density': 33959.0, 'radius_in_miles': 0.9090909090909091, 'timezone': u'Eastern', 'lng': -73.99, 'common_city_list': [u'New York'], 'zipcode_type': u'Standard', 'zipcode': u'10001', 'state': u'NY', 'major_city': u'New York', 'population': 21102, 'bounds_west': -74.008621, 'land_area_in_sqmi': 0.62, 'lat': 40.75, 'median_household_income': 81671, 'occupied_housing_units': 11031, 'bounds_north': 40.759731, 'bounds_south': 40.743451, 'area_code_list': [u'718', u'917', u'347', u'646'], 'median_home_value': 650200, 'water_area_in_sqmi': 0.0}

>>> zipcode.to_json() # to json
{
    "zipcode": "10001",
    "zipcode_type": "Standard",
    "major_city": "New York",
    "post_office_city": "New York, NY",
    "common_city_list": [
        "New York"
    ],
    "county": "New York County",
    "state": "NY",
    "lat": 40.75,
    "lng": -73.99,
    "timezone": "Eastern",
    "radius_in_miles": 0.9090909090909091,
    "area_code_list": [
        "718",
        "917",
        "347",
        "646"
    ],
    "population": 21102,
    "population_density": 33959.0,
    "land_area_in_sqmi": 0.62,
    "water_area_in_sqmi": 0.0,
    "housing_units": 12476,
    "occupied_housing_units": 11031,
    "median_home_value": 650200,
    "median_household_income": 81671,
    "bounds_west": -74.008621,
    "bounds_east": -73.984076,
    "bounds_north": 40.759731,
    "bounds_south": 40.743451
}
```


```python
from uszipcode import Zipcode
# Search zipcode within 30 miles, ordered from closest to farthest
result = search.by_coordinates(39.122229, -77.133578, radius=30, returns=5)
len(res) # by default 5 results returned

for zipcode in result:
 # do whatever you want...
```

---
### DateTime
We need to convert the datatypes of numeric values currently stored as objects. This will allow us to include them in the preliminary exploration and analysis. 

    * date -> recast as datetime
    * sqft_basement -> recast as float

#### ['date'] 
convert to datetime


```python
df['date'] = pd.to_datetime(df['date'])
df['date'].dtype
```




    dtype('<M8[ns]')




```python
hot_stats(df, 'date')
```

    -------->
    HOT!STATS
    <--------
    
    DATE
    Data Type: datetime64[ns]
    
    min   2014-05-02
    max   2015-05-27
    Name: date, dtype: datetime64[ns]
    
    No Nulls Found!
    
    Non-Null Value Counts:
    2014-06-23    142
    2014-06-25    131
    2014-06-26    131
    2014-07-08    127
    2015-04-27    126
                 ... 
    2014-07-27      1
    2015-03-08      1
    2014-11-02      1
    2015-05-15      1
    2015-05-24      1
    Name: date, Length: 372, dtype: int64
    
    # Unique Values: 372
    


### Continuous

* SQUARE-FOOTAGE

Redundancy check: is there any overlap in the measurements?

* sqft_living = sqft_basement + sqft_above ?
* sqft_lot - sqft_living = sqft_above ?


**Continuous Variables:**
a continuous variable can take on any value within a range

* **sqft_living** --> highest corr : price (before transformation/scaling); most normal distribution
* sqft_lot --> not normal distribution
* sqft_living15 #Highly skewed
* sqft_lot15 #Highly skewed
* sqft_above #High corr with sqft_living
* sqft_basement #Very high number of null values --> treat '0' = no basement?


#### ['sqft_basement']


```python
# let's figure out what datatype is more appropriate for sqft_basement
hot_stats(df, 'sqft_basement')
```

    -------->
    HOT!STATS
    <--------
    
    SQFT_BASEMENT
    Data Type: object
    
    min    0.0
    max      ?
    Name: sqft_basement, dtype: object
    
    No Nulls Found!
    
    Non-Null Value Counts:
    0.0       12826
    ?           454
    600.0       217
    500.0       209
    700.0       208
              ...  
    862.0         1
    172.0         1
    784.0         1
    516.0         1
    2580.0        1
    Name: sqft_basement, Length: 304, dtype: int64
    
    # Unique Values: 304
    



```python
# Note the majority of the values are zero...we could bin this as a binary 
# where the property either has a basement or does not...

# For now we'll replace '?'s with string value '0.0'
df['sqft_basement'].replace(to_replace='?', value='0.0', inplace=True)
```


```python
# and change datatype to float.
df['sqft_basement'] = df['sqft_basement'].astype('float')
```


```python
hot_stats(df, 'sqft_basement', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    SQFT_BASEMENT
    Data Type: float64
    
    count    21597.00
    mean       285.72
    std        439.82
    min          0.00
    25%          0.00
    50%          0.00
    75%        550.00
    max       4820.00
    Name: sqft_basement, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    0.0       13280
    600.0       217
    500.0       209
    700.0       208
    800.0       201
              ...  
    915.0         1
    295.0         1
    1281.0        1
    2130.0        1
    906.0         1
    Name: sqft_basement, Length: 303, dtype: int64
    
    # Unique Values: 303
    
    Correlation with PRICE: 0.3211


    That's much better. Although, lot's of zeros could be a problem. 
    Let's check out the other square-footage features and see if those might be more useful.

#### ['sqft_above']


```python
hot_stats(df, 'sqft_above', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    SQFT_ABOVE
    Data Type: int64
    
    count    21597.000000
    mean      1788.596842
    std        827.759761
    min        370.000000
    25%       1190.000000
    50%       1560.000000
    75%       2210.000000
    max       9410.000000
    Name: sqft_above, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    1300    212
    1010    210
    1200    206
    1220    192
    1140    184
           ... 
    2601      1
    440       1
    2473      1
    2441      1
    1975      1
    Name: sqft_above, Length: 942, dtype: int64
    
    # Unique Values: 942
    
    Correlation with PRICE: 0.6054


    Some correlation with price here!

#### ['sqft_living']


```python
hot_stats(df, 'sqft_living', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    SQFT_LIVING
    Data Type: int64
    
    count    21597.000000
    mean      2080.321850
    std        918.106125
    min        370.000000
    25%       1430.000000
    50%       1910.000000
    75%       2550.000000
    max      13540.000000
    Name: sqft_living, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    1300    138
    1400    135
    1440    133
    1660    129
    1010    129
           ... 
    4970      1
    2905      1
    2793      1
    4810      1
    1975      1
    Name: sqft_living, Length: 1034, dtype: int64
    
    # Unique Values: 1034
    
    Correlation with PRICE: 0.7019


    sqft_living shows correlation value of 0.7 with price -- our highest coefficient yet!


```python
# add sqft_living and sqft_above to our correlation threshold dict
corrs = ['sqft_living', 'grade', 'sqft_above', 'bathrooms']
corr_thresh_dict = corr_dict(corrs, 'price')
corr_thresh_dict
```




    {'sqft_living': 0.7019173021377595,
     'grade': 0.6679507713876452,
     'sqft_above': 0.6053679437051795,
     'bathrooms': 0.5259056214532012}



#### ['sqft_lot']


```python
hot_stats(df, 'sqft_lot', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    SQFT_LOT
    Data Type: int64
    
    count    2.159700e+04
    mean     1.509941e+04
    std      4.141264e+04
    min      5.200000e+02
    25%      5.040000e+03
    50%      7.618000e+03
    75%      1.068500e+04
    max      1.651359e+06
    Name: sqft_lot, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    5000      358
    6000      290
    4000      251
    7200      220
    7500      119
             ... 
    1448        1
    38884       1
    17313       1
    35752       1
    315374      1
    Name: sqft_lot, Length: 9776, dtype: int64
    
    # Unique Values: 9776
    
    Correlation with PRICE: 0.0899


#### ['sqft_living15']


```python
hot_stats(df, 'sqft_living15', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    SQFT_LIVING15
    Data Type: int64
    
    count    21597.000000
    mean      1986.620318
    std        685.230472
    min        399.000000
    25%       1490.000000
    50%       1840.000000
    75%       2360.000000
    max       6210.000000
    Name: sqft_living15, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    1540    197
    1440    195
    1560    192
    1500    180
    1460    169
           ... 
    4890      1
    2873      1
    952       1
    3193      1
    2049      1
    Name: sqft_living15, Length: 777, dtype: int64
    
    # Unique Values: 777
    
    Correlation with PRICE: 0.5852


    We've identified another coefficient over the 0.5 correlation threshold.


```python
hot_stats(df, 'sqft_lot15', target='price')
```

    -------->
    HOT!STATS
    <--------
    
    SQFT_LOT15
    Data Type: int64
    
    count     21597.000000
    mean      12758.283512
    std       27274.441950
    min         651.000000
    25%        5100.000000
    50%        7620.000000
    75%       10083.000000
    max      871200.000000
    Name: sqft_lot15, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    5000      427
    4000      356
    6000      288
    7200      210
    4800      145
             ... 
    11036       1
    8989        1
    871200      1
    809         1
    6147        1
    Name: sqft_lot15, Length: 8682, dtype: int64
    
    # Unique Values: 8682
    
    Correlation with PRICE: 0.0828



```python
corrs.append('sqft_living15')

corr_thresh_dict = corr_dict(corrs, 'price')
corr_thresh_dict
```




    {'sqft_living': 0.7019173021377595,
     'grade': 0.6679507713876452,
     'sqft_above': 0.6053679437051795,
     'bathrooms': 0.5259056214532012,
     'sqft_living15': 0.5852412017040663}



### Index

#### ['id']


```python
hot_stats(df, 'id')
```

    -------->
    HOT!STATS
    <--------
    
    ID
    Data Type: int64
    
    count    2.159700e+04
    mean     4.580474e+09
    std      2.876736e+09
    min      1.000102e+06
    25%      2.123049e+09
    50%      3.904930e+09
    75%      7.308900e+09
    max      9.900000e+09
    Name: id, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    795000620     3
    1825069031    2
    2019200220    2
    7129304540    2
    1781500435    2
                 ..
    7812801125    1
    4364700875    1
    3021059276    1
    880000205     1
    1777500160    1
    Name: id, Length: 21420, dtype: int64
    
    # Unique Values: 21420
    


The primary key we'd use as an index for this data set would be 'id'. Our assumption therefore is that the 'id' for each observation (row) is unique. Let's do a quick scan for duplicate entries to confirm this is true.


```python
# check for duplicate id's
df['id'].duplicated().value_counts() 
```




    False    21420
    True       177
    Name: id, dtype: int64




```python
# Looks like there are in fact some duplicate ID's! Not many, but worth investigating.

# Let's flag the duplicate id's by creating a new column 'is_dupe':
df.loc[df.duplicated(subset='id', keep=False), 'is_dupe'] = 1 # mark all duplicates 

# verify all duplicates were flagged
df.is_dupe.value_counts() # 353
```




    1.0    353
    Name: is_dupe, dtype: int64




```python
# the non-duplicate rows show as null in our new column
df.is_dupe.isna().sum()
```




    21244




```python
# Replace 'nan' rows in is_dupe with 0.0
df.loc[df['is_dupe'].isna(), 'is_dupe'] = 0

# verify
df['is_dupe'].unique()
```




    array([0., 1.])




```python
# convert column to boolean data type
df['is_dupe'] = df['is_dupe'].astype('bool')
# verify
df['is_dupe'].value_counts()
```




    False    21244
    True       353
    Name: is_dupe, dtype: int64




```python
# Let's now copy the duplicates into a dataframe subset for closer inspection
# It's possible the pairs contain data missing from the other which 
# we can use to fill nulls identified previously.

df_dupes = df.loc[df['is_dupe'] == True]

# check out the data discrepancies between duplicates (first 3 pairs)
df_dupes.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>is_waterfront</th>
      <th>is_renovated</th>
      <th>viewed</th>
      <th>floor_cat</th>
      <th>bedroom_cat</th>
      <th>bathroom_cat</th>
      <th>condition_cat</th>
      <th>grade_cat</th>
      <th>yb_range</th>
      <th>yb_cat</th>
      <th>zip_range</th>
      <th>zip_cat</th>
      <th>is_dupe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>93</td>
      <td>6021501535</td>
      <td>2014-07-25</td>
      <td>430000.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1580</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1290</td>
      <td>290.0</td>
      <td>1939</td>
      <td>0.0</td>
      <td>98117</td>
      <td>47.6870</td>
      <td>-122.386</td>
      <td>1570</td>
      <td>4500</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>3</td>
      <td>8</td>
      <td>(1900, 1951]</td>
      <td>1</td>
      <td>(98065, 98118]</td>
      <td>3</td>
      <td>True</td>
    </tr>
    <tr>
      <td>94</td>
      <td>6021501535</td>
      <td>2014-12-23</td>
      <td>700000.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1580</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1290</td>
      <td>290.0</td>
      <td>1939</td>
      <td>0.0</td>
      <td>98117</td>
      <td>47.6870</td>
      <td>-122.386</td>
      <td>1570</td>
      <td>4500</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>3</td>
      <td>8</td>
      <td>(1900, 1951]</td>
      <td>1</td>
      <td>(98065, 98118]</td>
      <td>3</td>
      <td>True</td>
    </tr>
    <tr>
      <td>313</td>
      <td>4139480200</td>
      <td>2014-06-18</td>
      <td>1380000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>4290</td>
      <td>12103</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>11</td>
      <td>2690</td>
      <td>1600.0</td>
      <td>1997</td>
      <td>0.0</td>
      <td>98006</td>
      <td>47.5503</td>
      <td>-122.102</td>
      <td>3860</td>
      <td>11244</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>1.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>3</td>
      <td>11</td>
      <td>(1975, 1997]</td>
      <td>3</td>
      <td>(98000, 98033]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <td>314</td>
      <td>4139480200</td>
      <td>2014-12-09</td>
      <td>1400000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>4290</td>
      <td>12103</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>11</td>
      <td>2690</td>
      <td>1600.0</td>
      <td>1997</td>
      <td>0.0</td>
      <td>98006</td>
      <td>47.5503</td>
      <td>-122.102</td>
      <td>3860</td>
      <td>11244</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>1.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>3</td>
      <td>11</td>
      <td>(1975, 1997]</td>
      <td>3</td>
      <td>(98000, 98033]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <td>324</td>
      <td>7520000520</td>
      <td>2014-09-05</td>
      <td>232000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>1240</td>
      <td>12092</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>960</td>
      <td>280.0</td>
      <td>1922</td>
      <td>1984.0</td>
      <td>98146</td>
      <td>47.4957</td>
      <td>-122.352</td>
      <td>1820</td>
      <td>7460</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>3</td>
      <td>6</td>
      <td>(1900, 1951]</td>
      <td>1</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>True</td>
    </tr>
    <tr>
      <td>325</td>
      <td>7520000520</td>
      <td>2015-03-11</td>
      <td>240500.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>1240</td>
      <td>12092</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>960</td>
      <td>280.0</td>
      <td>1922</td>
      <td>1984.0</td>
      <td>98146</td>
      <td>47.4957</td>
      <td>-122.352</td>
      <td>1820</td>
      <td>7460</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>3</td>
      <td>6</td>
      <td>(1900, 1951]</td>
      <td>1</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Looks like the only discrepancies might occur between 'date' and 'price' values
# Some of the prices nearly double, even when the re-sale is just a few months later!

df_dupes.loc[df_dupes['id'] == 6021501535]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>is_waterfront</th>
      <th>is_renovated</th>
      <th>viewed</th>
      <th>floor_cat</th>
      <th>bedroom_cat</th>
      <th>bathroom_cat</th>
      <th>condition_cat</th>
      <th>grade_cat</th>
      <th>yb_range</th>
      <th>yb_cat</th>
      <th>zip_range</th>
      <th>zip_cat</th>
      <th>is_dupe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>93</td>
      <td>6021501535</td>
      <td>2014-07-25</td>
      <td>430000.0</td>
      <td>3</td>
      <td>1.5</td>
      <td>1580</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1290</td>
      <td>290.0</td>
      <td>1939</td>
      <td>0.0</td>
      <td>98117</td>
      <td>47.687</td>
      <td>-122.386</td>
      <td>1570</td>
      <td>4500</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>3</td>
      <td>1.5</td>
      <td>3</td>
      <td>8</td>
      <td>(1900, 1951]</td>
      <td>1</td>
      <td>(98065, 98118]</td>
      <td>3</td>
      <td>True</td>
    </tr>
    <tr>
      <td>94</td>
      <td>6021501535</td>
      <td>2014-12-23</td>
      <td>700000.0</td>
      <td>3</td>
      <td>1.5</td>
      <td>1580</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1290</td>
      <td>290.0</td>
      <td>1939</td>
      <td>0.0</td>
      <td>98117</td>
      <td>47.687</td>
      <td>-122.386</td>
      <td>1570</td>
      <td>4500</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>3</td>
      <td>1.5</td>
      <td>3</td>
      <td>8</td>
      <td>(1900, 1951]</td>
      <td>1</td>
      <td>(98065, 98118]</td>
      <td>3</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Set index of df_dupes to 'id'
df_dupes.set_index('id')
# Set index of df to 'id'
df.set_index('id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>is_waterfront</th>
      <th>is_renovated</th>
      <th>viewed</th>
      <th>floor_cat</th>
      <th>bedroom_cat</th>
      <th>bathroom_cat</th>
      <th>condition_cat</th>
      <th>grade_cat</th>
      <th>yb_range</th>
      <th>yb_cat</th>
      <th>zip_range</th>
      <th>zip_cat</th>
      <th>is_dupe</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7129300520</td>
      <td>2014-10-13</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>3</td>
      <td>7</td>
      <td>(1951, 1975]</td>
      <td>2</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6414100192</td>
      <td>2014-12-09</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>2.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>3</td>
      <td>7</td>
      <td>(1900, 1951]</td>
      <td>1</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <td>5631500400</td>
      <td>2015-02-25</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>3</td>
      <td>6</td>
      <td>(1900, 1951]</td>
      <td>1</td>
      <td>(98000, 98033]</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2487200875</td>
      <td>2014-12-09</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>5</td>
      <td>7</td>
      <td>(1951, 1975]</td>
      <td>2</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1954400510</td>
      <td>2015-02-18</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>3</td>
      <td>8</td>
      <td>(1975, 1997]</td>
      <td>3</td>
      <td>(98065, 98118]</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>263000018</td>
      <td>2014-05-21</td>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>0.0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>1530</td>
      <td>1509</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>3</td>
      <td>8</td>
      <td>(1997, 2015]</td>
      <td>4</td>
      <td>(98065, 98118]</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6600060120</td>
      <td>2015-02-23</td>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>0.0</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98146</td>
      <td>47.5107</td>
      <td>-122.362</td>
      <td>1830</td>
      <td>7200</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>3</td>
      <td>8</td>
      <td>(1997, 2015]</td>
      <td>4</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1523300141</td>
      <td>2014-06-23</td>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0.0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98144</td>
      <td>47.5944</td>
      <td>-122.299</td>
      <td>1020</td>
      <td>2007</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>3</td>
      <td>7</td>
      <td>(1997, 2015]</td>
      <td>4</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <td>291310100</td>
      <td>2015-01-16</td>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>0.0</td>
      <td>2004</td>
      <td>0.0</td>
      <td>98027</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>1410</td>
      <td>1287</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>3</td>
      <td>8</td>
      <td>(1997, 2015]</td>
      <td>4</td>
      <td>(98000, 98033]</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1523300157</td>
      <td>2014-10-15</td>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0.0</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98144</td>
      <td>47.5941</td>
      <td>-122.299</td>
      <td>1020</td>
      <td>1357</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>3</td>
      <td>7</td>
      <td>(1997, 2015]</td>
      <td>4</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 33 columns</p>
</div>




```python
# Before we drop the duplicates, let's save a backup copy of the current df using pickle.
import pickle
# create pickle data_object
df_predrops = df
```


```python
with open('data.pickle', 'wb') as f:
    pickle.dump(df_predrops, f, pickle.HIGHEST_PROTOCOL)
```


```python
#import df (pre-drops) with pickle
with open('data.pickle', 'rb') as f:
    df = pickle.load(f)
```


```python
# let's drop the first occurring duplicate rows and keep the last ones 
# (since those more accurately reflect latest market data)

# save original df.shape for comparison after dropping duplicate rows
predrop = df.shape # (21597, 28)

# first occurrence, keep last
df.drop_duplicates(subset='id', keep ='last', inplace = True) 

# verify dropped rows by comparing df.shape before and after values
print(f"predrop: {predrop}")
print(f"postdrop: {df.shape}")
```

    predrop: (21597, 34)
    postdrop: (21420, 34)


## Target

#### ['price']


```python
# Let's take a quick look at the statistical data for our dependent variable (price):
hot_stats(df, 'price')
```

    -------->
    HOT!STATS
    <--------
    
    PRICE
    Data Type: float64
    
    count      21420.00
    mean      541861.43
    std       367556.94
    min        78000.00
    25%       324950.00
    50%       450550.00
    75%       645000.00
    max      7700000.00
    Name: price, dtype: float64
    
    No Nulls Found!
    
    Non-Null Value Counts:
    450000.0    172
    350000.0    167
    550000.0    157
    500000.0    151
    425000.0    149
               ... 
    234975.0      1
    804995.0      1
    870515.0      1
    336950.0      1
    884744.0      1
    Name: price, Length: 3595, dtype: int64
    
    # Unique Values: 3595
    


Keeping the below numbers in mind could be helpful as we start exploring the data:

* range: 78,000 to 7,700,000
* mean value: 540,296
* median value: 450,000


```python
# At this point we can begin exploring the data. Let's first review our current feature list and get rid
# of any columns we no longer need. As we go through our analysis we'll decide which additional columns to 
# drop, transform, scale, normalize, etc.

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21420 entries, 0 to 21596
    Data columns (total 34 columns):
    id               21420 non-null int64
    date             21420 non-null datetime64[ns]
    price            21420 non-null float64
    bedrooms         21420 non-null int64
    bathrooms        21420 non-null float64
    sqft_living      21420 non-null int64
    sqft_lot         21420 non-null int64
    floors           21420 non-null float64
    waterfront       21420 non-null float64
    view             21420 non-null float64
    condition        21420 non-null int64
    grade            21420 non-null int64
    sqft_above       21420 non-null int64
    sqft_basement    21420 non-null float64
    yr_built         21420 non-null int64
    yr_renovated     21420 non-null float64
    zipcode          21420 non-null int64
    lat              21420 non-null float64
    long             21420 non-null float64
    sqft_living15    21420 non-null int64
    sqft_lot15       21420 non-null int64
    is_waterfront    21420 non-null bool
    is_renovated     21420 non-null bool
    viewed           21420 non-null bool
    floor_cat        21420 non-null category
    bedroom_cat      21420 non-null category
    bathroom_cat     21420 non-null category
    condition_cat    21420 non-null category
    grade_cat        21420 non-null category
    yb_range         21334 non-null category
    yb_cat           21334 non-null category
    zip_range        21420 non-null category
    zip_cat          21420 non-null category
    is_dupe          21420 non-null bool
    dtypes: bool(4), category(9), datetime64[ns](1), float64(9), int64(11)
    memory usage: 3.9 MB



```python
# cols to drop bc irrelevant to linreg model or using new versions instead:
hot_drop = ['date','id','waterfront', 'yr_renovated', 'view', 'yr_built', 'yb_range', 'zip_range']
```


```python
# store hot_drop columns in separate df
df_drops = df[hot_drop].copy()
```


```python
# set index of df_drops to 'id'
df_drops.set_index('id')
# verify
df_drops.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21420 entries, 0 to 21596
    Data columns (total 8 columns):
    date            21420 non-null datetime64[ns]
    id              21420 non-null int64
    waterfront      21420 non-null float64
    yr_renovated    21420 non-null float64
    view            21420 non-null float64
    yr_built        21420 non-null int64
    yb_range        21334 non-null category
    zip_range       21420 non-null category
    dtypes: category(2), datetime64[ns](1), float64(3), int64(2)
    memory usage: 1.2 MB



```python
# drop it like its hot >> df.drop(hot_drop, axis=1, inplace=True)
df.drop(hot_drop, axis=1, inplace=True)

# verify dropped columns
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21420 entries, 0 to 21596
    Data columns (total 26 columns):
    price            21420 non-null float64
    bedrooms         21420 non-null int64
    bathrooms        21420 non-null float64
    sqft_living      21420 non-null int64
    sqft_lot         21420 non-null int64
    floors           21420 non-null float64
    condition        21420 non-null int64
    grade            21420 non-null int64
    sqft_above       21420 non-null int64
    sqft_basement    21420 non-null float64
    zipcode          21420 non-null int64
    lat              21420 non-null float64
    long             21420 non-null float64
    sqft_living15    21420 non-null int64
    sqft_lot15       21420 non-null int64
    is_waterfront    21420 non-null bool
    is_renovated     21420 non-null bool
    viewed           21420 non-null bool
    floor_cat        21420 non-null category
    bedroom_cat      21420 non-null category
    bathroom_cat     21420 non-null category
    condition_cat    21420 non-null category
    grade_cat        21420 non-null category
    yb_cat           21334 non-null category
    zip_cat          21420 non-null category
    is_dupe          21420 non-null bool
    dtypes: bool(4), category(7), float64(6), int64(9)
    memory usage: 2.8 MB



```python
# cols kept for EDA:

# target variable
target = ['price']

# categorical:
cats = ['grade', 'condition', 'zipcode', 'bathrooms', 'bedrooms', 'floors']

#numeric/continuous:
nums = ['sqft_living', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot', 'sqft_lot15']

# binned:
binned = ['is_waterfront', 'is_renovated', 'viewed', 'is_dupe', 'yb_cat', 'zip_cat', 'bathroom_cat', 'bedroom_cat', 'floor_cat', 'grade_cat',  'condition_cat']

# dummies
#dummies = ['yb_2','yb_3', 'yb_4', 'zip_2', 'zip_3', 'zip_4']
```


```python
# Create dataframe subsets for each datatype group

df_binned = df[binned]
df_cats = df[cats]
df_nums = df[nums]
#df_dummies = df[dummies]
df_target = df[target]
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>is_waterfront</th>
      <th>is_renovated</th>
      <th>viewed</th>
      <th>floor_cat</th>
      <th>bedroom_cat</th>
      <th>bathroom_cat</th>
      <th>condition_cat</th>
      <th>grade_cat</th>
      <th>yb_cat</th>
      <th>zip_cat</th>
      <th>is_dupe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>3</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>2.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>5</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



# EXPLORE:
    
    EDA CHECKLIST:
    linearity (scatter matrices)
    multicollinearity (heatmaps)
    distributions (histograms, KDEs)
    regression (regplot)

**QUESTION: Which features are the best candidates for predicting property values?**
    * Continuous / Numeric Variables
    * Categorical / Discrete Variables
    * Binarized / Binned Variables

## Linearity
During the scrub process we made some assumptions and guesses based on correlation coefficients and other values. Let's see what the visualizations tell us by creating some scatter plots.


```python
plt.style.use('fivethirtyeight')
# add target to numeric df subset
df_nums['price'] = df.price

# plot scatter matrix 
pd.plotting.scatter_matrix(df_nums, figsize=(20,20));
```


![png](output_142_0.png)


We can see a clear linear relationship between price and sqft_living, sqft_above, sqft_living15 and somewhat so for sqft_basement. However, it looks like there is covariance among these variables as well.


```python
# Visualize the relationship between square-footages and price
sqft_int = ['sqft_living', 'sqft_above', 'sqft_basement']
sub_scatter(df, sqft_int, 'price', color='#0A2FC4') #20C5C6
# need to add titles etc
```


![png](output_144_0.png)


Linear relationships with price show up clearly for sqft_living, sqft_above, sqft_living15.


```python
# visualize relationship between sqft_lot, sqft_lot15, sqft_living15 and price.
y = 'price'
x_cols = ['sqft_living15', 'sqft_lot', 'sqft_lot15']

sub_scatter(df, x_cols, y, color='#6A76FB')
# need to add titles etc
```


![png](output_146_0.png)


## Multicollinearity
**QUESTION: which predictors are closely related (and should be dropped)?**

    + multicollinearity: remove variable having most corr with largest # of variables


```python
# Heatmap: Absolute Correlation Matrix
corr = df.drop('is_dupe', axis=1).corr()

plt.figure(figsize=(20,20))
sns.heatmap(data=corr.abs(), annot=True, cmap=sns.color_palette('Greens'))
plt.show()
```


![png](output_149_0.png)


The square footages probably overlap. (In other words sqft_above and sqft_basement could be part of the total sqft_living measurement).


```python
# Visualize multicollinearity between interior square-footages
x_cols = ['sqft_above', 'sqft_basement']
sub_scatter(df, x_cols, 'sqft_living', ncols = 2, color='#FD6F6B')  # lightred
```


![png](output_151_0.png)


    Yikes. These are extremely linear. Just for fun, let's crunch the numbers...

**QUESTION: Is there any overlap in square-footage measurements?**


```python
#DataFrame.equals(self, other)	Test whether two objects contain the same elements.

#DataFrame.sum(self[, axis, skipna, level, …])	Return the sum of the values for the requested axis.
```


```python
# check random location in the index
print(df['sqft_living'].iloc[0]) #1180
print(df['sqft_above'].iloc[0] + df['sqft_basement'].iloc[0]) #1180

print(df['sqft_living'].iloc[1]) #2570
print(df['sqft_above'].iloc[1] + df['sqft_basement'].iloc[1]) #2570
```

    1180
    1180.0
    2570
    2570.0



```python
# sqft_living == sqft_basement + sqft_above ?
# sqft_lot - sqft_living == sqft_above ?

sqft_lv = np.array(df['sqft_living'])
sqft_ab = np.array(df['sqft_above'])
sqft_bs = np.array(df['sqft_basement'])

sqft_ab + sqft_bs == sqft_lv #array([ True,  True,  True, ...,  True,  True,  True])
```




    array([ True,  True,  True, ...,  True,  True,  True])




```python
# check them all at once
if sqft_ab.all() + sqft_bs.all() == sqft_lv.all():
    print("True")
```

    True


**ANSWER: Yes. Sqft_living is the sum of sqft_above and sqft_basement.**

    Sqft_living15 (the square-footage of the neighbors' houses) looks to be the only one that correlates well with 
    price. However, as we saw above, this also correlates with square-foot living, so we'd run into some 
    multicollinearity issues if we kept both. Since sqft_living is higher, that is likely to be a better candidate 
    for prediction.

**QUESTION: Can we combine features for a higher correlation?**


```python
# Correlation coefficient for sqft_living and price
np.corrcoef(df['sqft_living'], df['price'])[0][1]
```




    0.701294859117587




```python
# Correlation coefficient for sqft_living15 and price
np.corrcoef(df['sqft_living15'], df['price'])[0][1]
```




    0.5837916994556072




```python
# Let's see if we can combine them for a higher correlation

weights = np.linspace(0, 1, 50)

best_weight = 0
max_corr = 0

for weight in weights:
    #creating a new feature by taking a weighted sum
    new_feature = weight*df['sqft_living'] + (1 - weight)*df['sqft_living15']
    
    corr_coef = np.corrcoef(new_feature, df['price'])[0][1]
    if np.abs(corr_coef) > max_corr:
        max_corr = np.abs(corr_coef)
        best_weight = weight
        
print(best_weight, 1 - best_weight)
```

    0.7755102040816326 0.22448979591836737


    The combined correlation value of sqft_living and sqft_living15 (0.77) is indeed higher
    than sqft_living by itself (0.7).


```python
# Let's see if we can also combine bedrooms and bathrooms for a higher correlation

weights = np.linspace(0, 1, 50)

best_weight = 0
max_corr = 0

for weight in weights:
    #creating a new feature by taking a weighted sum
    new_feature = weight*df['bathrooms'] + (1 - weight)*df['bedrooms']
    
    corr_coef = np.corrcoef(new_feature, df['price'])[0][1]
    if np.abs(corr_coef) > max_corr:
        max_corr = np.abs(corr_coef)
        best_weight = weight

print(best_weight, 1 - best_weight)
```

    0.9183673469387754 0.08163265306122458


    0.92 combined correlation for bathrooms and bedrooms!

**ANSWER: We could combine bathrooms/bedrooms as well as sqft_living and sqft_living15 to achieve higher correlation values with price.**
 
...How do we do this?

## Distributions

### Histograms


```python
# histogram subplots
def sub_hists(data):
    plt.style.use('fivethirtyeight')
    for column in data.describe():
        fig = plt.figure(figsize=(12, 5))
        
        ax = fig.add_subplot(121)
        ax.hist(data[column], density=True, label = column+' histogram', bins=20)
        ax.set_title(column.capitalize())

        ax.legend()
        
        fig.tight_layout()
```


```python
# hist plot
sub_hists(df_nums)
```


![png](output_170_0.png)



![png](output_170_1.png)



![png](output_170_2.png)



![png](output_170_3.png)



![png](output_170_4.png)



![png](output_170_5.png)


    Although sqft_living15 didn't have as much linearity with price as other candidates, it appears
    to have the most normal distribution out of all of them.

### KDEs


```python
# Kernel Density Estimates (distplots) for square-footage variables
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(12,12))
sns.distplot(df['sqft_living'], ax=ax[0][0])
sns.distplot(df['sqft_living15'], ax=ax[0][1])
sns.distplot(df['sqft_lot'], ax=ax[1][0])
sns.distplot(df['sqft_lot15'], ax=ax[1][1])
sns.distplot(df['sqft_above'], ax=ax[2][0])
sns.distplot(df['sqft_basement'], ax=ax[2][1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1dddbe80>




![png](output_173_1.png)



```python
fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(12,12))
sns.distplot(df['bathroom_cat'], ax=ax[0])
sns.distplot(df['bedroom_cat'], ax=ax[1])
sns.distplot(df['floor_cat'], ax=ax[2])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1e68ad30>




![png](output_174_1.png)



```python
fig, ax = plt.subplots(ncols=2, figsize=(12,4))
sns.distplot(df['grade'], ax=ax[0])
sns.distplot(df['zipcode'], ax=ax[1]) # look at actual zipcode value dist instead of category
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1e3496d8>




![png](output_175_1.png)


### Geographic

**QUESTION: Does geography (location) have any relationship with the values of each categorical variable?**


```python
# bring lat / long columns back in for plotting geographic distribution of ordinal categories

latlong = ['lat', 'long']
latlong_df = df[latlong]
geo_cats_df = pd.concat([df_cats, latlong_df], axis=1)
```


```python
# run (ordinal) categorical features through lmplot as a forloop to plot geographic distribution visual

for x in cats:
    sns.lmplot(data=geo_cats_df, x="long", y="lat", fit_reg=False, hue=x, height=10)
plt.show()
```


![png](output_179_0.png)



![png](output_179_1.png)



![png](output_179_2.png)



![png](output_179_3.png)



![png](output_179_4.png)



![png](output_179_5.png)


    The highest grade properties appear to be most dense in the upper left quadrant. 
    Since we already know that grade has a strong correlation with price, we can 
    posit more confidently that grade, location, and price are all related.
    
    Note: if we were to look at an actual map, we'd see this is Seattle.


```python
# binary/categorical:
#binned = ['is_waterfront', 'is_renovated', 'viewed', 'is_dupe']

# bring lat / long columns back in for plotting geographic distribution of ordinal categories

geo_bins = pd.concat([df_binned, latlong_df], axis=1)
geo_bins

# run binned features through lmplot as a forloop to plot geographic distribution visual
for x in binned:
    sns.lmplot(data=geo_bins, x="long", y="lat", fit_reg=False, hue=x, height=10)
plt.show()
```


![png](output_181_0.png)



![png](output_181_1.png)



![png](output_181_2.png)



![png](output_181_3.png)



![png](output_181_4.png)



![png](output_181_5.png)



![png](output_181_6.png)



![png](output_181_7.png)



![png](output_181_8.png)



![png](output_181_9.png)



![png](output_181_10.png)


    Some obvious but also some interesting things to observe in the above lmplots:
    
    The good news is that waterfront properties do indeed show up as being on the water, 
    so we can rest assured that data is valid. Unfortunately as we saw earlier, this doesn't 
    seem to correlate much with price. 
    
    'is_dupe' (which represents properties that sold twice in the 2 year period of this dataset)
    tells us pretty much nothing about anything. At least not on its own here.
    
    Probably the most surprising observation is 'viewed'. They almost all line up with the 
    coastline, or very close to the water. This may not mean anything but it is worth noting.
    
    Lastly, is_renovated is pretty densely clumped up in the northwest quadrant (again, Seattle).
    We can assume therefore that a lot of renovations are taking place in the city. Not entirely
    useful but worth mentioning neverthless.

### Box Plots


```python
x = df['grade_cat']
y = df['price']

plt.style.use('seaborn')
fig, ax = plt.subplots(ncols=1,figsize=(12,6))
sns.boxplot(x=x, y=y, ax=ax)
# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='x-large',   
                  rotation=45)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='Grade Categories Boxplot'
ax.set_title(title.title())
ax.set_xlabel('grade_cat')
ax.set_ylabel('price')
fig.tight_layout()
```


![png](output_184_0.png)



```python
x = df['grade_cat']
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

title='Grade Categories Boxplot'
ax.set_title(title.title())
ax.set_xlabel('grade_cat')
ax.set_ylabel('price')
fig.tight_layout()
```


![png](output_185_0.png)



```python
x = df['bathroom_cat']
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

title='Bathroom Categories Boxplot'
ax.set_title(title.title())
ax.set_xlabel('bathroom_cat')
ax.set_ylabel('price')
fig.tight_layout()
```


![png](output_186_0.png)



```python
x = df['bedroom_cat']
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

title='Bedroom Categories Boxplot'
ax.set_title(title.title())
ax.set_xlabel('bedroom_cat')
ax.set_ylabel('price')
fig.tight_layout()
```


![png](output_187_0.png)



```python
x = df['floor_cat']
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

title='Floor Categories Boxplot'
ax.set_title(title.title())
ax.set_xlabel('floor_cat')
ax.set_ylabel('price')
fig.tight_layout()
```


![png](output_188_0.png)



```python
x = df['zip_cat']
y = df['price']

plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots(ncols=1,figsize=(12,6))
sns.boxplot(x=x, y=y, ax=ax, showfliers=False) #outliers removed

# Create keywords for .set_xticklabels()
tick_kwds = dict(horizontalalignment='right', 
                  fontweight='light', 
                  fontsize='x-large',   
                  rotation=45)

ax.set_xticklabels(ax.get_xticklabels(),**tick_kwds)

title='zipcode categories boxplot'
ax.set_title(title.title())
ax.set_xlabel('zip_cat')
ax.set_ylabel('price')
fig.tight_layout()
```


![png](output_189_0.png)


    Category 3 for zipcode seems to be a good candidate for higher priced homes.


```python
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
```


![png](output_191_0.png)


    This tells us pretty much nothing other than newer homes sell at a higher price point. As far as making a and 
    prediction goes, we can only really use this to suggest building a new property rather than renovating.

## Regression Plots


```python
plt.style.use('fivethirtyeight')
def plot_reg(data, feature, target):
    sns.regplot(x=feature, y=target, data=data)
    plt.show()
```


```python
plot_reg(df, 'sqft_living', 'price')
```


![png](output_195_0.png)



```python
plot_reg(df, 'bathrooms', df.price)
```


![png](output_196_0.png)



```python
plot_reg(df, 'grade', df.price)
```


![png](output_197_0.png)


# FIT AN INITIAL MODEL:
Various forms, detail later...
Assessing the model:
Assess parameters (slope,intercept)
Check if the model explains the variation in the data (RMSE, F, R_square)
Are the coeffs, slopes, intercepts in appropriate units?
Whats the impact of collinearity? Can we ignore?
Revise the fitted model
Multicollinearity is big issue for lin regression and cannot fully remove it
Use the predictive ability of model to test it (like R2 and RMSE)
Check for missed non-linearity
Holdout validation / Train/test split
use sklearn train_test_split


```python
preds = ['zipcode', 'grade', 'sqft_living', 'sqft_living15', 'price']
```


```python
df_pred = df[preds].copy()
```


```python
# Import packages
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stats
# Enter equation for selected predictors: (use C to run as categorical)
f1 = 'price~C(zipcode)+C(grade)+sqft_living+sqft_living15'
# Run model and report sumamry
model = smf.ols(formula=f1, data=df_pred).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.784</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.783</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   956.5</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 04 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>13:54:51</td>     <th>  Log-Likelihood:    </th> <td>-2.8847e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21420</td>      <th>  AIC:               </th>  <td>5.771e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21338</td>      <th>  BIC:               </th>  <td>5.778e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    81</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>           <td>  8.19e+04</td> <td> 1.72e+05</td> <td>    0.477</td> <td> 0.634</td> <td>-2.55e+05</td> <td> 4.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th> <td> 2.172e+04</td> <td> 1.52e+04</td> <td>    1.427</td> <td> 0.154</td> <td>-8109.593</td> <td> 5.16e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th> <td> 2158.3824</td> <td> 1.37e+04</td> <td>    0.157</td> <td> 0.875</td> <td>-2.47e+04</td> <td> 2.91e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th> <td> 7.575e+05</td> <td> 1.34e+04</td> <td>   56.696</td> <td> 0.000</td> <td> 7.31e+05</td> <td> 7.84e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th> <td> 2.975e+05</td> <td> 1.61e+04</td> <td>   18.485</td> <td> 0.000</td> <td> 2.66e+05</td> <td> 3.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th> <td>  2.55e+05</td> <td> 1.21e+04</td> <td>   21.091</td> <td> 0.000</td> <td> 2.31e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th> <td> 2.498e+05</td> <td> 1.71e+04</td> <td>   14.584</td> <td> 0.000</td> <td> 2.16e+05</td> <td> 2.83e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th> <td> 3.069e+05</td> <td> 1.36e+04</td> <td>   22.524</td> <td> 0.000</td> <td>  2.8e+05</td> <td> 3.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th> <td> 7.095e+04</td> <td> 1.95e+04</td> <td>    3.647</td> <td> 0.000</td> <td> 3.28e+04</td> <td> 1.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th> <td> 1.216e+05</td> <td> 1.53e+04</td> <td>    7.956</td> <td> 0.000</td> <td> 9.17e+04</td> <td> 1.52e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th> <td> 1.035e+05</td> <td> 1.79e+04</td> <td>    5.773</td> <td> 0.000</td> <td> 6.84e+04</td> <td> 1.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th> <td> 8.251e+04</td> <td> 1.54e+04</td> <td>    5.366</td> <td> 0.000</td> <td> 5.24e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th> <td> 4.715e+04</td> <td> 1.44e+04</td> <td>    3.268</td> <td> 0.001</td> <td> 1.89e+04</td> <td> 7.54e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th> <td>-2.637e+04</td> <td> 1.19e+04</td> <td>   -2.215</td> <td> 0.027</td> <td>-4.97e+04</td> <td>-3037.729</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th> <td> 1.731e+05</td> <td> 2.13e+04</td> <td>    8.129</td> <td> 0.000</td> <td> 1.31e+05</td> <td> 2.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th> <td>  1.55e+05</td> <td> 1.24e+04</td> <td>   12.457</td> <td> 0.000</td> <td> 1.31e+05</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th> <td> 1.296e+05</td> <td> 1.36e+04</td> <td>    9.498</td> <td> 0.000</td> <td> 1.03e+05</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th> <td> 2.066e+05</td> <td> 1.33e+04</td> <td>   15.578</td> <td> 0.000</td> <td> 1.81e+05</td> <td> 2.33e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th> <td>  519.8491</td> <td> 1.41e+04</td> <td>    0.037</td> <td> 0.971</td> <td> -2.7e+04</td> <td> 2.81e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th> <td> 1.432e+04</td> <td> 1.38e+04</td> <td>    1.038</td> <td> 0.299</td> <td>-1.27e+04</td> <td> 4.13e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th> <td> 1.089e+04</td> <td> 1.79e+04</td> <td>    0.608</td> <td> 0.543</td> <td>-2.42e+04</td> <td>  4.6e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th> <td> 3.706e+05</td> <td> 1.23e+04</td> <td>   30.174</td> <td> 0.000</td> <td> 3.46e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th> <td> 2.124e+05</td> <td> 1.17e+04</td> <td>   18.214</td> <td> 0.000</td> <td>  1.9e+05</td> <td> 2.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th> <td> 2.541e+04</td> <td> 1.15e+04</td> <td>    2.210</td> <td> 0.027</td> <td> 2870.711</td> <td> 4.79e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th> <td> 1.242e+06</td> <td> 2.63e+04</td> <td>   47.154</td> <td> 0.000</td> <td> 1.19e+06</td> <td> 1.29e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th> <td> 5.483e+05</td> <td> 1.38e+04</td> <td>   39.633</td> <td> 0.000</td> <td> 5.21e+05</td> <td> 5.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th> <td> 8126.0998</td> <td> 1.16e+04</td> <td>    0.698</td> <td> 0.485</td> <td>-1.47e+04</td> <td> 3.09e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th> <td> 1.066e+05</td> <td> 1.47e+04</td> <td>    7.244</td> <td> 0.000</td> <td> 7.77e+04</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th> <td> 2.291e+05</td> <td> 1.16e+04</td> <td>   19.725</td> <td> 0.000</td> <td> 2.06e+05</td> <td> 2.52e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th> <td>  1.94e+05</td> <td> 1.25e+04</td> <td>   15.482</td> <td> 0.000</td> <td> 1.69e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th> <td> 4.777e+04</td> <td>  1.4e+04</td> <td>    3.424</td> <td> 0.001</td> <td> 2.04e+04</td> <td> 7.51e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th> <td> 1.057e+05</td> <td> 1.24e+04</td> <td>    8.492</td> <td> 0.000</td> <td> 8.13e+04</td> <td>  1.3e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th> <td> 3.174e+04</td> <td> 1.21e+04</td> <td>    2.618</td> <td> 0.009</td> <td> 7974.526</td> <td> 5.55e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th> <td> 6.707e+04</td> <td> 1.21e+04</td> <td>    5.547</td> <td> 0.000</td> <td> 4.34e+04</td> <td> 9.08e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th> <td> 7.795e+04</td> <td> 1.34e+04</td> <td>    5.807</td> <td> 0.000</td> <td> 5.16e+04</td> <td> 1.04e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th> <td> 2.035e+05</td> <td> 1.83e+04</td> <td>   11.122</td> <td> 0.000</td> <td> 1.68e+05</td> <td> 2.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th> <td>   1.5e+05</td> <td> 1.38e+04</td> <td>   10.857</td> <td> 0.000</td> <td> 1.23e+05</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th> <td> 1.686e+05</td> <td> 1.24e+04</td> <td>   13.634</td> <td> 0.000</td> <td> 1.44e+05</td> <td> 1.93e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th> <td> 1.638e+05</td> <td> 1.31e+04</td> <td>   12.528</td> <td> 0.000</td> <td> 1.38e+05</td> <td> 1.89e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th> <td> 9.536e+04</td> <td> 1.54e+04</td> <td>    6.209</td> <td> 0.000</td> <td> 6.53e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th> <td>-2.608e+04</td> <td> 1.29e+04</td> <td>   -2.024</td> <td> 0.043</td> <td>-5.13e+04</td> <td> -825.942</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th> <td> 4.855e+05</td> <td> 1.91e+04</td> <td>   25.372</td> <td> 0.000</td> <td> 4.48e+05</td> <td> 5.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th> <td> 3.578e+05</td> <td> 1.15e+04</td> <td>   31.178</td> <td> 0.000</td> <td> 3.35e+05</td> <td>  3.8e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th> <td> 5.046e+05</td> <td> 1.45e+04</td> <td>   34.807</td> <td> 0.000</td> <td> 4.76e+05</td> <td> 5.33e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th> <td> 1.294e+05</td> <td> 1.31e+04</td> <td>    9.850</td> <td> 0.000</td> <td> 1.04e+05</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th> <td> 3.635e+05</td> <td> 1.39e+04</td> <td>   26.091</td> <td> 0.000</td> <td> 3.36e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th> <td> 1.235e+05</td> <td> 1.55e+04</td> <td>    7.962</td> <td> 0.000</td> <td> 9.31e+04</td> <td> 1.54e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th> <td> 5.311e+05</td> <td> 1.88e+04</td> <td>   28.320</td> <td> 0.000</td> <td> 4.94e+05</td> <td> 5.68e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th> <td> 6.294e+05</td> <td> 1.39e+04</td> <td>   45.296</td> <td> 0.000</td> <td> 6.02e+05</td> <td> 6.57e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th> <td> 3.562e+05</td> <td> 1.15e+04</td> <td>   30.901</td> <td> 0.000</td> <td> 3.34e+05</td> <td> 3.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th> <td> 3.376e+05</td> <td> 1.31e+04</td> <td>   25.783</td> <td> 0.000</td> <td> 3.12e+05</td> <td> 3.63e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th> <td> 3.402e+05</td> <td> 1.17e+04</td> <td>   29.178</td> <td> 0.000</td> <td> 3.17e+05</td> <td> 3.63e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th> <td> 1.809e+05</td> <td> 1.19e+04</td> <td>   15.199</td> <td> 0.000</td> <td> 1.58e+05</td> <td> 2.04e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th> <td> 5.189e+05</td> <td> 1.56e+04</td> <td>   33.326</td> <td> 0.000</td> <td> 4.88e+05</td> <td> 5.49e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th> <td> 3.637e+05</td> <td> 1.36e+04</td> <td>   26.783</td> <td> 0.000</td> <td> 3.37e+05</td> <td>  3.9e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th> <td> 2.281e+05</td> <td> 1.24e+04</td> <td>   18.330</td> <td> 0.000</td> <td> 2.04e+05</td> <td> 2.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th> <td> 2.205e+05</td> <td> 1.29e+04</td> <td>   17.085</td> <td> 0.000</td> <td> 1.95e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th> <td> 1.745e+05</td> <td>  1.2e+04</td> <td>   14.601</td> <td> 0.000</td> <td> 1.51e+05</td> <td> 1.98e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th> <td> 3.005e+05</td> <td> 1.39e+04</td> <td>   21.594</td> <td> 0.000</td> <td> 2.73e+05</td> <td> 3.28e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th> <td> 3.035e+05</td> <td>  1.3e+04</td> <td>   23.397</td> <td> 0.000</td> <td> 2.78e+05</td> <td> 3.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th> <td> 1.412e+05</td> <td> 1.37e+04</td> <td>   10.313</td> <td> 0.000</td> <td> 1.14e+05</td> <td> 1.68e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th> <td> 7.346e+04</td> <td> 2.46e+04</td> <td>    2.986</td> <td> 0.003</td> <td> 2.52e+04</td> <td> 1.22e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th> <td> 1.686e+05</td> <td> 1.22e+04</td> <td>   13.845</td> <td> 0.000</td> <td> 1.45e+05</td> <td> 1.92e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th> <td> 1.333e+05</td> <td> 1.41e+04</td> <td>    9.434</td> <td> 0.000</td> <td> 1.06e+05</td> <td> 1.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th> <td> 5.181e+04</td> <td>  1.4e+04</td> <td>    3.707</td> <td> 0.000</td> <td> 2.44e+04</td> <td> 7.92e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th> <td> 2.578e+05</td> <td> 1.41e+04</td> <td>   18.304</td> <td> 0.000</td> <td>  2.3e+05</td> <td> 2.85e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th> <td> 7.333e+04</td> <td>  1.4e+04</td> <td>    5.234</td> <td> 0.000</td> <td> 4.59e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th> <td> 3.693e+04</td> <td> 1.73e+04</td> <td>    2.135</td> <td> 0.033</td> <td> 3033.328</td> <td> 7.08e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th> <td> 5.652e+04</td> <td> 1.37e+04</td> <td>    4.117</td> <td> 0.000</td> <td> 2.96e+04</td> <td> 8.34e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th> <td> 4.144e+05</td> <td> 1.32e+04</td> <td>   31.284</td> <td> 0.000</td> <td> 3.88e+05</td> <td>  4.4e+05</td>
</tr>
<tr>
  <th>C(grade)[T.4]</th>       <td>-1.329e+05</td> <td> 1.75e+05</td> <td>   -0.761</td> <td> 0.447</td> <td>-4.75e+05</td> <td> 2.09e+05</td>
</tr>
<tr>
  <th>C(grade)[T.5]</th>       <td>-1.617e+05</td> <td> 1.72e+05</td> <td>   -0.941</td> <td> 0.347</td> <td>-4.99e+05</td> <td> 1.75e+05</td>
</tr>
<tr>
  <th>C(grade)[T.6]</th>       <td>-1.812e+05</td> <td> 1.72e+05</td> <td>   -1.056</td> <td> 0.291</td> <td>-5.17e+05</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>C(grade)[T.7]</th>       <td>-1.921e+05</td> <td> 1.72e+05</td> <td>   -1.120</td> <td> 0.263</td> <td>-5.28e+05</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>C(grade)[T.8]</th>       <td>-1.735e+05</td> <td> 1.72e+05</td> <td>   -1.012</td> <td> 0.312</td> <td> -5.1e+05</td> <td> 1.63e+05</td>
</tr>
<tr>
  <th>C(grade)[T.9]</th>       <td>-9.851e+04</td> <td> 1.72e+05</td> <td>   -0.574</td> <td> 0.566</td> <td>-4.35e+05</td> <td> 2.38e+05</td>
</tr>
<tr>
  <th>C(grade)[T.10]</th>      <td>  4.03e+04</td> <td> 1.72e+05</td> <td>    0.235</td> <td> 0.814</td> <td>-2.96e+05</td> <td> 3.77e+05</td>
</tr>
<tr>
  <th>C(grade)[T.11]</th>      <td> 2.656e+05</td> <td> 1.72e+05</td> <td>    1.545</td> <td> 0.122</td> <td>-7.14e+04</td> <td> 6.03e+05</td>
</tr>
<tr>
  <th>C(grade)[T.12]</th>      <td> 7.521e+05</td> <td> 1.73e+05</td> <td>    4.352</td> <td> 0.000</td> <td> 4.13e+05</td> <td> 1.09e+06</td>
</tr>
<tr>
  <th>C(grade)[T.13]</th>      <td> 1.783e+06</td> <td> 1.79e+05</td> <td>    9.984</td> <td> 0.000</td> <td> 1.43e+06</td> <td> 2.13e+06</td>
</tr>
<tr>
  <th>sqft_living</th>         <td>  161.9099</td> <td>    2.285</td> <td>   70.871</td> <td> 0.000</td> <td>  157.432</td> <td>  166.388</td>
</tr>
<tr>
  <th>sqft_living15</th>       <td>   36.3883</td> <td>    3.022</td> <td>   12.040</td> <td> 0.000</td> <td>   30.464</td> <td>   42.312</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>20049.285</td> <th>  Durbin-Watson:     </th>  <td>   1.995</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>2674297.290</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 4.103</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>57.121</td>   <th>  Cond. No.          </th>  <td>1.50e+06</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.5e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



R-squared value: 0.78


```python
# Check normalized values with historgrams and scatter
```

## Outliers

**QUESTION: Does removing outliers improve the distribution?**

## Scaling / Normalization

The dataset's remaining features vary significantly in magnitude which will throw off the R2 coefficient (essentially giving the false impression that some variables are less important).


```python
# ADDING OUTLIER REMOVAL FROM preprocessing.RobuseScaler
from sklearn.preprocessing import RobustScaler

robscaler = RobustScaler()
robscaler
```




    RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
                 with_scaling=True)




```python
RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
             with_scaling=True)
```




    RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
                 with_scaling=True)




```python
scaled_vars = ['sqft_living','sqft_lot','sqft_living15','sqft_lot15','bedrooms','bathrooms']
df_scaled = df[scaled_vars]
```


```python
for col in scaled_vars:
            
    col_data = np.array(np.array(df_scaled[col]))
    res = robscaler.fit_transform(col_data.reshape(-1,1)) #,df['price'])
    df_scaled['sca_'+col] = res.flatten()
```

### One-Hot Encoding




```python
# get_dummies for one-hot encoding
yb_dummies = pd.get_dummies(df['yb_cat'], prefix='yb', drop_first=True)
df = pd.concat([df, yb_dummies], axis=1)
df.head(2)


# get_dummies for one-hot encoding
zip_dummies = pd.get_dummies(df['zip_cat'], prefix='zip', drop_first=True)
# drop first dummy to avoid multicollinearity trap
df = pd.concat([df, zip_dummies], axis=1)
# inspect
df.head()
```

### Log Transformation

Reduces skewness and variability of data to improve regression algorithm.


```python
# Log Transform

#x = np.linspace(start=-100, stop=100, num=10**3)
#y = np.log(x)
#plt.plot(x,y)

# Make data more normal (improve skewness)

#data_log = pd.DataFrame([])
#data_log['log_sqftliv'] = np.log(df_pred['sqft_living'])
#data_log['log_bathrooms'] = np.log(df_pred['bathrooms'])
#data_log['log_grade'] = np.log(df_pred['grade'])
```

### MinMax Scaling

Reduces skewness and variability of data to improve regression algorithm.


```python
# create empty dataframe for scaled features
data_cont_scaled = pd.DataFrame([])

# minmax scaling on sqft_living:
sqftliv = df_pred['sqft_living']
scaled_sqftliv = (sqftliv - min(sqftliv)) / (max(sqftliv) - min(sqftliv))
data_cont_scaled['sqftliv'] = scaled_sqftliv
```


```python
# This did nothing...let's try standardization
#minmax scaling on grade

#grade = df_pred['grade']
#scaled_grade = (grade - min(grade)) / (max(grade) - min(grade))
#data_cont_scaled['grade'] = scaled_grade

# minmax scaling on bathrooms
#bathrooms = df_pred['bathrooms']
#scaled_bathrooms = (bathrooms - min(bathrooms)) / (max(bathrooms) - min(bathrooms))
#data_cont_scaled['bathrooms'] = scaled_bathrooms

# This did nothing... let's try mean normalization

#log_grade = data_log['log_grade']
#scaled_grade = (log_grade - np.mean(log_grade)) / np.sqrt(np.var(log_grade))
#data_cont_scaled['grade'] = scaled_grade


#log_bathrooms = data_log['log_bathrooms']
#scaled_bathrooms = (log_bathrooms - np.mean(log_bathrooms)) / np.sqrt(np.var(log_bathrooms))
#data_cont_scaled['bathrooms'] = scaled_bathrooms


# This did nothing...probably because they're categorical?
# mean normalization
#log_grade = data_log['log_grade']
#scaled_grade = (log_grade - np.mean(log_grade)) / (max(log_grade) - min(log_grade))
#data_cont_scaled['grade'] = scaled_grade

#log_bathrooms = data_log['log_bathrooms']
#scaled_bathrooms = (log_bathrooms - np.mean(log_bathrooms)) / (max(log_bathrooms) - min(log_bathrooms))
#data_cont_scaled['bathrooms'] = scaled_bathrooms

```


```python
df_pred['zip'] = df_drops['zipcode']
df_pred['bath'] = df_drops['bathrooms']
df_pred['grade'] = df_drops['grade']
df_pred['zip'] = df_pred['zip'].astype('category')
df_pred['bath'] = df_pred['bath'].astype('category')
df_pred['grade'] = df_pred['grade'].astype('category')
```


```python
data_cont_scaled.hist(figsize=(6,6))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1c2b35c978>]],
          dtype=object)




![png](output_218_1.png)



```python
# NOPE this just created a bunch of NaNs for grade and bathroom

# create a new dataframe and add dummy_vars
data_fin = pd.DataFrame([])
data_fin['sqftliv'] = scaled_sqftliv
#grade_dummies = pd.get_dummies(df_pred['grade'], prefix='grade')
#bathroom_dummies = pd.get_dummies(df_pred['bathrooms'], prefix='bath')
#zip_dummies = pd.get_dummies(df_pred['zip'], prefix='zip')
data_fin['grade'] = scaled_grade
data_fin['bath'] = scaled_bathrooms
price = df_pred['price']
data_fin = pd.concat([price, data_fin], axis=1)
data_fin

```


```python
# Run OLS on data using formula y~X where w n predictors X is x, + x1 + ... xn
import statsmodels.api as sm
from statsmodels.formula.api import ols
outcome = 'price'
predictors = data_fin.drop('price', axis=1)
pred_sum = '+'.join(predictors.columns)
formula = outcome + '~' + pred_sum
model = ols(formula=formula, data=data_fin).fit()
model.summary()
```

## Stepwise Selection
Start with empty model, find lowest p-value and perform a forward-backward feature selection based on pvalue


```python

```

## Forward Selection


```python
# Choose a linear model by forward selection
# The function below optimizes adjusted R-squared by adding features that help the most one at a time
# until the score goes down or you run out of features.

import statsmodels.formula.api as smf

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model
```


```python
model = forward_selected(df_pred, 'price')
print(model.model.formula)
print(model.rsquared_adj)
```

    price ~ sqft_living + grade + zip_3 + zip_4 + bedrooms + sqft_living15 + zip_2 + bathrooms + 1
    0.5629928750344877



```python
model = forward_selected(df_pred, 'price')
print(model.model.formula)
print(model.rsquared_adj)
```

    price ~ sqft_living + grade + zip_3 + zip_4 + bedrooms + sqft_living15 + zip_2 + bathrooms + 1
    0.5629928750344877


Explaining/Phrasing R-Squared values
An obtained R-squared value of say 0.85 can be put into a statement as

85% of the variations in dependent variable  𝑦  are explained by the independent variable in our model.


```python
# Alternative method

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import sklearn.metrics as metrics



# Define selector function combining RFE and linear regression
linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select=1)

# Drop already scaled variables for this feature testing
X = df_run.loc[:,~(df_run.columns.str.startswith(('bins','zip')))]
X = X.drop('price',axis=1)

# RUNNING RFE ON THE UNSCALED DATA(DEMONSTRATION)
Y = df_run['price']
# Y = df_run['logz_price']
# X = df_run.drop(['price'],axis=1)


# Run regressions on X,Y 
selector = selector.fit(X,Y)

# Saving unscaled rankings for demo purposes
no_scale = selector.ranking_




# Scale all variables to value between 0-1 to use RFE to determine which features are the most important for determining price?
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Scale the data before running RFE

# dummy vars zip?


# ONLY SCALE NON-CATEGORICAL, ONE-HOT CATEGORICAL
scaler.fit(X,Y)
scaled_data = scaler.transform(X)
scaled_data.shape
```


```python
# Running RFE with scaled data
selector = selector.fit(scaled_data, Y) 
scaled = selector.ranking_
type(scaled)


# Create a dataframe with the ranked values of each feature for both scaled and unscaled data
best_features = pd.DataFrame({'columns':X.columns, 'scaled_rank' : scaled,'unscaled_rank':no_scale})
best_features.set_index('columns',inplace=True)


# Display dataframe (sorted based on unscaled rank)
best_features.sort_values('unscaled_rank')
```


```python
# Concatenate X,Y for OLS
df_run_ols = pd.concat([Y,X],axis=1)

# Import packages
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stats

# Enter equation for selected predictors: (use C to run as categorical) 
# f1 = 'price~C(codezipcode)+C(grade)+sca_sqft_living+sca_sqft_living15' # 0.8 r1 Adjusted
f1 = 'price~C(codezipcode)+grade+sca_sqft_living+sca_sqft_living15' 

# Run model and report sumamry
model = smf.ols(formula=f1, data=df_run_ols).fit()
model.summary()
```

# FINAL MODEL

## OLS Multivariate Regression

DETERMINING IDEAL FEATURES TO USE
Use MinMaxScaler to get on same scale
Use RFE to find the best features


```python
# Define selector function combining RFE and linear regression
linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select=1)

# Drop already scaled variables for this feature testing
X =df_run.loc[:,~(df_run.columns.str.startswith(('bins','zip')))]
X = X.drop('price',axis=1)

# RUNNING RFE ON THE UNSCALED DATA(DEMONSTRATION)
Y = df_run['price']
```


```python
cat_cols = ['bedrooms','bathrooms']

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
new_df = pd.DataFrame()

for col in cat_cols:
    new_df[col] = encoder.fit_transform(df[col])
new_df.head()
```


```python
# Comcatenate X,Y for OLS
df_run_ols = pd.concat([Y,X],axis=1)

# Import packages
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stats

# Enter equation for selected predictors: (use C to run as categorical) 
# f1 = 'price~C(codezipcode)+C(grade)+sca_sqft_living+sca_sqft_living15' # 0.8 r1 Adjusted
f1 = 'price~C(codezipcode)+grade+sca_sqft_living+sca_sqft_living15' 

# Run model and report sumamry
model = smf.ols(formula=f1, data=df_run_ols).fit()
model.summary()
```


```python
# f1 = 'price~C(codezipcode)+C(grade)+sca_sqft_living+sca_sqft_living15'
f1 = 'price ~ C(codezipcode) + C(grade) + sca_sqft_living + sca_sqft_living15' 
```


```python
# save final output
# df_final_data.to_csv(data_filepath+'kc_housing_model_df_final_data.csv')
```

# VALIDATION

## K-Fold Validation with OLS


```python
# k_fold_val_ols(X,y,k=10):
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

y = df_run['price']


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
    f1 = 'price~C(codezipcode)+grade+sca_sqft_living+sca_sqft_living15' 
    model = smf.ols(formula=f1, data=data).fit()
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
```


```python
resid1=model.resid
fig = sm.graphics.qqplot(resid1, dist=stats.norm, line='45', fit=True,marker='.')
```

# Visualizations


```python
colormap = ('skyblue', 'salmon', 'lightgreen')
plt.figure()
pd.plotting.parallel_coordinates(df, 'price', cols=['sqft_living', 'grade_cat', 'zip_cat', 'bathroom_cat'], color=colormap);
pd.plotting.parallel_coordinates
#pd.plotting.scatter_matrix(df);
```


```python
import plotly.graph_objects as go

import pandas as pd

#df = pd.read_csv("https://raw.githubusercontent.com/bcdunbar/datasets/master/parcoords_data.csv")

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['colorVal'],
                   colorscale = 'Electric',
                   showscale = True,
                   cmin = -4000,
                   cmax = -100),
        dimensions = list([
            dict(range = [32000,227900],
                 constraintrange = [100000,150000],
                 label = "Block Height", values = df['blockHeight']),
            dict(range = [0,700000],
                 label = 'Block Width', values = df['blockWidth']),
            dict(tickvals = [0,0.5,1,2,3],
                 ticktext = ['A','AB','B','Y','Z'],
                 label = 'Cyclinder Material', values = df['cycMaterial']),
            dict(range = [-1,4],
                 tickvals = [0,1,2,3],
                 label = 'Block Material', values = df['blockMaterial']),
            dict(range = [134,3154],
                 visible = True,
                 label = 'Total Weight', values = df['totalWeight']),
            dict(range = [9,19984],
                 label = 'Assembly Penalty Wt', values = df['assemblyPW']),
            dict(range = [49000,568000],
                 label = 'Height st Width', values = df['HstW'])])
    )
)
fig.show()
```

##### TABLEAU HOW TO (temp)
Short how-to plot geo data in Tableau:
Load in your .csv dataset from your project.
Let it use data interpreter. It should identify zipcode as a location.
On your worksheet page:
For plotting each price for each house:
Drag the Measures Lat and Long onto the rows and columns boxes (top of sheet)
Drag the Measure price onto the Color Button under Marks.
It should now be listed at the bottom of the Marks panel.
Right-click and select "Dimension"
For plotting median income by zipcode:
Drag zipcode form the Dimensions panel onto the main graph window.
It will automatically load in map of location.
Drag price onto the color button (it will now appear in the Marks window)
Rich click on Price. Select "Measure" > Median
Customize map features by selecting "Map" > Map Layers on the Menu Bar.



```python
# https://www.youtube.com/watch?v=upBvuTqOy9k&feature=youtu.be

import plotly
plotly.offline.init_notebook_mode(connected=True)

import pandas as pd
import numpy as np

import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
```


```python
data = go.Scatter(x=[1,2,3,4], y=[3,6,8,9], name='Blue')
layout = go.Layout(title='Random Scatter')

fig = go.Figure(data=data, layout=layout)
```


```python

```

* "how did you pick the question(s) that you did?"
* "why are these questions important from a business perspective?"
* "how did you decide on the data cleaning options you performed?"
* "why did you choose a given method or library?"
* "why did you select those visualizations and what did you learn from each of them?"
* "why did you pick those features as predictors?"
* "how would you interpret the results?"
* "how confident are you in the predictive quality of the results?"
* "what are some of the things that could cause the results to be wrong?"

## DATE


```python
# group data by dates/months/years to explore comparison in market fluctuations

print(df.date.min())
print(df.date.max())

# Our dataset contains values spanning two years: beginning May 2014 to end of May 2015
```


```python
df.date.dt.year.value_counts(normalize=True)
# 2014    14622
# 2015     6975
# The majority of our data (67%) is from 2014
```


```python
df.date.dt.month.value_counts()
```

### Impact of date (month or year) on price
Question: are housing prices lower or higher in certain months (better to buy)?


```python
# create new columns for year and month
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
```


```python
# There does not appear to be any correlation whatsoever with 
weights = np.linspace(0,1)
np.corrcoef(df['month'], df['price'])
np.corrcoef(df['year'], df['price'])
```


```python
df.month.value_counts(normalize=True)
```


```python
# create variables for each series you want to pass into the xcols list and compare them against one
y_sub = 'price'
date_sub = ['year','month']

sub_scatter(date_sub, y_sub)
```


```python
# check data discrepancies between duplicates (if any):

# 1 - compare price (house value) and date of sale:

# dupes         date        price
# 6021501535   12/23/2014   700,000
#               7/25/2014   430,000
    
# 4139480200   12/9/2014
# 7520000520   3/11/2015
# 3969300030   12/29/2014
# 2231500030   3/24/2015



#for df['id'] in df:
#    if df['id'] == 
#        print(f"{df.id} : {date} : {price}\n")
     #  4139480200
#dupes_id = dupes['id']
#dupes_price = dupes['price']

# 2 - compare other discrepancies and/or missing values 
```


```python
DataFrame.to_period(self[, freq, axis, copy])	Convert DataFrame from DatetimeIndex to PeriodIndex with desired frequency (inferred from index if not passed).
```
