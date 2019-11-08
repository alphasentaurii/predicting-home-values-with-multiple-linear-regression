
---
**Module 1 Final Project Submission**
* Student name: **Ru Keïn**
* Student pace: **Full-Time**
* Project review date/time: **November 4, 2019 at 2:00 PM PST**
* Instructor name: **James Irving, PhD**
---
> Blog post URL: 
http://www.hakkeray.com/projects/datascience/2019/11/06/predicting-home-values-with-multiple-linear-regression.html

> Link to video: 
https://vimeo.com/rukein/datascience-project-1

> Link to tableau public: https://public.tableau.com/views/HousePricesbyZipCodeinKingCountyWA/KingCounty?:display_count=y&:origin=viz_share_link

**GOAL**
* Identify best combination of variable(s) for predicting property values in King County, Washington, USA. 

**OBJECTIVES**
* Address null, missing, duplicate, and unreliable values in the data.
* Determine best approach for analyzing each feature: continuous vs. categorical values
* Identify which combination of features (X) are the best predictors of the target (y). 

**QUESTIONS TO EXPLORE**
* *Scrub*
    * 1. How should we address each feature to prepare it for EDA?
 
* *Explore*
    * 2. Which predictors are closely related (and should be dropped)?
    * 3. Is there an overlap in square-footage measurements?
    * 4. Can we combine two features into one to achieve a higher correlation?
    * 5. Does geography (location) have any relationship with the values of each categorical variable?
 
* *Model*
    * 6. Which features are the best candidates for predicting property values?
    * 7. Does removing outliers improve the distribution?
    * 8. Does scaling/transforming variables improve the regression algorithm?

**TABLE OF CONTENTS**

**[1  OBTAIN]**
Import libraries, packages, data set
* 1.1 Import libraries and packages
* 1.2 Import custom functions
* 1.3 Import dataset and review columns, variables

**[2  SCRUB]**
Clean and organize the data.
* 2.1 Find and replace missing values (nulls)
* 2.2 Identify/Address characteristics of each variable (numeric vs categorical) 
* 2.3 Check for and drop any duplicate observations (rows)
* 2.4 Decide which variables to keep for EDA

**[3  EXPLORE]**
Preliminary analysis and visualizations.
* 3.1 Linearity: Scatterplots, scattermatrix
* 3.2 Multicollinearity: Heatmaps, scatterplots
* 3.3 Distribution: Histograms, Kernel Density Estimates (KDE), LMplots, Boxplots
* 3.4 Regression: regression plots

**[4  MODEL]**
Iterate through linreg models to find best fit predictors
* 4.1 Model 1: OLS Linear Regression
* 4.2 Model 2: One-Hot Encoding
* 4.3 Model 3: Error terms
* 4.4 Model 4: QQ Plots
* 4.5 Model 5: Outliers
* 4.6 Model 6: Robust Scaler

**[5  VALIDATION]**
Validate the results.
* 5.1 K-Fold Cross Validation

**[6  INTERPRET]**
Summarize the findings and make recommendations.
* 6.1 Briefly summarize the results of analysis
* 6.2 Make recommendations
* 6.3 Describe possible future directions

**[7  Additional Research]**
Extracting median home values based on zipcodes

---
# OBTAIN

## Import libraries + packaes


```python
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
```

    fsds_1007219  v0.4.8 loaded.  Read the docs: https://fsds.readthedocs.io/en/latest/ 
    > For convenient loading of standard modules use: `>> from fsds_100719.imports import *`
    



<style  type="text/css" >
</style><table id="T_184fa97e_018d_11ea_a2de_f40f2405a054" ><caption>Loaded Packages and Handles</caption><thead>    <tr>        <th class="col_heading level0 col0" >Package</th>        <th class="col_heading level0 col1" >Handle</th>        <th class="col_heading level0 col2" >Description</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row0_col0" class="data row0 col0" >IPython.display</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row0_col1" class="data row0 col1" >dp</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row0_col2" class="data row0 col2" >Display modules with helpful display and clearing commands.</td>
            </tr>
            <tr>
                                <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row1_col0" class="data row1 col0" >fsds_100719</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row1_col1" class="data row1 col1" >fs</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row1_col2" class="data row1 col2" >Custom data science bootcamp student package</td>
            </tr>
            <tr>
                                <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row2_col0" class="data row2 col0" >matplotlib</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row2_col1" class="data row2 col1" >mpl</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row2_col2" class="data row2 col2" >Matplotlib's base OOP module with formatting artists</td>
            </tr>
            <tr>
                                <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row3_col0" class="data row3 col0" >matplotlib.pyplot</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row3_col1" class="data row3 col1" >plt</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row3_col2" class="data row3 col2" >Matplotlib's matlab-like plotting module</td>
            </tr>
            <tr>
                                <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row4_col0" class="data row4 col0" >numpy</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row4_col1" class="data row4 col1" >np</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row4_col2" class="data row4 col2" >scientific computing with Python</td>
            </tr>
            <tr>
                                <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row5_col0" class="data row5 col0" >pandas</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row5_col1" class="data row5 col1" >pd</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row5_col2" class="data row5 col2" >High performance data structures and tools</td>
            </tr>
            <tr>
                                <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row6_col0" class="data row6 col0" >seaborn</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row6_col1" class="data row6 col1" >sns</td>
                        <td id="T_184fa97e_018d_11ea_a2de_f40f2405a054row6_col2" class="data row6 col2" >High-level data visualization library based on matplotlib</td>
            </tr>
    </tbody></table>



        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


## Import custom functions


```python
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
        ax.set_title(x_col.capitalize() + " vs. " + y.capitalize())
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

## Import Data


```python
# import dataset and review data types, columns, variables
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

Clean and organize the data.

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

## Missing Values
Find and replace missing values using null_hunter() function.


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

## Data Casting

Identify/Address characteristics of each variable (numeric vs categorical)

### ['waterfront']


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
    
    à-la-Mode: 
    0    0.0
    dtype: float64
    
    Found
    2376 Nulls out of 21597(11.0%)
    
    Non-Null Value Counts:
    0.0    19075
    1.0      146
    Name: waterfront, dtype: int64
    
    # Unique Values: 3
    



```python
# Fill nulls with most common value (0.0) # float value
df['waterfront'].fillna(0.0, inplace=True)
#  verify changes
df['waterfront'].isna().sum()
```




    0




```python
# Convert datatype to boolean (values can be either 0 (not waterfront) or 1(is waterfront)
df['is_wf'] = df['waterfront'].astype('bool')
# verify
df['is_wf'].value_counts()
```




    False    21451
    True       146
    Name: is_wf, dtype: int64



### ['yr_renovated']


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
    
    à-la-Mode: 
    0    0.0
    dtype: float64
    
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
df['is_ren'] = is_renovated
df['is_ren'].value_counts()
```




    0.0    20853
    1.0      744
    Name: is_ren, dtype: int64




```python
# Convert to boolean
df['is_ren'] = df['is_ren'].astype('bool')

# verify
df['is_ren'].value_counts()
```




    False    20853
    True       744
    Name: is_ren, dtype: int64



### ['view']


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
    
    à-la-Mode: 
    0    0.0
    dtype: float64
    
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
df['viewed'].value_counts()
```




    False    19485
    True      2112
    Name: viewed, dtype: int64



### ['sqft_basement']


```python
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
    
    à-la-Mode: 
    0    0.0
    dtype: object
    
    
    No Nulls Found!
    
    Non-Null Value Counts:
    0.0       12826
    ?           454
    600.0       217
    500.0       209
    700.0       208
              ...  
    861.0         1
    652.0         1
    1990.0        1
    2190.0        1
    2300.0        1
    Name: sqft_basement, Length: 304, dtype: int64
    
    # Unique Values: 304
    



```python
# Note the majority of the values are zero...we could bin this as a binary 
# where the property either has a basement or does not...

# First replace '?'s with string value '0.0'
df['sqft_basement'].replace(to_replace='?', value='0.0', inplace=True)
```


```python
# and change datatype to float.
df['sqft_basement'] = df['sqft_basement'].astype('float')
```


```python
hot_stats(df, 'sqft_basement', t='price')
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
    
    à-la-Mode: 
    0    0.0
    dtype: float64
    
    
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



```python
df['basement'] = df['sqft_basement'].astype('bool')
```


```python
df['basement'].value_counts()
```




    False    13280
    True      8317
    Name: basement, dtype: int64




```python
corrs = ['is_wf', 'is_ren', 'viewed', 'basement']

# check correlation coefficients
corr_dict(corrs, 'price')
```




    {'is_wf': 0.2643062804831158,
     'is_ren': 0.11754308700194353,
     'viewed': 0.3562431893938023,
     'basement': 0.17826351932053328}



None of these correlation values look strong enough to be predictive of price (min threshold > 0.5, ideally 0.7)

### ['floors']


```python
hot_stats(df, 'floors', t='price')
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
    
    à-la-Mode: 
    0    1.0
    dtype: float64
    
    
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


Bathrooms appears to have a very linear relationship with price. Bedrooms is somewhat linear up to a certain point. Let's look at the hot stats for both.

### ['bedrooms']


```python
hot_stats(df, 'bedrooms', t='price')
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
    
    à-la-Mode: 
    0    3
    dtype: int64
    
    
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


### ['bathrooms']


```python
hot_stats(df, 'bathrooms', t='price')
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
    
    à-la-Mode: 
    0    2.5
    dtype: float64
    
    
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


Bathrooms is the only feature showing correlation over the 0.5 threshold.

The column-like distributions of these features in the scatterplots below indicate the values are categorical.


```python
# sub_scatter() creates scatter plots for multiple features side by side.
y = 'price'
x_cols = ['floors','bedrooms', 'bathrooms']

sub_scatter(df, x_cols, y)
```


![png](output_55_0.png)


Looking at each one more closely using seaborn's catplot:


```python
for col in x_cols:
    sns.catplot(x=col, y='price', height=10, legend=True, data=df)
```


![png](output_57_0.png)



![png](output_57_1.png)



![png](output_57_2.png)



```python
# save correlation coefficients higher than 0.5 in a dict
corr_thresh_dict = {}
corrs = ['bathrooms']
corr_thresh_dict = corr_dict(corrs, 'price')
corr_thresh_dict
```




    {'bathrooms': 0.5259056214532007}



### ['condition']


```python
hot_stats(df, 'condition', t='price')
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
    
    Correlation with PRICE: 0.0361



```python
sns.catplot(x='condition', y='price', data=df, height=8)
```




    <seaborn.axisgrid.FacetGrid at 0x1c2075b358>




![png](output_61_1.png)


Positive linear correlation between price and condition up to a point, but with diminishing returns.

### ['grade']


```python
# View grade stats
hot_stats(df, 'grade', t='price')
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
    
    à-la-Mode: 
    0    7
    dtype: int64
    
    
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
x_cols = ['condition', 'grade']
for col in x_cols:
    sns.catplot(x=col, y='price', height=10, legend=True, data=df)
```


![png](output_65_0.png)



![png](output_65_1.png)


Grade shows a relatively strong positive correlation with price.

### ['yr_built'] 


```python
hot_stats(df, 'yr_built', t='price')
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
    
    à-la-Mode: 
    0    2014
    dtype: int64
    
    
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


![png](output_69_0.png)


Most houses were built during the second half of the century (after 1950). We'll use adaptive binning based on quantiles for yr_built in order to create a more normal distribution.


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
yb_bins = [1900, 1951, 1975, 1997, 2015]

# label the bins for each value 
yb_labels = [1, 2, 3, 4]

# store the yr_range and its corresponding yr_label as new columns in df

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


![png](output_74_0.png)



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




![png](output_76_1.png)



```python
sns.catplot(x='yb_cat', y='price', data=df, height=8)
```




    <seaborn.axisgrid.FacetGrid at 0x1c211c2780>




![png](output_77_1.png)


###  ['zipcode']


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
    
    à-la-Mode: 
    0    98103
    dtype: int64
    
    
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
# Let's look at the data distribution of the 70 unique zipcode values 
fig, ax = plt.subplots()
df['zipcode'].hist(bins=7, color='#67F86F',
edgecolor='black', grid=True)
ax.set_title('Zipcode Histogram', fontsize=16)
ax.set_xlabel('Zipcodes', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
```




    Text(0, 0.5, 'Frequency')




![png](output_80_1.png)



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


![png](output_83_0.png)



```python
sns.catplot(x='zipcode', y='price', data=df, height=10)
```




    <seaborn.axisgrid.FacetGrid at 0x1c206d7c18>




![png](output_84_1.png)


> Some zip codes may have higher priced homes than others, so it's hard to determine from the catplot how this could be used as a predictor. We'll have to explore this variable using geographic plots to see how the distributions trend on a map (i.e. proximity).

### ['lat']  ['long']

> The coordinates for latitude and longitude are not going to be useful to us as far as regression models since we already have zipcodes as a geographic identifier. However we can put them to use for our geographic plotting.

### ['date'] 
convert to datetime


```python
df['date'] = pd.to_datetime(df['date'])
df['date'].dtype
```




    dtype('<M8[ns]')




```python
hot_stats(df, 'date', t='price')
```

    -------->
    HOT!STATS
    <--------
    
    DATE
    Data Type: datetime64[ns]
    
    min   2014-05-02
    max   2015-05-27
    Name: date, dtype: datetime64[ns] 
    
    à-la-Mode: 
    0   2014-06-23
    dtype: datetime64[ns]
    
    
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
    


### ['sqft_above']


```python
hot_stats(df, 'sqft_above', t='price')
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
    
    à-la-Mode: 
    0    1300
    dtype: int64
    
    
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

### ['sqft_living']


```python
hot_stats(df, 'sqft_living', t='price')
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
    
    à-la-Mode: 
    0    1300
    dtype: int64
    
    
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

### ['sqft_lot']


```python
hot_stats(df, 'sqft_lot', t='price')
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
    
    à-la-Mode: 
    0    5000
    dtype: int64
    
    
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


### ['sqft_living15']


```python
hot_stats(df, 'sqft_living15', t='price')
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
    
    à-la-Mode: 
    0    1540
    dtype: int64
    
    
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
hot_stats(df, 'sqft_lot15', t='price')
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
    
    à-la-Mode: 
    0    5000
    dtype: int64
    
    
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


## Duplicates

The primary key we'd use as an index for this data set would be 'id'. Our assumption therefore is that the 'id' for each observation (row) is unique. Let's do a quick scan for duplicate entries to confirm this is true.

### ['id']


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
    
    à-la-Mode: 
    0    795000620
    dtype: int64
    
    
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
      <th>is_wf</th>
      <th>is_ren</th>
      <th>viewed</th>
      <th>basement</th>
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
      <td>True</td>
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
      <td>True</td>
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
      <td>True</td>
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
      <td>True</td>
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
      <td>True</td>
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
      <td>True</td>
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
      <th>is_wf</th>
      <th>is_ren</th>
      <th>viewed</th>
      <th>basement</th>
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
      <td>True</td>
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
      <td>True</td>
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
      <th>is_wf</th>
      <th>is_ren</th>
      <th>viewed</th>
      <th>basement</th>
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
      <td>False</td>
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
      <td>True</td>
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
      <td>False</td>
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
      <td>True</td>
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
      <td>False</td>
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
      <td>False</td>
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
      <td>False</td>
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
      <td>False</td>
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
      <td>False</td>
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
      <td>False</td>
      <td>(1997, 2015]</td>
      <td>4</td>
      <td>(98118, 98200]</td>
      <td>4</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 29 columns</p>
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
#with open('data.pickle', 'rb') as f:
#    df = pickle.load(f)
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

    predrop: (21597, 30)
    postdrop: (21420, 30)


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
    
    à-la-Mode: 
    0    450000.0
    dtype: float64
    
    
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
    


> Keeping the below numbers in mind could be helpful as we start exploring the data:

* range: 78,000 to 7,700,000
* mean value: 540,296
* median value: 450,000


```python
# long tails in price and the median is lower than the mean - distribution is skewed to the right
sns.distplot(df.price)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c218db278>




![png](output_123_1.png)


At this point we can begin exploring the data. Let's first review our current feature list and get rid of any columns we no longer need. As we go through our analysis we'll decide which additional columns to drop, transform, scale, normalize, etc.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21420 entries, 0 to 21596
    Data columns (total 30 columns):
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
    is_wf            21420 non-null bool
    is_ren           21420 non-null bool
    viewed           21420 non-null bool
    basement         21420 non-null bool
    yb_range         21334 non-null category
    yb_cat           21334 non-null category
    zip_range        21420 non-null category
    zip_cat          21420 non-null category
    is_dupe          21420 non-null bool
    dtypes: bool(5), category(4), datetime64[ns](1), float64(9), int64(11)
    memory usage: 3.8 MB



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
    Data columns (total 22 columns):
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
    is_wf            21420 non-null bool
    is_ren           21420 non-null bool
    viewed           21420 non-null bool
    basement         21420 non-null bool
    yb_cat           21334 non-null category
    zip_cat          21420 non-null category
    is_dupe          21420 non-null bool
    dtypes: bool(5), category(2), float64(6), int64(9)
    memory usage: 2.8 MB


# EXPLORE:
    
    EDA CHECKLIST:
    linearity (scatter matrices)
    multicollinearity (heatmaps)
    distributions (histograms, KDEs)
    regression (regplot)

**QUESTION: Which features are the best candidates for predicting property values?**


```python
df.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.142000e+04</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>2.142000e+04</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.00000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
      <td>21420.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5.418614e+05</td>
      <td>3.373950</td>
      <td>2.118429</td>
      <td>2083.132633</td>
      <td>1.512804e+04</td>
      <td>1.495985</td>
      <td>3.410784</td>
      <td>7.662792</td>
      <td>1791.170215</td>
      <td>285.937021</td>
      <td>98077.87437</td>
      <td>47.560197</td>
      <td>-122.213784</td>
      <td>1988.384080</td>
      <td>12775.718161</td>
    </tr>
    <tr>
      <td>std</td>
      <td>3.675569e+05</td>
      <td>0.925405</td>
      <td>0.768720</td>
      <td>918.808412</td>
      <td>4.153080e+04</td>
      <td>0.540081</td>
      <td>0.650035</td>
      <td>1.171971</td>
      <td>828.692965</td>
      <td>440.012962</td>
      <td>53.47748</td>
      <td>0.138589</td>
      <td>0.140791</td>
      <td>685.537057</td>
      <td>27345.621867</td>
    </tr>
    <tr>
      <td>min</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>0.000000</td>
      <td>98001.00000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3.249500e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1200.000000</td>
      <td>0.000000</td>
      <td>98033.00000</td>
      <td>47.471200</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>4.505500e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1920.000000</td>
      <td>7.614000e+03</td>
      <td>1.500000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>0.000000</td>
      <td>98065.00000</td>
      <td>47.572100</td>
      <td>-122.230000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.069050e+04</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2220.000000</td>
      <td>550.000000</td>
      <td>98117.00000</td>
      <td>47.678100</td>
      <td>-122.125000</td>
      <td>2370.000000</td>
      <td>10086.250000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>4820.000000</td>
      <td>98199.00000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Linearity
During the scrub process we made some assumptions and guesses based on correlation coefficients and other values. Let's see what the visualizations tell us by creating some scatter plots.


```python
# Visualize the relationship between square-footages and price
sqft_int = ['sqft_living', 'sqft_above', 'sqft_basement']
sub_scatter(df, sqft_int, 'price', color='#0A2FC4') #20C5C6
```


![png](output_134_0.png)



```python
# visualize relationship between sqft_lot, sqft_lot15, sqft_living15 and price.
y = 'price'
sqft_ext = ['sqft_living15', 'sqft_lot', 'sqft_lot15']

sub_scatter(df, sqft_ext, y, color='#6A76FB')
```


![png](output_135_0.png)


Linear relationships with price show up clearly for sqft_living, sqft_above, sqft_living15.

## Multicollinearity
**QUESTION: which predictors are closely related (and should be dropped)?**

    + multicollinearity: remove variable having most corr with largest # of variables


```python
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
```




    (<Figure size 1440x1440 with 2 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1c1f176240>)




![png](output_139_1.png)


The square footages probably overlap. (In other words sqft_above and sqft_basement could be part of the total sqft_living measurement).


```python
# Visualize multicollinearity between interior square-footages
x_cols = ['sqft_above', 'sqft_basement']
sub_scatter(df, x_cols, 'sqft_living', ncols = 2, color='#FD6F6B')  # lightred
```


![png](output_141_0.png)


    Yikes. These are extremely linear. Just for fun, let's crunch the numbers...

**QUESTION: Is there any overlap in square-footage measurements?**


```python
# create new col containing sum of above and basement
df['sqft_sums'] = df['sqft_above'] + df['sqft_basement']
df['sqft_diffs'] = df['sqft_living'] - df['sqft_above']


sqft_cols = ['sqft_sums', 'sqft_living', 'sqft_above','sqft_diffs', 'sqft_basement']
df_sqft = df[sqft_cols]
df_sqft
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
      <th>sqft_sums</th>
      <th>sqft_living</th>
      <th>sqft_above</th>
      <th>sqft_diffs</th>
      <th>sqft_basement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1180.0</td>
      <td>1180</td>
      <td>1180</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2570.0</td>
      <td>2570</td>
      <td>2170</td>
      <td>400</td>
      <td>400.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>770.0</td>
      <td>770</td>
      <td>770</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1960.0</td>
      <td>1960</td>
      <td>1050</td>
      <td>910</td>
      <td>910.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1680.0</td>
      <td>1680</td>
      <td>1680</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21592</td>
      <td>1530.0</td>
      <td>1530</td>
      <td>1530</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>21593</td>
      <td>2310.0</td>
      <td>2310</td>
      <td>2310</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>21594</td>
      <td>1020.0</td>
      <td>1020</td>
      <td>1020</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>21595</td>
      <td>1600.0</td>
      <td>1600</td>
      <td>1600</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>21596</td>
      <td>1020.0</td>
      <td>1020</td>
      <td>1020</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>21420 rows × 5 columns</p>
</div>



> Looks like the 0.0 values in sqft_basement do in fact mean there is no basement, and for those houses the sqft_above is exactly the same as sqft_living. With this now confirmed, we can be confident that sqft_living is the only measurement worth keeping for analysis.


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

## Distributions


```python
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
```

### Histograms


```python
# hist plot
sub_hists(df)
```


![png](output_152_0.png)



![png](output_152_1.png)



![png](output_152_2.png)



![png](output_152_3.png)



![png](output_152_4.png)



![png](output_152_5.png)



![png](output_152_6.png)



![png](output_152_7.png)



![png](output_152_8.png)



![png](output_152_9.png)



![png](output_152_10.png)



![png](output_152_11.png)



![png](output_152_12.png)



![png](output_152_13.png)



![png](output_152_14.png)


Although sqft_living15 didn't have as much linearity with price as other candidates, it appears to have the most normal distribution out of all of them.

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




    <matplotlib.axes._subplots.AxesSubplot at 0x1c20b826d8>




![png](output_155_1.png)



```python
fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(12,12))
sns.distplot(df['bathrooms'], ax=ax[0])
sns.distplot(df['bedrooms'], ax=ax[1])
sns.distplot(df['floors'], ax=ax[2])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1e9e7080>




![png](output_156_1.png)



```python
fig, ax = plt.subplots(ncols=3, figsize=(12,3))
sns.distplot(df['condition'], ax=ax[0])
sns.distplot(df['grade'], ax=ax[1])
sns.distplot(df['zipcode'], ax=ax[2]) # look at actual zipcode value dist instead of category
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1fa93ef0>




![png](output_157_1.png)


> Diminishing returns for condition (highest scores = min 3 out of 5) and grade (score of 7 out of 13)
Zip Codes are all over the map...literally.

### Geographic

**QUESTION: Does geography (location) have any relationship with the values of each categorical variable?**


```python
cats
```




    ['grade', 'condition', 'bathrooms', 'bedrooms', 'floors']




```python
# lmplot geographic distribution by iterating over list of cat feats
for col in cats:
    sns.lmplot(data=df, x="long", y="lat", fit_reg=False, hue=col, height=10)
plt.show()
```


![png](output_162_0.png)



![png](output_162_1.png)



![png](output_162_2.png)



![png](output_162_3.png)



![png](output_162_4.png)


> The highest graded properties appear to be most dense in the upper left (NW) quadrant. Since we already know that grade has a strong correlation with price, we can posit more confidently that grade, location, and price are strongly related.

> Homes with highest number of floors tend to be located in the NW as well. If we were to look at an actual map, we'd see this is Seattle. 


```python
bins
```




    ['is_wf', 'is_ren', 'viewed', 'basement', 'yb_cat', 'zip_cat', 'is_dupe']




```python
# run binned features through lmplot as a forloop to plot geographic distribution visual
for col in bins:
    sns.lmplot(data=df, x="long", y="lat", fit_reg=False, hue=col, height=10)
plt.show()
```


![png](output_165_0.png)



![png](output_165_1.png)



![png](output_165_2.png)



![png](output_165_3.png)



![png](output_165_4.png)



![png](output_165_5.png)



![png](output_165_6.png)


> Some obvious but also some interesting things to observe in the above lmplots:

* waterfront properties do indeed show up as being on the water. Unfortunately as we saw earlier, this doesn't seem to correlate much with price. This is odd (at least to me) because I'd expect those homes to be more expensive. If this were Los Angeles (where I live) that's a well-known fact...
    
* 'is_dupe' (which represents properties that sold twice in the 2 year period of this dataset) tells us pretty much nothing about anything. They look evenly distributed geographically - we can eliminate this from the model. 
    
* Probably the most surprising observation is 'viewed'. They almost all line up with the coastline, or very close to the water. This may not mean anything but it is worth noting.
    
* Lastly, is_renovated is pretty densely clumped up in the northwest quadrant (again, Seattle). We can assume therefore that a lot of renovations are taking place in the city. Not entirely useful but worth mentioning neverthless.

### Box Plots


```python
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
```


![png](output_168_0.png)



```python
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
```


![png](output_169_0.png)



```python
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
```


![png](output_170_0.png)



```python
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
```


![png](output_171_0.png)



```python
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
```


![png](output_172_0.png)


Certain zipcodes certainly contain higher home prices (mean and median) than others. This is definitely worth exploring further.


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


![png](output_174_0.png)


To a certain degree, there is some linearity in year built, with newer homes (category4) falling within higher price ranges than older homes.

# MODEL:

Let's run the first model using features that have the highest correlation with price. (Min threshold > 0.5)


```python
# BINNED VARS
corr_dict(bins, 'price') # none are over 0.5
```




    {'is_wf': 0.26491453896199596,
     'is_ren': 0.11797451976936697,
     'viewed': 0.3554833537898794,
     'basement': 0.17830254026220943,
     'yb_cat': 0.0848440693678135,
     'zip_cat': -0.02889098458398676,
     'is_dupe': -0.013145274112223848}




```python
# GEOGRAPHIC VARS
corr_dict(geo, 'price') # none are over 0.5
```




    {'lat': 0.3064389377124828,
     'long': 0.019825644946585803,
     'zipcode': -0.05116905613944957}




```python
# CATEGORICAL VARS
corr_dict(cats, 'price') # grade and bathrooms are over 0.5
```




    {'grade': 0.6668349564389758,
     'condition': 0.03421927419820454,
     'bathrooms': 0.5252152971165565,
     'bedrooms': 0.30964001522735674,
     'floors': 0.25497163287127883}




```python
# NUMERIC VARS
corr_dict(nums, 'price') #sqft_living, sqft_above, sqft_living15 are over 0.5
```




    {'sqft_living': 0.7012948591175869,
     'sqft_above': 0.6044238993986454,
     'sqft_basement': 0.3212640164141515,
     'sqft_living15': 0.5837916994556076,
     'sqft_lot': 0.0887889532628066,
     'sqft_lot15': 0.08204522248404933}



NOTES: 
> The coefficients above are based on the raw values. It's possible that some of the variables will produce a higher correlation with price after scaling / transformation. We'll test this out in the second model iteration.

> We also need to take covariance/multicollinearity into consideration. As we saw when we created the multiplot heatmap (as well as scatterplots), the sqft variables have covariance. To make things more difficult, they're also collinear with grade and bathrooms. This could cause our model to overfit.

## Model 1


```python
# highest corr coefs with price - initial model using raw values
pred1 = ['grade', 'sqft_living', 'bathrooms']
```


```python
corr_dict(pred1, 'price')
```




    {'grade': 0.6668349564389758,
     'sqft_living': 0.7012948591175869,
     'bathrooms': 0.5252152971165565}




```python
f1 = '+'.join(pred1)
f1
```




    'grade+sqft_living+bathrooms'




```python
f ='price~'+f1
model = smf.ols(formula=f, data=df).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.536</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.536</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   8251.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 07 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>01:43:38</td>     <th>  Log-Likelihood:    </th> <td>-2.9666e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21420</td>      <th>  AIC:               </th>  <td>5.933e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21416</td>      <th>  BIC:               </th>  <td>5.934e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>    -6e+05</td> <td> 1.34e+04</td> <td>  -44.765</td> <td> 0.000</td> <td>-6.26e+05</td> <td>-5.74e+05</td>
</tr>
<tr>
  <th>grade</th>       <td> 1.044e+05</td> <td> 2307.935</td> <td>   45.215</td> <td> 0.000</td> <td> 9.98e+04</td> <td> 1.09e+05</td>
</tr>
<tr>
  <th>sqft_living</th> <td>  203.2884</td> <td>    3.354</td> <td>   60.617</td> <td> 0.000</td> <td>  196.715</td> <td>  209.862</td>
</tr>
<tr>
  <th>bathrooms</th>   <td>-3.834e+04</td> <td> 3475.069</td> <td>  -11.033</td> <td> 0.000</td> <td>-4.52e+04</td> <td>-3.15e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16794.800</td> <th>  Durbin-Watson:     </th>  <td>   1.986</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1000555.673</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.293</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>35.828</td>   <th>  Cond. No.          </th>  <td>1.81e+04</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.81e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



> P-values look good, but the R-squared (0.536) could be much higher. 

> Skew and Kurtosis are not bad (Skew:	3.293, Kurtosis:	35.828)

> Let's do a second iteration after one-hot-encoding condition and run OLS again with zipcode included and the condition dummies.


```python
# save key regression values in a dict for making quick comparisons between models
reg_mods = dict()
reg_mods['model1'] = {'vars':f1,'r2':0.536, 's': 3.293, 'k': 35.828}

reg_mods
```




    {'model1': {'vars': 'grade+sqft_living+bathrooms',
      'r2': 0.536,
      's': 3.293,
      'k': 35.828}}



## Model 2

### One-Hot Encoding

* Create Dummies for Condition


```python
# apply one-hot encoding to condition
df2 = pd.get_dummies(df, columns=['condition'], drop_first=True)
df2.head()
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
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>is_wf</th>
      <th>is_ren</th>
      <th>viewed</th>
      <th>basement</th>
      <th>yb_cat</th>
      <th>zip_cat</th>
      <th>is_dupe</th>
      <th>condition_2</th>
      <th>condition_3</th>
      <th>condition_4</th>
      <th>condition_5</th>
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
      <td>False</td>
      <td>2</td>
      <td>4</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
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
      <td>True</td>
      <td>1</td>
      <td>4</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
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
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
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
      <td>True</td>
      <td>2</td>
      <td>4</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
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
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.columns
```




    Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
           'grade', 'sqft_above', 'sqft_basement', 'zipcode', 'lat', 'long',
           'sqft_living15', 'sqft_lot15', 'is_wf', 'is_ren', 'viewed', 'basement',
           'yb_cat', 'zip_cat', 'is_dupe', 'condition_2', 'condition_3',
           'condition_4', 'condition_5'],
          dtype='object')




```python
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
      <th>condition_2</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>38</td>
      <td>1</td>
      <td>240000.0</td>
    </tr>
    <tr>
      <td>242</td>
      <td>1</td>
      <td>455000.0</td>
    </tr>
    <tr>
      <td>328</td>
      <td>1</td>
      <td>186375.0</td>
    </tr>
    <tr>
      <td>465</td>
      <td>1</td>
      <td>80000.0</td>
    </tr>
    <tr>
      <td>702</td>
      <td>1</td>
      <td>480000.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>19284</td>
      <td>1</td>
      <td>174900.0</td>
    </tr>
    <tr>
      <td>19348</td>
      <td>1</td>
      <td>290000.0</td>
    </tr>
    <tr>
      <td>19433</td>
      <td>1</td>
      <td>450000.0</td>
    </tr>
    <tr>
      <td>19496</td>
      <td>1</td>
      <td>246500.0</td>
    </tr>
    <tr>
      <td>19605</td>
      <td>1</td>
      <td>235000.0</td>
    </tr>
  </tbody>
</table>
<p>162 rows × 2 columns</p>
</div>




```python
# show diffs between subcats of condition using histograms
for k, v in groups.items():
    plt.figure()
    plt.hist(v['price'], label=k)
    plt.legend()
```


![png](output_195_0.png)



![png](output_195_1.png)



![png](output_195_2.png)



```python
# As we saw before, there are diminishing returns as far as condition goes...
# visualize another way with distplots
for k, v in groups.items():
    plt.figure()
    sns.distplot(v['price'], label=k)
    plt.legend()
```


![png](output_196_0.png)



![png](output_196_1.png)



![png](output_196_2.png)


> NOTE condition is skewed by tails/outliers)


```python
# use list comp to grab condition dummies
c_bins = [col for col in df2.columns if 'condition' in col]
c_bins
```




    ['condition_2', 'condition_3', 'condition_4', 'condition_5']




```python
pred2 = ['C(zipcode)', 'grade', 'sqft_living', 'sqft_living15']
```


```python
pred2.extend(c_bins)
```


```python
pred2
```




    ['C(zipcode)',
     'grade',
     'sqft_living',
     'sqft_living15',
     'condition_2',
     'condition_3',
     'condition_4',
     'condition_5']




```python
f2 = '+'.join(pred2)
f2
```




    'C(zipcode)+grade+sqft_living+sqft_living15+condition_2+condition_3+condition_4+condition_5'




```python
f ='price~'+f2
f
```




    'price~C(zipcode)+grade+sqft_living+sqft_living15+condition_2+condition_3+condition_4+condition_5'




```python
model = smf.ols(formula=f, data=df2).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.750</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.749</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   842.7</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 07 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>01:43:43</td>     <th>  Log-Likelihood:    </th> <td>-2.9003e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21420</td>      <th>  AIC:               </th>  <td>5.802e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21343</td>      <th>  BIC:               </th>  <td>5.808e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    76</td>      <th>                     </th>      <td> </td>     
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
  <th>Intercept</th>           <td>-5.742e+05</td> <td> 3.73e+04</td> <td>  -15.415</td> <td> 0.000</td> <td>-6.47e+05</td> <td>-5.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th> <td> 3.993e+04</td> <td> 1.64e+04</td> <td>    2.438</td> <td> 0.015</td> <td> 7828.078</td> <td>  7.2e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th> <td>-8927.8223</td> <td> 1.48e+04</td> <td>   -0.605</td> <td> 0.545</td> <td>-3.78e+04</td> <td>    2e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th> <td> 7.669e+05</td> <td> 1.44e+04</td> <td>   53.327</td> <td> 0.000</td> <td> 7.39e+05</td> <td> 7.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th> <td> 2.756e+05</td> <td> 1.73e+04</td> <td>   15.892</td> <td> 0.000</td> <td> 2.42e+05</td> <td>  3.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th> <td> 2.569e+05</td> <td>  1.3e+04</td> <td>   19.702</td> <td> 0.000</td> <td> 2.31e+05</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th> <td> 2.235e+05</td> <td> 1.84e+04</td> <td>   12.113</td> <td> 0.000</td> <td> 1.87e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th> <td> 2.814e+05</td> <td> 1.47e+04</td> <td>   19.175</td> <td> 0.000</td> <td> 2.53e+05</td> <td>  3.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th> <td> 8.078e+04</td> <td> 2.09e+04</td> <td>    3.862</td> <td> 0.000</td> <td> 3.98e+04</td> <td> 1.22e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th> <td> 1.041e+05</td> <td> 1.64e+04</td> <td>    6.331</td> <td> 0.000</td> <td> 7.19e+04</td> <td> 1.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th> <td> 1.316e+05</td> <td> 1.92e+04</td> <td>    6.839</td> <td> 0.000</td> <td> 9.39e+04</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th> <td> 8.045e+04</td> <td> 1.65e+04</td> <td>    4.862</td> <td> 0.000</td> <td>  4.8e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th> <td>  4.08e+04</td> <td> 1.55e+04</td> <td>    2.629</td> <td> 0.009</td> <td> 1.04e+04</td> <td> 7.12e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th> <td>-3.604e+04</td> <td> 1.28e+04</td> <td>   -2.815</td> <td> 0.005</td> <td>-6.11e+04</td> <td>-1.09e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th> <td> 1.935e+05</td> <td> 2.29e+04</td> <td>    8.450</td> <td> 0.000</td> <td> 1.49e+05</td> <td> 2.38e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th> <td> 1.462e+05</td> <td> 1.34e+04</td> <td>   10.935</td> <td> 0.000</td> <td>  1.2e+05</td> <td> 1.72e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th> <td> 1.129e+05</td> <td> 1.47e+04</td> <td>    7.697</td> <td> 0.000</td> <td> 8.42e+04</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th> <td> 1.872e+05</td> <td> 1.42e+04</td> <td>   13.143</td> <td> 0.000</td> <td> 1.59e+05</td> <td> 2.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th> <td>-5017.7418</td> <td> 1.51e+04</td> <td>   -0.332</td> <td> 0.740</td> <td>-3.47e+04</td> <td> 2.46e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th> <td>  821.4336</td> <td> 1.48e+04</td> <td>    0.055</td> <td> 0.956</td> <td>-2.82e+04</td> <td> 2.99e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th> <td> 5944.5034</td> <td> 1.93e+04</td> <td>    0.309</td> <td> 0.758</td> <td>-3.18e+04</td> <td> 4.37e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th> <td> 3.675e+05</td> <td> 1.32e+04</td> <td>   27.814</td> <td> 0.000</td> <td> 3.42e+05</td> <td> 3.93e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th> <td> 2.099e+05</td> <td> 1.25e+04</td> <td>   16.742</td> <td> 0.000</td> <td> 1.85e+05</td> <td> 2.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th> <td> 1.699e+04</td> <td> 1.24e+04</td> <td>    1.374</td> <td> 0.170</td> <td>-7254.764</td> <td> 4.12e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th> <td> 1.353e+06</td> <td> 2.82e+04</td> <td>   47.908</td> <td> 0.000</td> <td>  1.3e+06</td> <td> 1.41e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th> <td>  5.33e+05</td> <td> 1.49e+04</td> <td>   35.700</td> <td> 0.000</td> <td> 5.04e+05</td> <td> 5.62e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th> <td>-3933.0593</td> <td> 1.25e+04</td> <td>   -0.314</td> <td> 0.754</td> <td>-2.85e+04</td> <td> 2.06e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th> <td> 1.115e+05</td> <td> 1.58e+04</td> <td>    7.046</td> <td> 0.000</td> <td> 8.05e+04</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th> <td> 2.063e+05</td> <td> 1.25e+04</td> <td>   16.531</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.31e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th> <td> 1.896e+05</td> <td> 1.35e+04</td> <td>   14.064</td> <td> 0.000</td> <td> 1.63e+05</td> <td> 2.16e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th> <td> 5.794e+04</td> <td>  1.5e+04</td> <td>    3.862</td> <td> 0.000</td> <td> 2.85e+04</td> <td> 8.73e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th> <td> 9.847e+04</td> <td> 1.34e+04</td> <td>    7.355</td> <td> 0.000</td> <td> 7.22e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th> <td>  1.76e+04</td> <td>  1.3e+04</td> <td>    1.350</td> <td> 0.177</td> <td>-7958.768</td> <td> 4.32e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th> <td> 6.182e+04</td> <td>  1.3e+04</td> <td>    4.756</td> <td> 0.000</td> <td> 3.63e+04</td> <td> 8.73e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th> <td> 7.349e+04</td> <td> 1.44e+04</td> <td>    5.087</td> <td> 0.000</td> <td> 4.52e+04</td> <td> 1.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th> <td>  1.98e+05</td> <td> 1.97e+04</td> <td>   10.059</td> <td> 0.000</td> <td> 1.59e+05</td> <td> 2.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th> <td> 1.378e+05</td> <td> 1.49e+04</td> <td>    9.273</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th> <td> 1.622e+05</td> <td> 1.33e+04</td> <td>   12.212</td> <td> 0.000</td> <td> 1.36e+05</td> <td> 1.88e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th> <td> 1.611e+05</td> <td>  1.4e+04</td> <td>   11.488</td> <td> 0.000</td> <td> 1.34e+05</td> <td> 1.89e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th> <td> 9.933e+04</td> <td> 1.65e+04</td> <td>    6.014</td> <td> 0.000</td> <td>  6.7e+04</td> <td> 1.32e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th> <td>-4.483e+04</td> <td> 1.39e+04</td> <td>   -3.237</td> <td> 0.001</td> <td> -7.2e+04</td> <td>-1.77e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th> <td> 5.053e+05</td> <td> 2.06e+04</td> <td>   24.574</td> <td> 0.000</td> <td> 4.65e+05</td> <td> 5.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th> <td> 3.487e+05</td> <td> 1.24e+04</td> <td>   28.231</td> <td> 0.000</td> <td> 3.25e+05</td> <td> 3.73e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th> <td> 4.844e+05</td> <td> 1.56e+04</td> <td>   31.024</td> <td> 0.000</td> <td> 4.54e+05</td> <td> 5.15e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th> <td> 1.621e+05</td> <td> 1.41e+04</td> <td>   11.502</td> <td> 0.000</td> <td> 1.34e+05</td> <td>  1.9e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th> <td> 3.571e+05</td> <td>  1.5e+04</td> <td>   23.830</td> <td> 0.000</td> <td> 3.28e+05</td> <td> 3.86e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th> <td> 1.345e+05</td> <td> 1.67e+04</td> <td>    8.063</td> <td> 0.000</td> <td> 1.02e+05</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th> <td> 5.173e+05</td> <td> 2.02e+04</td> <td>   25.637</td> <td> 0.000</td> <td> 4.78e+05</td> <td> 5.57e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th> <td> 6.111e+05</td> <td>  1.5e+04</td> <td>   40.856</td> <td> 0.000</td> <td> 5.82e+05</td> <td>  6.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th> <td>  3.47e+05</td> <td> 1.24e+04</td> <td>   27.968</td> <td> 0.000</td> <td> 3.23e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th> <td> 3.279e+05</td> <td> 1.41e+04</td> <td>   23.256</td> <td> 0.000</td> <td>    3e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th> <td> 3.398e+05</td> <td> 1.25e+04</td> <td>   27.080</td> <td> 0.000</td> <td> 3.15e+05</td> <td> 3.64e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th> <td> 2.023e+05</td> <td> 1.28e+04</td> <td>   15.841</td> <td> 0.000</td> <td> 1.77e+05</td> <td> 2.27e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th> <td> 5.023e+05</td> <td> 1.67e+04</td> <td>   29.995</td> <td> 0.000</td> <td> 4.69e+05</td> <td> 5.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th> <td> 3.517e+05</td> <td> 1.46e+04</td> <td>   24.071</td> <td> 0.000</td> <td> 3.23e+05</td> <td>  3.8e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th> <td> 2.308e+05</td> <td> 1.34e+04</td> <td>   17.240</td> <td> 0.000</td> <td> 2.05e+05</td> <td> 2.57e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th> <td> 2.367e+05</td> <td> 1.39e+04</td> <td>   17.075</td> <td> 0.000</td> <td> 2.09e+05</td> <td> 2.64e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th> <td> 1.746e+05</td> <td> 1.29e+04</td> <td>   13.581</td> <td> 0.000</td> <td> 1.49e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th> <td> 2.991e+05</td> <td>  1.5e+04</td> <td>   19.980</td> <td> 0.000</td> <td>  2.7e+05</td> <td> 3.28e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th> <td> 2.999e+05</td> <td>  1.4e+04</td> <td>   21.490</td> <td> 0.000</td> <td> 2.73e+05</td> <td> 3.27e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th> <td>  1.72e+05</td> <td> 1.47e+04</td> <td>   11.709</td> <td> 0.000</td> <td> 1.43e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th> <td> 9.032e+04</td> <td> 2.65e+04</td> <td>    3.412</td> <td> 0.001</td> <td> 3.84e+04</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th> <td> 1.718e+05</td> <td> 1.31e+04</td> <td>   13.118</td> <td> 0.000</td> <td> 1.46e+05</td> <td> 1.97e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th> <td> 1.364e+05</td> <td> 1.52e+04</td> <td>    8.979</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 1.66e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th> <td> 1.014e+05</td> <td>  1.5e+04</td> <td>    6.774</td> <td> 0.000</td> <td> 7.21e+04</td> <td> 1.31e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th> <td> 2.582e+05</td> <td> 1.51e+04</td> <td>   17.057</td> <td> 0.000</td> <td> 2.29e+05</td> <td> 2.88e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th> <td> 9.797e+04</td> <td> 1.51e+04</td> <td>    6.508</td> <td> 0.000</td> <td> 6.85e+04</td> <td> 1.27e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th> <td> 5.018e+04</td> <td> 1.86e+04</td> <td>    2.697</td> <td> 0.007</td> <td> 1.37e+04</td> <td> 8.66e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th> <td> 6.358e+04</td> <td> 1.48e+04</td> <td>    4.306</td> <td> 0.000</td> <td> 3.46e+04</td> <td> 9.25e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th> <td>  4.08e+05</td> <td> 1.43e+04</td> <td>   28.628</td> <td> 0.000</td> <td>  3.8e+05</td> <td> 4.36e+05</td>
</tr>
<tr>
  <th>grade</th>               <td> 6.215e+04</td> <td> 1877.292</td> <td>   33.108</td> <td> 0.000</td> <td> 5.85e+04</td> <td> 6.58e+04</td>
</tr>
<tr>
  <th>sqft_living</th>         <td>  179.9504</td> <td>    2.435</td> <td>   73.912</td> <td> 0.000</td> <td>  175.178</td> <td>  184.722</td>
</tr>
<tr>
  <th>sqft_living15</th>       <td>   40.8132</td> <td>    3.222</td> <td>   12.667</td> <td> 0.000</td> <td>   34.498</td> <td>   47.129</td>
</tr>
<tr>
  <th>condition_2</th>         <td> 1.187e+04</td> <td> 3.78e+04</td> <td>    0.314</td> <td> 0.753</td> <td>-6.22e+04</td> <td> 8.59e+04</td>
</tr>
<tr>
  <th>condition_3</th>         <td>-2.837e+04</td> <td>  3.5e+04</td> <td>   -0.811</td> <td> 0.418</td> <td> -9.7e+04</td> <td> 4.02e+04</td>
</tr>
<tr>
  <th>condition_4</th>         <td> 7145.7855</td> <td>  3.5e+04</td> <td>    0.204</td> <td> 0.838</td> <td>-6.15e+04</td> <td> 7.58e+04</td>
</tr>
<tr>
  <th>condition_5</th>         <td> 5.343e+04</td> <td> 3.52e+04</td> <td>    1.517</td> <td> 0.129</td> <td>-1.56e+04</td> <td> 1.22e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>23232.521</td> <th>  Durbin-Watson:     </th>  <td>   1.982</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>4782309.186</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 5.170</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>75.467</td>   <th>  Cond. No.          </th>  <td>2.04e+05</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.04e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# store model2 values to dict
reg_mods['model2'] = {'vars':f2, 'r2':0.750, 's': 5.170, 'k': 75.467}

reg_mods
```




    {'model1': {'vars': 'grade+sqft_living+bathrooms',
      'r2': 0.536,
      's': 3.293,
      'k': 35.828},
     'model2': {'vars': 'C(zipcode)+grade+sqft_living+sqft_living15+condition_2+condition_3+condition_4+condition_5',
      'r2': 0.75,
      's': 5.17,
      'k': 75.467}}



> Much higher R-squared but there are some fatal issues with this model:
1. There are a handful of zipcodes with very high P-values. 
2. Condition dummies have very high p-values.
3. Skew: 5.170 (increased)
4. Kurtosis: 75.467	(almost doubled from model1)

> Let's drop condition and try running it again.

## Model 3


```python
# create list of selected predictors
pred3 = ['C(zipcode)','grade', 'sqft_living']

# convert to string with + added
f3 = '+'.join(pred3)

# append target
f ='price~'+f3 

# Run model and show sumamry
model = smf.ols(formula=f, data=df).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.744</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.743</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   874.7</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 07 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>01:43:44</td>     <th>  Log-Likelihood:    </th> <td>-2.9028e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21420</td>      <th>  AIC:               </th>  <td>5.807e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21348</td>      <th>  BIC:               </th>  <td>5.813e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    71</td>      <th>                     </th>      <td> </td>     
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
  <th>Intercept</th>           <td>-5.465e+05</td> <td> 1.43e+04</td> <td>  -38.190</td> <td> 0.000</td> <td>-5.75e+05</td> <td>-5.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th> <td> 4.496e+04</td> <td> 1.65e+04</td> <td>    2.719</td> <td> 0.007</td> <td> 1.26e+04</td> <td> 7.74e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th> <td>-6822.5330</td> <td> 1.49e+04</td> <td>   -0.457</td> <td> 0.648</td> <td>-3.61e+04</td> <td> 2.24e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th> <td> 7.897e+05</td> <td> 1.45e+04</td> <td>   54.481</td> <td> 0.000</td> <td> 7.61e+05</td> <td> 8.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th> <td> 3.065e+05</td> <td> 1.75e+04</td> <td>   17.538</td> <td> 0.000</td> <td> 2.72e+05</td> <td> 3.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th> <td> 2.911e+05</td> <td> 1.31e+04</td> <td>   22.254</td> <td> 0.000</td> <td> 2.65e+05</td> <td> 3.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th> <td> 2.408e+05</td> <td> 1.86e+04</td> <td>   12.926</td> <td> 0.000</td> <td> 2.04e+05</td> <td> 2.77e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th> <td> 2.968e+05</td> <td> 1.48e+04</td> <td>   20.029</td> <td> 0.000</td> <td> 2.68e+05</td> <td> 3.26e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th> <td> 9.059e+04</td> <td> 2.11e+04</td> <td>    4.284</td> <td> 0.000</td> <td> 4.91e+04</td> <td> 1.32e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th> <td> 1.108e+05</td> <td> 1.66e+04</td> <td>    6.669</td> <td> 0.000</td> <td> 7.82e+04</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th> <td> 1.266e+05</td> <td> 1.95e+04</td> <td>    6.506</td> <td> 0.000</td> <td> 8.85e+04</td> <td> 1.65e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th> <td> 7.734e+04</td> <td> 1.67e+04</td> <td>    4.626</td> <td> 0.000</td> <td> 4.46e+04</td> <td>  1.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th> <td> 5.591e+04</td> <td> 1.57e+04</td> <td>    3.567</td> <td> 0.000</td> <td> 2.52e+04</td> <td> 8.66e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th> <td>-2.971e+04</td> <td> 1.29e+04</td> <td>   -2.295</td> <td> 0.022</td> <td>-5.51e+04</td> <td>-4339.713</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th> <td> 1.954e+05</td> <td> 2.32e+04</td> <td>    8.436</td> <td> 0.000</td> <td>  1.5e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th> <td> 1.597e+05</td> <td> 1.35e+04</td> <td>   11.821</td> <td> 0.000</td> <td> 1.33e+05</td> <td> 1.86e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th> <td> 1.182e+05</td> <td> 1.48e+04</td> <td>    7.967</td> <td> 0.000</td> <td> 8.91e+04</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th> <td> 1.883e+05</td> <td> 1.44e+04</td> <td>   13.074</td> <td> 0.000</td> <td>  1.6e+05</td> <td> 2.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th> <td>-4392.7892</td> <td> 1.53e+04</td> <td>   -0.287</td> <td> 0.774</td> <td>-3.44e+04</td> <td> 2.56e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th> <td> 8757.3262</td> <td>  1.5e+04</td> <td>    0.584</td> <td> 0.559</td> <td>-2.06e+04</td> <td> 3.81e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th> <td> 1.241e+04</td> <td> 1.95e+04</td> <td>    0.638</td> <td> 0.524</td> <td>-2.57e+04</td> <td> 5.06e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th> <td> 3.806e+05</td> <td> 1.34e+04</td> <td>   28.512</td> <td> 0.000</td> <td> 3.54e+05</td> <td> 4.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th> <td> 2.133e+05</td> <td> 1.27e+04</td> <td>   16.821</td> <td> 0.000</td> <td> 1.88e+05</td> <td> 2.38e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th> <td> 1.851e+04</td> <td> 1.25e+04</td> <td>    1.481</td> <td> 0.139</td> <td>-5984.381</td> <td>  4.3e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th> <td> 1.378e+06</td> <td> 2.85e+04</td> <td>   48.288</td> <td> 0.000</td> <td> 1.32e+06</td> <td> 1.43e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th> <td>  5.73e+05</td> <td>  1.5e+04</td> <td>   38.259</td> <td> 0.000</td> <td> 5.44e+05</td> <td> 6.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th> <td> 6777.9856</td> <td> 1.27e+04</td> <td>    0.535</td> <td> 0.593</td> <td> -1.8e+04</td> <td> 3.16e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th> <td> 1.059e+05</td> <td>  1.6e+04</td> <td>    6.619</td> <td> 0.000</td> <td> 7.46e+04</td> <td> 1.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th> <td> 2.178e+05</td> <td> 1.26e+04</td> <td>   17.280</td> <td> 0.000</td> <td> 1.93e+05</td> <td> 2.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th> <td> 1.927e+05</td> <td> 1.36e+04</td> <td>   14.163</td> <td> 0.000</td> <td> 1.66e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th> <td> 5.831e+04</td> <td> 1.52e+04</td> <td>    3.843</td> <td> 0.000</td> <td> 2.86e+04</td> <td> 8.81e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th> <td> 1.142e+05</td> <td> 1.35e+04</td> <td>    8.446</td> <td> 0.000</td> <td> 8.77e+04</td> <td> 1.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th> <td> 2.656e+04</td> <td> 1.32e+04</td> <td>    2.015</td> <td> 0.044</td> <td>  719.117</td> <td> 5.24e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th> <td> 7.649e+04</td> <td> 1.31e+04</td> <td>    5.831</td> <td> 0.000</td> <td> 5.08e+04</td> <td> 1.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th> <td> 7.965e+04</td> <td> 1.45e+04</td> <td>    5.475</td> <td> 0.000</td> <td> 5.11e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th> <td> 2.111e+05</td> <td> 1.99e+04</td> <td>   10.610</td> <td> 0.000</td> <td> 1.72e+05</td> <td>  2.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th> <td> 1.493e+05</td> <td>  1.5e+04</td> <td>    9.946</td> <td> 0.000</td> <td>  1.2e+05</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th> <td> 1.724e+05</td> <td> 1.34e+04</td> <td>   12.862</td> <td> 0.000</td> <td> 1.46e+05</td> <td> 1.99e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th> <td> 1.838e+05</td> <td> 1.41e+04</td> <td>   13.055</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 2.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th> <td>  1.22e+05</td> <td> 1.66e+04</td> <td>    7.329</td> <td> 0.000</td> <td> 8.94e+04</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th> <td>-3.848e+04</td> <td>  1.4e+04</td> <td>   -2.748</td> <td> 0.006</td> <td>-6.59e+04</td> <td> -1.1e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th> <td> 5.073e+05</td> <td> 2.08e+04</td> <td>   24.399</td> <td> 0.000</td> <td> 4.67e+05</td> <td> 5.48e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th> <td> 3.465e+05</td> <td> 1.25e+04</td> <td>   27.813</td> <td> 0.000</td> <td> 3.22e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th> <td> 4.995e+05</td> <td> 1.58e+04</td> <td>   31.674</td> <td> 0.000</td> <td> 4.69e+05</td> <td>  5.3e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th> <td> 1.474e+05</td> <td> 1.42e+04</td> <td>   10.359</td> <td> 0.000</td> <td> 1.19e+05</td> <td> 1.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th> <td> 3.538e+05</td> <td> 1.51e+04</td> <td>   23.382</td> <td> 0.000</td> <td> 3.24e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th> <td> 1.313e+05</td> <td> 1.69e+04</td> <td>    7.786</td> <td> 0.000</td> <td> 9.82e+04</td> <td> 1.64e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th> <td> 5.241e+05</td> <td> 2.04e+04</td> <td>   25.693</td> <td> 0.000</td> <td> 4.84e+05</td> <td> 5.64e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th> <td> 6.272e+05</td> <td> 1.51e+04</td> <td>   41.511</td> <td> 0.000</td> <td> 5.98e+05</td> <td> 6.57e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th> <td> 3.497e+05</td> <td> 1.25e+04</td> <td>   27.899</td> <td> 0.000</td> <td> 3.25e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th> <td> 3.311e+05</td> <td> 1.42e+04</td> <td>   23.264</td> <td> 0.000</td> <td> 3.03e+05</td> <td> 3.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th> <td> 3.395e+05</td> <td> 1.27e+04</td> <td>   26.813</td> <td> 0.000</td> <td> 3.15e+05</td> <td> 3.64e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th> <td> 1.999e+05</td> <td> 1.29e+04</td> <td>   15.486</td> <td> 0.000</td> <td> 1.75e+05</td> <td> 2.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th> <td> 5.032e+05</td> <td> 1.69e+04</td> <td>   29.725</td> <td> 0.000</td> <td>  4.7e+05</td> <td> 5.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th> <td> 3.491e+05</td> <td> 1.48e+04</td> <td>   23.655</td> <td> 0.000</td> <td>  3.2e+05</td> <td> 3.78e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th> <td> 2.273e+05</td> <td> 1.35e+04</td> <td>   16.792</td> <td> 0.000</td> <td> 2.01e+05</td> <td> 2.54e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th> <td> 2.314e+05</td> <td>  1.4e+04</td> <td>   16.541</td> <td> 0.000</td> <td> 2.04e+05</td> <td> 2.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th> <td> 1.753e+05</td> <td>  1.3e+04</td> <td>   13.494</td> <td> 0.000</td> <td>  1.5e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th> <td> 2.986e+05</td> <td> 1.51e+04</td> <td>   19.734</td> <td> 0.000</td> <td> 2.69e+05</td> <td> 3.28e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th> <td> 3.033e+05</td> <td> 1.41e+04</td> <td>   21.508</td> <td> 0.000</td> <td> 2.76e+05</td> <td> 3.31e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th> <td> 1.656e+05</td> <td> 1.49e+04</td> <td>   11.149</td> <td> 0.000</td> <td> 1.36e+05</td> <td> 1.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th> <td> 8.162e+04</td> <td> 2.68e+04</td> <td>    3.050</td> <td> 0.002</td> <td> 2.92e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th> <td>  1.75e+05</td> <td> 1.32e+04</td> <td>   13.217</td> <td> 0.000</td> <td> 1.49e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th> <td> 1.458e+05</td> <td> 1.53e+04</td> <td>    9.498</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 1.76e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th> <td> 9.344e+04</td> <td> 1.51e+04</td> <td>    6.173</td> <td> 0.000</td> <td> 6.38e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th> <td> 2.709e+05</td> <td> 1.53e+04</td> <td>   17.706</td> <td> 0.000</td> <td> 2.41e+05</td> <td> 3.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th> <td> 9.334e+04</td> <td> 1.52e+04</td> <td>    6.133</td> <td> 0.000</td> <td> 6.35e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th> <td> 4.351e+04</td> <td> 1.88e+04</td> <td>    2.313</td> <td> 0.021</td> <td> 6641.555</td> <td> 8.04e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th> <td> 6.591e+04</td> <td> 1.49e+04</td> <td>    4.414</td> <td> 0.000</td> <td> 3.66e+04</td> <td> 9.52e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th> <td> 4.163e+05</td> <td> 1.44e+04</td> <td>   28.904</td> <td> 0.000</td> <td> 3.88e+05</td> <td> 4.45e+05</td>
</tr>
<tr>
  <th>grade</th>               <td> 6.203e+04</td> <td> 1780.697</td> <td>   34.836</td> <td> 0.000</td> <td> 5.85e+04</td> <td> 6.55e+04</td>
</tr>
<tr>
  <th>sqft_living</th>         <td>  197.1375</td> <td>    2.208</td> <td>   89.276</td> <td> 0.000</td> <td>  192.809</td> <td>  201.466</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>22502.250</td> <th>  Durbin-Watson:     </th>  <td>   1.978</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>4023694.335</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 4.927</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>69.417</td>   <th>  Cond. No.          </th>  <td>1.49e+05</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.49e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
reg_mods['model3'] = {'vars':f3, 'r2':0.744, 's': 4.927, 'k':69.417 }
reg_mods
```




    {'model1': {'vars': 'grade+sqft_living+bathrooms',
      'r2': 0.536,
      's': 3.293,
      'k': 35.828},
     'model2': {'vars': 'C(zipcode)+grade+sqft_living+sqft_living15+condition_2+condition_3+condition_4+condition_5',
      'r2': 0.75,
      's': 5.17,
      'k': 75.467},
     'model3': {'vars': 'C(zipcode)+grade+sqft_living',
      'r2': 0.744,
      's': 4.927,
      'k': 69.417}}



> R-squared value slightly decreased 

> P-values look good

> Skew and Kurtosis decreased slightly (compared to Model2 at least)

### Error Terms ('grade')


```python
# Visualize Error Terms for Grade
f = 'price~grade'
model = ols(formula=f, data=df).fit()
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, 'grade', fig=fig)
plt.show()
```


![png](output_212_0.png)


## Model 4


```python
# checking same model with GRADE as cat.
pred4 = ['C(zipcode)', 'C(grade)', 'sqft_living']

# convert to string with + added
f4 = '+'.join(pred4)

# append target
f ='price~'+f4 

# Run model and show sumamry
model = smf.ols(formula=f, data=df).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.783</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.782</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   960.1</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 07 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>01:57:11</td>     <th>  Log-Likelihood:    </th> <td>-2.8854e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21420</td>      <th>  AIC:               </th>  <td>5.772e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21339</td>      <th>  BIC:               </th>  <td>5.779e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    80</td>      <th>                     </th>      <td> </td>     
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
  <th>Intercept</th>           <td> 1.245e+05</td> <td> 1.72e+05</td> <td>    0.722</td> <td> 0.470</td> <td>-2.13e+05</td> <td> 4.62e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th> <td> 1.512e+04</td> <td> 1.53e+04</td> <td>    0.991</td> <td> 0.322</td> <td>-1.48e+04</td> <td>  4.5e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th> <td> 2575.7351</td> <td> 1.38e+04</td> <td>    0.187</td> <td> 0.852</td> <td>-2.44e+04</td> <td> 2.96e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th> <td> 7.673e+05</td> <td> 1.34e+04</td> <td>   57.349</td> <td> 0.000</td> <td> 7.41e+05</td> <td> 7.94e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th> <td>  3.08e+05</td> <td> 1.61e+04</td> <td>   19.095</td> <td> 0.000</td> <td> 2.76e+05</td> <td>  3.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th> <td> 2.688e+05</td> <td> 1.21e+04</td> <td>   22.252</td> <td> 0.000</td> <td> 2.45e+05</td> <td> 2.92e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th> <td> 2.533e+05</td> <td> 1.72e+04</td> <td>   14.742</td> <td> 0.000</td> <td>  2.2e+05</td> <td> 2.87e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th> <td> 3.092e+05</td> <td> 1.37e+04</td> <td>   22.621</td> <td> 0.000</td> <td> 2.82e+05</td> <td> 3.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th> <td> 7.262e+04</td> <td> 1.95e+04</td> <td>    3.721</td> <td> 0.000</td> <td> 3.44e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th> <td> 1.298e+05</td> <td> 1.53e+04</td> <td>    8.473</td> <td> 0.000</td> <td> 9.98e+04</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th> <td> 1.022e+05</td> <td>  1.8e+04</td> <td>    5.685</td> <td> 0.000</td> <td>  6.7e+04</td> <td> 1.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th> <td> 8.799e+04</td> <td> 1.54e+04</td> <td>    5.706</td> <td> 0.000</td> <td> 5.78e+04</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th> <td> 4.724e+04</td> <td> 1.45e+04</td> <td>    3.263</td> <td> 0.001</td> <td> 1.89e+04</td> <td> 7.56e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th> <td>-2.411e+04</td> <td> 1.19e+04</td> <td>   -2.019</td> <td> 0.044</td> <td>-4.75e+04</td> <td> -699.359</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th> <td> 1.756e+05</td> <td> 2.14e+04</td> <td>    8.216</td> <td> 0.000</td> <td> 1.34e+05</td> <td> 2.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th> <td> 1.618e+05</td> <td> 1.25e+04</td> <td>   12.976</td> <td> 0.000</td> <td> 1.37e+05</td> <td> 1.86e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th> <td> 1.348e+05</td> <td> 1.37e+04</td> <td>    9.853</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 1.62e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th> <td> 2.091e+05</td> <td> 1.33e+04</td> <td>   15.713</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th> <td> 2521.7309</td> <td> 1.41e+04</td> <td>    0.179</td> <td> 0.858</td> <td>-2.51e+04</td> <td> 3.02e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th> <td> 1.681e+04</td> <td> 1.38e+04</td> <td>    1.215</td> <td> 0.224</td> <td>-1.03e+04</td> <td> 4.39e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th> <td> 7757.5538</td> <td>  1.8e+04</td> <td>    0.432</td> <td> 0.666</td> <td>-2.74e+04</td> <td> 4.29e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th> <td> 3.736e+05</td> <td> 1.23e+04</td> <td>   30.325</td> <td> 0.000</td> <td> 3.49e+05</td> <td> 3.98e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th> <td> 2.135e+05</td> <td> 1.17e+04</td> <td>   18.255</td> <td> 0.000</td> <td> 1.91e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th> <td>  3.13e+04</td> <td> 1.15e+04</td> <td>    2.715</td> <td> 0.007</td> <td> 8707.828</td> <td> 5.39e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th> <td> 1.254e+06</td> <td> 2.64e+04</td> <td>   47.461</td> <td> 0.000</td> <td>  1.2e+06</td> <td> 1.31e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th> <td>  5.62e+05</td> <td> 1.38e+04</td> <td>   40.625</td> <td> 0.000</td> <td> 5.35e+05</td> <td> 5.89e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th> <td> 1.102e+04</td> <td> 1.17e+04</td> <td>    0.944</td> <td> 0.345</td> <td>-1.19e+04</td> <td> 3.39e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th> <td> 1.055e+05</td> <td> 1.48e+04</td> <td>    7.145</td> <td> 0.000</td> <td> 7.65e+04</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th> <td> 2.363e+05</td> <td> 1.16e+04</td> <td>   20.305</td> <td> 0.000</td> <td> 2.13e+05</td> <td> 2.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th> <td> 2.027e+05</td> <td> 1.26e+04</td> <td>   16.150</td> <td> 0.000</td> <td> 1.78e+05</td> <td> 2.27e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th> <td> 4.796e+04</td> <td>  1.4e+04</td> <td>    3.425</td> <td> 0.001</td> <td> 2.05e+04</td> <td> 7.54e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th> <td> 1.059e+05</td> <td> 1.25e+04</td> <td>    8.485</td> <td> 0.000</td> <td> 8.15e+04</td> <td>  1.3e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th> <td> 3.368e+04</td> <td> 1.22e+04</td> <td>    2.768</td> <td> 0.006</td> <td> 9829.716</td> <td> 5.75e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th> <td> 7.623e+04</td> <td> 1.21e+04</td> <td>    6.296</td> <td> 0.000</td> <td> 5.25e+04</td> <td>    1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th> <td> 9.139e+04</td> <td> 1.34e+04</td> <td>    6.810</td> <td> 0.000</td> <td> 6.51e+04</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th> <td> 2.006e+05</td> <td> 1.84e+04</td> <td>   10.927</td> <td> 0.000</td> <td> 1.65e+05</td> <td> 2.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th> <td> 1.588e+05</td> <td> 1.38e+04</td> <td>   11.471</td> <td> 0.000</td> <td> 1.32e+05</td> <td> 1.86e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th> <td> 1.779e+05</td> <td> 1.24e+04</td> <td>   14.367</td> <td> 0.000</td> <td> 1.54e+05</td> <td> 2.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th> <td> 1.812e+05</td> <td>  1.3e+04</td> <td>   13.899</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th> <td>  1.09e+05</td> <td> 1.54e+04</td> <td>    7.093</td> <td> 0.000</td> <td> 7.89e+04</td> <td> 1.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th> <td> -2.21e+04</td> <td> 1.29e+04</td> <td>   -1.710</td> <td> 0.087</td> <td>-4.74e+04</td> <td> 3232.225</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th> <td> 4.813e+05</td> <td> 1.92e+04</td> <td>   25.067</td> <td> 0.000</td> <td> 4.44e+05</td> <td> 5.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th> <td> 3.492e+05</td> <td> 1.15e+04</td> <td>   30.387</td> <td> 0.000</td> <td> 3.27e+05</td> <td> 3.72e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th> <td> 5.055e+05</td> <td> 1.45e+04</td> <td>   34.753</td> <td> 0.000</td> <td> 4.77e+05</td> <td> 5.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th> <td>  1.19e+05</td> <td> 1.31e+04</td> <td>    9.051</td> <td> 0.000</td> <td> 9.32e+04</td> <td> 1.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th> <td> 3.556e+05</td> <td>  1.4e+04</td> <td>   25.463</td> <td> 0.000</td> <td> 3.28e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th> <td> 1.179e+05</td> <td> 1.56e+04</td> <td>    7.578</td> <td> 0.000</td> <td> 8.74e+04</td> <td> 1.48e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th> <td> 5.284e+05</td> <td> 1.88e+04</td> <td>   28.083</td> <td> 0.000</td> <td> 4.91e+05</td> <td> 5.65e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th> <td> 6.313e+05</td> <td> 1.39e+04</td> <td>   45.283</td> <td> 0.000</td> <td> 6.04e+05</td> <td> 6.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th> <td> 3.519e+05</td> <td> 1.16e+04</td> <td>   30.437</td> <td> 0.000</td> <td> 3.29e+05</td> <td> 3.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th> <td> 3.315e+05</td> <td> 1.31e+04</td> <td>   25.251</td> <td> 0.000</td> <td> 3.06e+05</td> <td> 3.57e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th> <td> 3.321e+05</td> <td> 1.17e+04</td> <td>   28.439</td> <td> 0.000</td> <td> 3.09e+05</td> <td> 3.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th> <td> 1.767e+05</td> <td> 1.19e+04</td> <td>   14.804</td> <td> 0.000</td> <td> 1.53e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th> <td> 5.144e+05</td> <td> 1.56e+04</td> <td>   32.937</td> <td> 0.000</td> <td> 4.84e+05</td> <td> 5.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th> <td> 3.581e+05</td> <td> 1.36e+04</td> <td>   26.294</td> <td> 0.000</td> <td> 3.31e+05</td> <td> 3.85e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th> <td> 2.245e+05</td> <td> 1.25e+04</td> <td>   17.984</td> <td> 0.000</td> <td>    2e+05</td> <td> 2.49e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th> <td> 2.106e+05</td> <td> 1.29e+04</td> <td>   16.293</td> <td> 0.000</td> <td> 1.85e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th> <td>  1.69e+05</td> <td>  1.2e+04</td> <td>   14.107</td> <td> 0.000</td> <td> 1.46e+05</td> <td> 1.93e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th> <td>  2.96e+05</td> <td>  1.4e+04</td> <td>   21.210</td> <td> 0.000</td> <td> 2.69e+05</td> <td> 3.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th> <td> 2.998e+05</td> <td>  1.3e+04</td> <td>   23.044</td> <td> 0.000</td> <td> 2.74e+05</td> <td> 3.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th> <td> 1.356e+05</td> <td> 1.37e+04</td> <td>    9.875</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 1.62e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th> <td> 7.102e+04</td> <td> 2.47e+04</td> <td>    2.877</td> <td> 0.004</td> <td> 2.26e+04</td> <td> 1.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th> <td> 1.667e+05</td> <td> 1.22e+04</td> <td>   13.648</td> <td> 0.000</td> <td> 1.43e+05</td> <td> 1.91e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th> <td> 1.323e+05</td> <td> 1.42e+04</td> <td>    9.337</td> <td> 0.000</td> <td> 1.05e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th> <td>  4.74e+04</td> <td>  1.4e+04</td> <td>    3.381</td> <td> 0.001</td> <td> 1.99e+04</td> <td> 7.49e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th> <td> 2.619e+05</td> <td> 1.41e+04</td> <td>   18.538</td> <td> 0.000</td> <td> 2.34e+05</td> <td>  2.9e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th> <td>  7.11e+04</td> <td> 1.41e+04</td> <td>    5.058</td> <td> 0.000</td> <td> 4.36e+04</td> <td> 9.87e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th> <td> 3.256e+04</td> <td> 1.73e+04</td> <td>    1.877</td> <td> 0.061</td> <td>-1445.858</td> <td> 6.66e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th> <td> 5.515e+04</td> <td> 1.38e+04</td> <td>    4.003</td> <td> 0.000</td> <td> 2.81e+04</td> <td> 8.21e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th> <td> 4.127e+05</td> <td> 1.33e+04</td> <td>   31.052</td> <td> 0.000</td> <td> 3.87e+05</td> <td> 4.39e+05</td>
</tr>
<tr>
  <th>C(grade)[T.4]</th>       <td>-1.297e+05</td> <td> 1.75e+05</td> <td>   -0.740</td> <td> 0.459</td> <td>-4.73e+05</td> <td> 2.14e+05</td>
</tr>
<tr>
  <th>C(grade)[T.5]</th>       <td>-1.642e+05</td> <td> 1.72e+05</td> <td>   -0.952</td> <td> 0.341</td> <td>-5.02e+05</td> <td> 1.74e+05</td>
</tr>
<tr>
  <th>C(grade)[T.6]</th>       <td>-1.846e+05</td> <td> 1.72e+05</td> <td>   -1.073</td> <td> 0.283</td> <td>-5.22e+05</td> <td> 1.53e+05</td>
</tr>
<tr>
  <th>C(grade)[T.7]</th>       <td> -1.93e+05</td> <td> 1.72e+05</td> <td>   -1.122</td> <td> 0.262</td> <td> -5.3e+05</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>C(grade)[T.8]</th>       <td>-1.683e+05</td> <td> 1.72e+05</td> <td>   -0.978</td> <td> 0.328</td> <td>-5.06e+05</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>C(grade)[T.9]</th>       <td>-8.396e+04</td> <td> 1.72e+05</td> <td>   -0.488</td> <td> 0.626</td> <td>-4.21e+05</td> <td> 2.54e+05</td>
</tr>
<tr>
  <th>C(grade)[T.10]</th>      <td> 6.008e+04</td> <td> 1.72e+05</td> <td>    0.349</td> <td> 0.727</td> <td>-2.78e+05</td> <td> 3.98e+05</td>
</tr>
<tr>
  <th>C(grade)[T.11]</th>      <td> 2.922e+05</td> <td> 1.72e+05</td> <td>    1.694</td> <td> 0.090</td> <td>-4.59e+04</td> <td>  6.3e+05</td>
</tr>
<tr>
  <th>C(grade)[T.12]</th>      <td> 7.805e+05</td> <td> 1.73e+05</td> <td>    4.502</td> <td> 0.000</td> <td> 4.41e+05</td> <td> 1.12e+06</td>
</tr>
<tr>
  <th>C(grade)[T.13]</th>      <td> 1.793e+06</td> <td> 1.79e+05</td> <td>   10.005</td> <td> 0.000</td> <td> 1.44e+06</td> <td> 2.14e+06</td>
</tr>
<tr>
  <th>sqft_living</th>         <td>  173.6159</td> <td>    2.074</td> <td>   83.697</td> <td> 0.000</td> <td>  169.550</td> <td>  177.682</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>19794.031</td> <th>  Durbin-Watson:     </th>  <td>   1.994</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>2492038.612</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 4.032</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>55.222</td>   <th>  Cond. No.          </th>  <td>1.11e+06</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.11e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
reg_mods['model4'] = {'vars': f4, 'r2':0.783, 's':4.032,'k':55.222 }
reg_mods
```




    {'model1': {'vars': 'grade+sqft_living+bathrooms',
      'r2': 0.536,
      's': 3.293,
      'k': 35.828},
     'model2': {'vars': 'C(zipcode)+grade+sqft_living+sqft_living15+condition_2+condition_3+condition_4+condition_5',
      'r2': 0.75,
      's': 5.17,
      'k': 75.467},
     'model3': {'vars': 'C(zipcode)+grade+sqft_living',
      'r2': 0.744,
      's': 4.927,
      'k': 69.417},
     'model4': {'vars': 'C(zipcode)+C(grade)+sqft_living',
      'r2': 0.783,
      's': 4.032,
      'k': 55.222}}



> R-squared value increased from 0.75 to 0.78 - better.  

> P-values for Grade as a categorical are horrible except for scores of 11, 12, and 13. This could mean we recommend Grade as a factor still, with the requirement that the home score above 10 in order to have an impact on price. 

> Kurtosis and Skew both decreased to levels lower than models 2 and 3. However, the model would most likely benefit further from scaling/normalization. 

### QQ Plots
Investigate high p-values


```python
import scipy.stats as stats
residuals = model.resid
fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
fig.show()
```


![png](output_218_0.png)


> This is not what we want to see...Let's take a closer look at the outliers and find out if removing them helps at all. If not, we may need to drop Grade from the model.

## Model 5

### Outliers

**QUESTION: Does removing outliers improve the distribution?**


```python
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
```


![png](output_222_0.png)



```python
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
```


![png](output_223_0.png)


> NOTE: If the assumption about zipcode (i.e. location) being a critical factor for home price is correct, we could identify from this a list of zipcodes with the highest prices of homes based on median home values -- the assumption for this being that people will pay more for a house located in a certain area than they would for a house in other parts of the county (even if that house is much bigger, has a higher grade, etc). 


```python
# Detect actual number of outliers for our predictors

out_vars = ['sqft_living', 'zipcode', 'grade', 'price']

df_outs = df[out_vars]
df_outs
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
      <th>sqft_living</th>
      <th>zipcode</th>
      <th>grade</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1180</td>
      <td>98178</td>
      <td>7</td>
      <td>221900.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2570</td>
      <td>98125</td>
      <td>7</td>
      <td>538000.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>770</td>
      <td>98028</td>
      <td>6</td>
      <td>180000.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1960</td>
      <td>98136</td>
      <td>7</td>
      <td>604000.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1680</td>
      <td>98074</td>
      <td>8</td>
      <td>510000.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21592</td>
      <td>1530</td>
      <td>98103</td>
      <td>8</td>
      <td>360000.0</td>
    </tr>
    <tr>
      <td>21593</td>
      <td>2310</td>
      <td>98146</td>
      <td>8</td>
      <td>400000.0</td>
    </tr>
    <tr>
      <td>21594</td>
      <td>1020</td>
      <td>98144</td>
      <td>7</td>
      <td>402101.0</td>
    </tr>
    <tr>
      <td>21595</td>
      <td>1600</td>
      <td>98027</td>
      <td>8</td>
      <td>400000.0</td>
    </tr>
    <tr>
      <td>21596</td>
      <td>1020</td>
      <td>98144</td>
      <td>7</td>
      <td>325000.0</td>
    </tr>
  </tbody>
</table>
<p>21420 rows × 4 columns</p>
</div>




```python
# Get IQR scores
Q1 = df_outs.quantile(0.25)
Q3 = df_outs.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
```

    sqft_living      1120.0
    zipcode            84.0
    grade               1.0
    price          320050.0
    dtype: float64



```python

# True indicates outliers present
outliers = (df_outs < (Q1 - 1.5 * IQR)) |(df_outs > (Q3 + 1.5 * IQR))

for col in outliers:
    print(outliers[col].value_counts(normalize=True))
```

    False    0.973483
    True     0.026517
    Name: sqft_living, dtype: float64
    False    1.0
    Name: zipcode, dtype: float64
    False    0.911811
    True     0.088189
    Name: grade, dtype: float64
    False    0.946218
    True     0.053782
    Name: price, dtype: float64


> 8% of the values in grade and 5% in price are outliers.


```python
# Remove outliers 
df_zero_outs = df_outs[~((df_outs < (Q1 - 1.5 * IQR)) |(df_outs > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_outs.shape, df_zero_outs.shape)
```

    (21420, 4) (19032, 4)



```python
# number of outliers removed
df_outs.shape[0] - df_zero_outs.shape[0] # 2388
```




    2388




```python
# rerun OLS with outliers removed
pred5 = ['C(zipcode)', 'C(grade)', 'sqft_living']

# convert to string with + added
f5 = '+'.join(pred5)

# append target
f ='price~'+f5 

# Run model and show sumamry
model = smf.ols(formula=f, data=df_zero_outs).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.780</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.780</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   922.9</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 07 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>03:20:25</td>     <th>  Log-Likelihood:    </th> <td>-2.4420e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 19032</td>      <th>  AIC:               </th>  <td>4.886e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 18958</td>      <th>  BIC:               </th>  <td>4.891e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    73</td>      <th>                     </th>      <td> </td>     
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
  <th>Intercept</th>           <td> 1.231e+04</td> <td> 5543.838</td> <td>    2.221</td> <td> 0.026</td> <td> 1445.035</td> <td> 2.32e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th> <td> 6351.0940</td> <td> 8273.604</td> <td>    0.768</td> <td> 0.443</td> <td>-9865.908</td> <td> 2.26e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th> <td> 9032.9996</td> <td> 7434.520</td> <td>    1.215</td> <td> 0.224</td> <td>-5539.322</td> <td> 2.36e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th> <td> 5.348e+05</td> <td> 8930.554</td> <td>   59.882</td> <td> 0.000</td> <td> 5.17e+05</td> <td> 5.52e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th> <td> 3.542e+05</td> <td> 9156.645</td> <td>   38.683</td> <td> 0.000</td> <td> 3.36e+05</td> <td> 3.72e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th> <td> 2.922e+05</td> <td> 6944.783</td> <td>   42.071</td> <td> 0.000</td> <td> 2.79e+05</td> <td> 3.06e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th> <td> 2.603e+05</td> <td> 9524.305</td> <td>   27.331</td> <td> 0.000</td> <td> 2.42e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th> <td> 2.605e+05</td> <td> 7415.963</td> <td>   35.126</td> <td> 0.000</td> <td> 2.46e+05</td> <td> 2.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th> <td> 9.544e+04</td> <td>  1.1e+04</td> <td>    8.650</td> <td> 0.000</td> <td> 7.38e+04</td> <td> 1.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th> <td>  1.45e+05</td> <td> 8267.970</td> <td>   17.543</td> <td> 0.000</td> <td> 1.29e+05</td> <td> 1.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th> <td> 1.229e+05</td> <td> 1.01e+04</td> <td>   12.119</td> <td> 0.000</td> <td> 1.03e+05</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th> <td>  1.02e+05</td> <td> 8295.019</td> <td>   12.294</td> <td> 0.000</td> <td> 8.57e+04</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th> <td> 4.557e+04</td> <td> 7792.524</td> <td>    5.847</td> <td> 0.000</td> <td> 3.03e+04</td> <td> 6.08e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th> <td>-1.375e+04</td> <td> 6452.171</td> <td>   -2.130</td> <td> 0.033</td> <td>-2.64e+04</td> <td>-1099.048</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th> <td> 1.729e+05</td> <td> 1.21e+04</td> <td>   14.280</td> <td> 0.000</td> <td> 1.49e+05</td> <td> 1.97e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th> <td> 1.937e+05</td> <td> 6992.109</td> <td>   27.702</td> <td> 0.000</td> <td>  1.8e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th> <td>  1.37e+05</td> <td> 7340.785</td> <td>   18.667</td> <td> 0.000</td> <td> 1.23e+05</td> <td> 1.51e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th> <td> 2.216e+05</td> <td> 7283.822</td> <td>   30.428</td> <td> 0.000</td> <td> 2.07e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th> <td> 4581.1771</td> <td> 7556.468</td> <td>    0.606</td> <td> 0.544</td> <td>-1.02e+04</td> <td> 1.94e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th> <td> 1.607e+04</td> <td> 7423.110</td> <td>    2.165</td> <td> 0.030</td> <td> 1519.260</td> <td> 3.06e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th> <td>-1990.2601</td> <td> 9729.845</td> <td>   -0.205</td> <td> 0.838</td> <td>-2.11e+04</td> <td> 1.71e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th> <td> 3.304e+05</td> <td> 6905.022</td> <td>   47.845</td> <td> 0.000</td> <td> 3.17e+05</td> <td> 3.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th> <td> 1.871e+05</td> <td> 6333.409</td> <td>   29.544</td> <td> 0.000</td> <td> 1.75e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th> <td> 3.926e+04</td> <td> 6208.414</td> <td>    6.324</td> <td> 0.000</td> <td> 2.71e+04</td> <td> 5.14e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th> <td> 6.659e+05</td> <td> 4.08e+04</td> <td>   16.315</td> <td> 0.000</td> <td> 5.86e+05</td> <td> 7.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th> <td> 4.608e+05</td> <td> 9085.589</td> <td>   50.714</td> <td> 0.000</td> <td> 4.43e+05</td> <td> 4.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th> <td> 1.535e+04</td> <td> 6287.770</td> <td>    2.442</td> <td> 0.015</td> <td> 3027.646</td> <td> 2.77e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th> <td> 1.126e+05</td> <td> 8046.330</td> <td>   13.997</td> <td> 0.000</td> <td> 9.69e+04</td> <td> 1.28e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th> <td>  2.54e+05</td> <td> 6353.557</td> <td>   39.982</td> <td> 0.000</td> <td> 2.42e+05</td> <td> 2.66e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th> <td> 2.452e+05</td> <td> 6996.748</td> <td>   35.047</td> <td> 0.000</td> <td> 2.32e+05</td> <td> 2.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th> <td>  4.43e+04</td> <td> 7585.387</td> <td>    5.840</td> <td> 0.000</td> <td> 2.94e+04</td> <td> 5.92e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th> <td> 1.071e+05</td> <td> 6866.303</td> <td>   15.601</td> <td> 0.000</td> <td> 9.37e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th> <td> 4.314e+04</td> <td> 6569.234</td> <td>    6.566</td> <td> 0.000</td> <td> 3.03e+04</td> <td>  5.6e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th> <td> 9.813e+04</td> <td> 6631.989</td> <td>   14.797</td> <td> 0.000</td> <td> 8.51e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th> <td> 1.313e+05</td> <td> 7412.222</td> <td>   17.708</td> <td> 0.000</td> <td> 1.17e+05</td> <td> 1.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th> <td> 1.903e+05</td> <td> 1.01e+04</td> <td>   18.897</td> <td> 0.000</td> <td> 1.71e+05</td> <td>  2.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th> <td> 1.794e+05</td> <td> 7668.242</td> <td>   23.395</td> <td> 0.000</td> <td> 1.64e+05</td> <td> 1.94e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th> <td> 2.257e+05</td> <td> 7074.476</td> <td>   31.908</td> <td> 0.000</td> <td> 2.12e+05</td> <td>  2.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th> <td>  2.53e+05</td> <td> 8068.718</td> <td>   31.358</td> <td> 0.000</td> <td> 2.37e+05</td> <td> 2.69e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th> <td> 1.827e+05</td> <td> 9300.869</td> <td>   19.643</td> <td> 0.000</td> <td> 1.64e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th> <td>-6800.2479</td> <td> 6963.481</td> <td>   -0.977</td> <td> 0.329</td> <td>-2.04e+04</td> <td> 6848.796</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th> <td> 4.028e+05</td> <td> 1.09e+04</td> <td>   36.796</td> <td> 0.000</td> <td> 3.81e+05</td> <td> 4.24e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th> <td> 3.226e+05</td> <td> 6184.861</td> <td>   52.162</td> <td> 0.000</td> <td>  3.1e+05</td> <td> 3.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th> <td> 3.884e+05</td> <td> 8368.956</td> <td>   46.414</td> <td> 0.000</td> <td> 3.72e+05</td> <td> 4.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th> <td> 1.031e+05</td> <td> 7061.032</td> <td>   14.600</td> <td> 0.000</td> <td> 8.93e+04</td> <td> 1.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th> <td> 3.206e+05</td> <td> 7484.857</td> <td>   42.838</td> <td> 0.000</td> <td> 3.06e+05</td> <td> 3.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th> <td> 1.099e+05</td> <td> 8323.530</td> <td>   13.204</td> <td> 0.000</td> <td> 9.36e+04</td> <td> 1.26e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th> <td> 4.207e+05</td> <td>  1.1e+04</td> <td>   38.106</td> <td> 0.000</td> <td> 3.99e+05</td> <td> 4.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th> <td> 4.282e+05</td> <td> 8666.211</td> <td>   49.413</td> <td> 0.000</td> <td> 4.11e+05</td> <td> 4.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th> <td> 3.253e+05</td> <td> 6250.539</td> <td>   52.044</td> <td> 0.000</td> <td> 3.13e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th> <td> 3.072e+05</td> <td> 7126.957</td> <td>   43.109</td> <td> 0.000</td> <td> 2.93e+05</td> <td> 3.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th> <td> 3.169e+05</td> <td> 6310.895</td> <td>   50.214</td> <td> 0.000</td> <td> 3.05e+05</td> <td> 3.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th> <td> 1.642e+05</td> <td> 6424.024</td> <td>   25.566</td> <td> 0.000</td> <td> 1.52e+05</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th> <td> 4.135e+05</td> <td> 8887.517</td> <td>   46.531</td> <td> 0.000</td> <td> 3.96e+05</td> <td> 4.31e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th> <td> 3.089e+05</td> <td> 7411.970</td> <td>   41.677</td> <td> 0.000</td> <td> 2.94e+05</td> <td> 3.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th> <td> 1.961e+05</td> <td> 6716.675</td> <td>   29.190</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th> <td> 1.959e+05</td> <td> 6918.765</td> <td>   28.310</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th> <td> 1.529e+05</td> <td> 6401.448</td> <td>   23.892</td> <td> 0.000</td> <td>  1.4e+05</td> <td> 1.65e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th> <td> 2.678e+05</td> <td> 7554.486</td> <td>   35.446</td> <td> 0.000</td> <td> 2.53e+05</td> <td> 2.83e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th> <td> 2.428e+05</td> <td> 7099.297</td> <td>   34.199</td> <td> 0.000</td> <td> 2.29e+05</td> <td> 2.57e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th> <td> 1.183e+05</td> <td> 7454.125</td> <td>   15.874</td> <td> 0.000</td> <td> 1.04e+05</td> <td> 1.33e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th> <td>  5.19e+04</td> <td> 1.32e+04</td> <td>    3.944</td> <td> 0.000</td> <td> 2.61e+04</td> <td> 7.77e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th> <td> 1.457e+05</td> <td> 6553.932</td> <td>   22.235</td> <td> 0.000</td> <td> 1.33e+05</td> <td> 1.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th> <td> 1.281e+05</td> <td> 7736.756</td> <td>   16.557</td> <td> 0.000</td> <td> 1.13e+05</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th> <td> 3.781e+04</td> <td> 7578.553</td> <td>    4.989</td> <td> 0.000</td> <td>  2.3e+04</td> <td> 5.27e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th> <td> 2.388e+05</td> <td> 7810.730</td> <td>   30.573</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.54e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th> <td> 6.369e+04</td> <td> 7531.210</td> <td>    8.457</td> <td> 0.000</td> <td> 4.89e+04</td> <td> 7.85e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th> <td> 3.219e+04</td> <td> 9382.838</td> <td>    3.430</td> <td> 0.001</td> <td> 1.38e+04</td> <td> 5.06e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th> <td> 4.488e+04</td> <td> 7407.290</td> <td>    6.058</td> <td> 0.000</td> <td> 3.04e+04</td> <td> 5.94e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th> <td>  3.68e+05</td> <td> 7470.601</td> <td>   49.262</td> <td> 0.000</td> <td> 3.53e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>C(grade)[T.7]</th>       <td> 1.631e+04</td> <td> 2411.464</td> <td>    6.762</td> <td> 0.000</td> <td> 1.16e+04</td> <td>  2.1e+04</td>
</tr>
<tr>
  <th>C(grade)[T.8]</th>       <td>  5.37e+04</td> <td> 2785.058</td> <td>   19.283</td> <td> 0.000</td> <td> 4.82e+04</td> <td> 5.92e+04</td>
</tr>
<tr>
  <th>C(grade)[T.9]</th>       <td> 1.282e+05</td> <td> 3572.600</td> <td>   35.873</td> <td> 0.000</td> <td> 1.21e+05</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>sqft_living</th>         <td>  123.6130</td> <td>    1.292</td> <td>   95.690</td> <td> 0.000</td> <td>  121.081</td> <td>  126.145</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2928.305</td> <th>  Durbin-Watson:     </th> <td>   1.988</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>10189.665</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.766</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 6.241</td>  <th>  Cond. No.          </th> <td>1.34e+05</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.34e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
reg_mods['model5'] = {'vars': f5, 'r2':0.780, 's':0.766, 'k':6.241}
reg_mods
```




    {'model1': {'vars': 'grade+sqft_living+bathrooms',
      'r2': 0.536,
      's': 3.293,
      'k': 35.828},
     'model2': {'vars': 'C(zipcode)+grade+sqft_living+sqft_living15+condition_2+condition_3+condition_4+condition_5',
      'r2': 0.75,
      's': 5.17,
      'k': 75.467},
     'model3': {'vars': 'C(zipcode)+grade+sqft_living',
      'r2': 0.744,
      's': 4.927,
      'k': 69.417},
     'model4': {'vars': 'C(zipcode)+C(grade)+sqft_living',
      'r2': 0.783,
      's': 4.032,
      'k': 55.222},
     'model5': {'vars': 'C(zipcode)+C(grade)+sqft_living',
      'r2': 0.78,
      's': 0.766,
      'k': 6.241}}



> Removing outliers drastically improved the skew and kurtosis values while maintaining R-squared at 0.78. However, this was at the cost of losing the majority of the grade score levels, leaving us with only 7,8,9 in the model. 

> We could use this to recommend aiming for a minimum grade score between 7 and 9. 


```python
# check QQ Plot for Model5
residuals = model.resid
fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
fig.show()
```


![png](output_234_0.png)


> Not perfect, but definitely a significant improvement over Model 4.

## Model 6 (FINAL)

### Robust Scaler

Considering sqft_living values are in the 1000's while grade is 1 to 13, the model could most likely be improved further by scaling the square-footages down to a magnitude that aligns more closely with the other variables.


```python
# ADDING OUTLIER REMOVAL FROM preprocessing.RobuseScaler
# good to use when you have outliers bc uses median 
from sklearn.preprocessing import RobustScaler

robscaler = RobustScaler()
robscaler
```




    RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
                 with_scaling=True)




```python
scale_vars = ['sqft_living']
```


```python
for col in scale_vars:
    col_data = df[col].values
    res = robscaler.fit_transform(col_data.reshape(-1,1)) # don't scale target
    df['sca_'+col] = res.flatten()
```


```python
df.describe().round(3)
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
      <th>sca_sqft_living</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
      <td>21420.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>541861.428</td>
      <td>3.374</td>
      <td>2.118</td>
      <td>2083.133</td>
      <td>15128.038</td>
      <td>1.496</td>
      <td>3.411</td>
      <td>7.663</td>
      <td>1791.170</td>
      <td>285.937</td>
      <td>98077.874</td>
      <td>47.560</td>
      <td>-122.214</td>
      <td>1988.384</td>
      <td>12775.718</td>
      <td>0.146</td>
    </tr>
    <tr>
      <td>std</td>
      <td>367556.938</td>
      <td>0.925</td>
      <td>0.769</td>
      <td>918.808</td>
      <td>41530.797</td>
      <td>0.540</td>
      <td>0.650</td>
      <td>1.172</td>
      <td>828.693</td>
      <td>440.013</td>
      <td>53.477</td>
      <td>0.139</td>
      <td>0.141</td>
      <td>685.537</td>
      <td>27345.622</td>
      <td>0.820</td>
    </tr>
    <tr>
      <td>min</td>
      <td>78000.000</td>
      <td>1.000</td>
      <td>0.500</td>
      <td>370.000</td>
      <td>520.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>3.000</td>
      <td>370.000</td>
      <td>0.000</td>
      <td>98001.000</td>
      <td>47.156</td>
      <td>-122.519</td>
      <td>399.000</td>
      <td>651.000</td>
      <td>-1.384</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>324950.000</td>
      <td>3.000</td>
      <td>1.750</td>
      <td>1430.000</td>
      <td>5040.000</td>
      <td>1.000</td>
      <td>3.000</td>
      <td>7.000</td>
      <td>1200.000</td>
      <td>0.000</td>
      <td>98033.000</td>
      <td>47.471</td>
      <td>-122.328</td>
      <td>1490.000</td>
      <td>5100.000</td>
      <td>-0.438</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>450550.000</td>
      <td>3.000</td>
      <td>2.250</td>
      <td>1920.000</td>
      <td>7614.000</td>
      <td>1.500</td>
      <td>3.000</td>
      <td>7.000</td>
      <td>1560.000</td>
      <td>0.000</td>
      <td>98065.000</td>
      <td>47.572</td>
      <td>-122.230</td>
      <td>1840.000</td>
      <td>7620.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>645000.000</td>
      <td>4.000</td>
      <td>2.500</td>
      <td>2550.000</td>
      <td>10690.500</td>
      <td>2.000</td>
      <td>4.000</td>
      <td>8.000</td>
      <td>2220.000</td>
      <td>550.000</td>
      <td>98117.000</td>
      <td>47.678</td>
      <td>-122.125</td>
      <td>2370.000</td>
      <td>10086.250</td>
      <td>0.562</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7700000.000</td>
      <td>33.000</td>
      <td>8.000</td>
      <td>13540.000</td>
      <td>1651359.000</td>
      <td>3.500</td>
      <td>5.000</td>
      <td>13.000</td>
      <td>9410.000</td>
      <td>4820.000</td>
      <td>98199.000</td>
      <td>47.778</td>
      <td>-121.315</td>
      <td>6210.000</td>
      <td>871200.000</td>
      <td>10.375</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot histogram to check normality
df['sca_sqft_living'].hist(figsize=(6,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c248b3ac8>




![png](output_242_1.png)



```python
df_zero_outs['sca_sqft_living'] = df['sca_sqft_living'].copy()
df_zero_outs
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
      <th>sqft_living</th>
      <th>zipcode</th>
      <th>grade</th>
      <th>price</th>
      <th>sca_sqft_living</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1180</td>
      <td>98178</td>
      <td>7</td>
      <td>221900.0</td>
      <td>-0.660714</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2570</td>
      <td>98125</td>
      <td>7</td>
      <td>538000.0</td>
      <td>0.580357</td>
    </tr>
    <tr>
      <td>2</td>
      <td>770</td>
      <td>98028</td>
      <td>6</td>
      <td>180000.0</td>
      <td>-1.026786</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1960</td>
      <td>98136</td>
      <td>7</td>
      <td>604000.0</td>
      <td>0.035714</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1680</td>
      <td>98074</td>
      <td>8</td>
      <td>510000.0</td>
      <td>-0.214286</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21592</td>
      <td>1530</td>
      <td>98103</td>
      <td>8</td>
      <td>360000.0</td>
      <td>-0.348214</td>
    </tr>
    <tr>
      <td>21593</td>
      <td>2310</td>
      <td>98146</td>
      <td>8</td>
      <td>400000.0</td>
      <td>0.348214</td>
    </tr>
    <tr>
      <td>21594</td>
      <td>1020</td>
      <td>98144</td>
      <td>7</td>
      <td>402101.0</td>
      <td>-0.803571</td>
    </tr>
    <tr>
      <td>21595</td>
      <td>1600</td>
      <td>98027</td>
      <td>8</td>
      <td>400000.0</td>
      <td>-0.285714</td>
    </tr>
    <tr>
      <td>21596</td>
      <td>1020</td>
      <td>98144</td>
      <td>7</td>
      <td>325000.0</td>
      <td>-0.803571</td>
    </tr>
  </tbody>
</table>
<p>19032 rows × 5 columns</p>
</div>




```python
# rerun OLS with outliers removed
pred6 = ['C(zipcode)', 'C(grade)', 'sca_sqft_living']

# convert to string with + added
f6 = '+'.join(pred6)

# append target
f ='price~'+f6 

# Run model and show sumamry
model = smf.ols(formula=f, data=df_zero_outs).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.780</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.780</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   922.9</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 07 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>04:13:01</td>     <th>  Log-Likelihood:    </th> <td>-2.4420e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 19032</td>      <th>  AIC:               </th>  <td>4.886e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 18958</td>      <th>  BIC:               </th>  <td>4.891e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    73</td>      <th>                     </th>      <td> </td>     
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
  <th>Intercept</th>           <td> 2.496e+05</td> <td> 5371.330</td> <td>   46.478</td> <td> 0.000</td> <td> 2.39e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th> <td> 6351.0940</td> <td> 8273.604</td> <td>    0.768</td> <td> 0.443</td> <td>-9865.908</td> <td> 2.26e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th> <td> 9032.9996</td> <td> 7434.520</td> <td>    1.215</td> <td> 0.224</td> <td>-5539.322</td> <td> 2.36e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th> <td> 5.348e+05</td> <td> 8930.554</td> <td>   59.882</td> <td> 0.000</td> <td> 5.17e+05</td> <td> 5.52e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th> <td> 3.542e+05</td> <td> 9156.645</td> <td>   38.683</td> <td> 0.000</td> <td> 3.36e+05</td> <td> 3.72e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th> <td> 2.922e+05</td> <td> 6944.783</td> <td>   42.071</td> <td> 0.000</td> <td> 2.79e+05</td> <td> 3.06e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th> <td> 2.603e+05</td> <td> 9524.305</td> <td>   27.331</td> <td> 0.000</td> <td> 2.42e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th> <td> 2.605e+05</td> <td> 7415.963</td> <td>   35.126</td> <td> 0.000</td> <td> 2.46e+05</td> <td> 2.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th> <td> 9.544e+04</td> <td>  1.1e+04</td> <td>    8.650</td> <td> 0.000</td> <td> 7.38e+04</td> <td> 1.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th> <td>  1.45e+05</td> <td> 8267.970</td> <td>   17.543</td> <td> 0.000</td> <td> 1.29e+05</td> <td> 1.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th> <td> 1.229e+05</td> <td> 1.01e+04</td> <td>   12.119</td> <td> 0.000</td> <td> 1.03e+05</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th> <td>  1.02e+05</td> <td> 8295.019</td> <td>   12.294</td> <td> 0.000</td> <td> 8.57e+04</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th> <td> 4.557e+04</td> <td> 7792.524</td> <td>    5.847</td> <td> 0.000</td> <td> 3.03e+04</td> <td> 6.08e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th> <td>-1.375e+04</td> <td> 6452.171</td> <td>   -2.130</td> <td> 0.033</td> <td>-2.64e+04</td> <td>-1099.048</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th> <td> 1.729e+05</td> <td> 1.21e+04</td> <td>   14.280</td> <td> 0.000</td> <td> 1.49e+05</td> <td> 1.97e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th> <td> 1.937e+05</td> <td> 6992.109</td> <td>   27.702</td> <td> 0.000</td> <td>  1.8e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th> <td>  1.37e+05</td> <td> 7340.785</td> <td>   18.667</td> <td> 0.000</td> <td> 1.23e+05</td> <td> 1.51e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th> <td> 2.216e+05</td> <td> 7283.822</td> <td>   30.428</td> <td> 0.000</td> <td> 2.07e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th> <td> 4581.1771</td> <td> 7556.468</td> <td>    0.606</td> <td> 0.544</td> <td>-1.02e+04</td> <td> 1.94e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th> <td> 1.607e+04</td> <td> 7423.110</td> <td>    2.165</td> <td> 0.030</td> <td> 1519.260</td> <td> 3.06e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th> <td>-1990.2601</td> <td> 9729.845</td> <td>   -0.205</td> <td> 0.838</td> <td>-2.11e+04</td> <td> 1.71e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th> <td> 3.304e+05</td> <td> 6905.022</td> <td>   47.845</td> <td> 0.000</td> <td> 3.17e+05</td> <td> 3.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th> <td> 1.871e+05</td> <td> 6333.409</td> <td>   29.544</td> <td> 0.000</td> <td> 1.75e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th> <td> 3.926e+04</td> <td> 6208.414</td> <td>    6.324</td> <td> 0.000</td> <td> 2.71e+04</td> <td> 5.14e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th> <td> 6.659e+05</td> <td> 4.08e+04</td> <td>   16.315</td> <td> 0.000</td> <td> 5.86e+05</td> <td> 7.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th> <td> 4.608e+05</td> <td> 9085.589</td> <td>   50.714</td> <td> 0.000</td> <td> 4.43e+05</td> <td> 4.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th> <td> 1.535e+04</td> <td> 6287.770</td> <td>    2.442</td> <td> 0.015</td> <td> 3027.646</td> <td> 2.77e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th> <td> 1.126e+05</td> <td> 8046.330</td> <td>   13.997</td> <td> 0.000</td> <td> 9.69e+04</td> <td> 1.28e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th> <td>  2.54e+05</td> <td> 6353.557</td> <td>   39.982</td> <td> 0.000</td> <td> 2.42e+05</td> <td> 2.66e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th> <td> 2.452e+05</td> <td> 6996.748</td> <td>   35.047</td> <td> 0.000</td> <td> 2.32e+05</td> <td> 2.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th> <td>  4.43e+04</td> <td> 7585.387</td> <td>    5.840</td> <td> 0.000</td> <td> 2.94e+04</td> <td> 5.92e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th> <td> 1.071e+05</td> <td> 6866.303</td> <td>   15.601</td> <td> 0.000</td> <td> 9.37e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th> <td> 4.314e+04</td> <td> 6569.234</td> <td>    6.566</td> <td> 0.000</td> <td> 3.03e+04</td> <td>  5.6e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th> <td> 9.813e+04</td> <td> 6631.989</td> <td>   14.797</td> <td> 0.000</td> <td> 8.51e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th> <td> 1.313e+05</td> <td> 7412.222</td> <td>   17.708</td> <td> 0.000</td> <td> 1.17e+05</td> <td> 1.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th> <td> 1.903e+05</td> <td> 1.01e+04</td> <td>   18.897</td> <td> 0.000</td> <td> 1.71e+05</td> <td>  2.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th> <td> 1.794e+05</td> <td> 7668.242</td> <td>   23.395</td> <td> 0.000</td> <td> 1.64e+05</td> <td> 1.94e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th> <td> 2.257e+05</td> <td> 7074.476</td> <td>   31.908</td> <td> 0.000</td> <td> 2.12e+05</td> <td>  2.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th> <td>  2.53e+05</td> <td> 8068.718</td> <td>   31.358</td> <td> 0.000</td> <td> 2.37e+05</td> <td> 2.69e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th> <td> 1.827e+05</td> <td> 9300.869</td> <td>   19.643</td> <td> 0.000</td> <td> 1.64e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th> <td>-6800.2479</td> <td> 6963.481</td> <td>   -0.977</td> <td> 0.329</td> <td>-2.04e+04</td> <td> 6848.796</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th> <td> 4.028e+05</td> <td> 1.09e+04</td> <td>   36.796</td> <td> 0.000</td> <td> 3.81e+05</td> <td> 4.24e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th> <td> 3.226e+05</td> <td> 6184.861</td> <td>   52.162</td> <td> 0.000</td> <td>  3.1e+05</td> <td> 3.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th> <td> 3.884e+05</td> <td> 8368.956</td> <td>   46.414</td> <td> 0.000</td> <td> 3.72e+05</td> <td> 4.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th> <td> 1.031e+05</td> <td> 7061.032</td> <td>   14.600</td> <td> 0.000</td> <td> 8.93e+04</td> <td> 1.17e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th> <td> 3.206e+05</td> <td> 7484.857</td> <td>   42.838</td> <td> 0.000</td> <td> 3.06e+05</td> <td> 3.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th> <td> 1.099e+05</td> <td> 8323.530</td> <td>   13.204</td> <td> 0.000</td> <td> 9.36e+04</td> <td> 1.26e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th> <td> 4.207e+05</td> <td>  1.1e+04</td> <td>   38.106</td> <td> 0.000</td> <td> 3.99e+05</td> <td> 4.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th> <td> 4.282e+05</td> <td> 8666.211</td> <td>   49.413</td> <td> 0.000</td> <td> 4.11e+05</td> <td> 4.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th> <td> 3.253e+05</td> <td> 6250.539</td> <td>   52.044</td> <td> 0.000</td> <td> 3.13e+05</td> <td> 3.38e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th> <td> 3.072e+05</td> <td> 7126.957</td> <td>   43.109</td> <td> 0.000</td> <td> 2.93e+05</td> <td> 3.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th> <td> 3.169e+05</td> <td> 6310.895</td> <td>   50.214</td> <td> 0.000</td> <td> 3.05e+05</td> <td> 3.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th> <td> 1.642e+05</td> <td> 6424.024</td> <td>   25.566</td> <td> 0.000</td> <td> 1.52e+05</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th> <td> 4.135e+05</td> <td> 8887.517</td> <td>   46.531</td> <td> 0.000</td> <td> 3.96e+05</td> <td> 4.31e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th> <td> 3.089e+05</td> <td> 7411.970</td> <td>   41.677</td> <td> 0.000</td> <td> 2.94e+05</td> <td> 3.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th> <td> 1.961e+05</td> <td> 6716.675</td> <td>   29.190</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th> <td> 1.959e+05</td> <td> 6918.765</td> <td>   28.310</td> <td> 0.000</td> <td> 1.82e+05</td> <td> 2.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th> <td> 1.529e+05</td> <td> 6401.448</td> <td>   23.892</td> <td> 0.000</td> <td>  1.4e+05</td> <td> 1.65e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th> <td> 2.678e+05</td> <td> 7554.486</td> <td>   35.446</td> <td> 0.000</td> <td> 2.53e+05</td> <td> 2.83e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th> <td> 2.428e+05</td> <td> 7099.297</td> <td>   34.199</td> <td> 0.000</td> <td> 2.29e+05</td> <td> 2.57e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th> <td> 1.183e+05</td> <td> 7454.125</td> <td>   15.874</td> <td> 0.000</td> <td> 1.04e+05</td> <td> 1.33e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th> <td>  5.19e+04</td> <td> 1.32e+04</td> <td>    3.944</td> <td> 0.000</td> <td> 2.61e+04</td> <td> 7.77e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th> <td> 1.457e+05</td> <td> 6553.932</td> <td>   22.235</td> <td> 0.000</td> <td> 1.33e+05</td> <td> 1.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th> <td> 1.281e+05</td> <td> 7736.756</td> <td>   16.557</td> <td> 0.000</td> <td> 1.13e+05</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th> <td> 3.781e+04</td> <td> 7578.553</td> <td>    4.989</td> <td> 0.000</td> <td>  2.3e+04</td> <td> 5.27e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th> <td> 2.388e+05</td> <td> 7810.730</td> <td>   30.573</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.54e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th> <td> 6.369e+04</td> <td> 7531.210</td> <td>    8.457</td> <td> 0.000</td> <td> 4.89e+04</td> <td> 7.85e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th> <td> 3.219e+04</td> <td> 9382.838</td> <td>    3.430</td> <td> 0.001</td> <td> 1.38e+04</td> <td> 5.06e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th> <td> 4.488e+04</td> <td> 7407.290</td> <td>    6.058</td> <td> 0.000</td> <td> 3.04e+04</td> <td> 5.94e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th> <td>  3.68e+05</td> <td> 7470.601</td> <td>   49.262</td> <td> 0.000</td> <td> 3.53e+05</td> <td> 3.83e+05</td>
</tr>
<tr>
  <th>C(grade)[T.7]</th>       <td> 1.631e+04</td> <td> 2411.464</td> <td>    6.762</td> <td> 0.000</td> <td> 1.16e+04</td> <td>  2.1e+04</td>
</tr>
<tr>
  <th>C(grade)[T.8]</th>       <td>  5.37e+04</td> <td> 2785.058</td> <td>   19.283</td> <td> 0.000</td> <td> 4.82e+04</td> <td> 5.92e+04</td>
</tr>
<tr>
  <th>C(grade)[T.9]</th>       <td> 1.282e+05</td> <td> 3572.600</td> <td>   35.873</td> <td> 0.000</td> <td> 1.21e+05</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>sca_sqft_living</th>     <td> 1.384e+05</td> <td> 1446.825</td> <td>   95.690</td> <td> 0.000</td> <td> 1.36e+05</td> <td> 1.41e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2928.305</td> <th>  Durbin-Watson:     </th> <td>   1.988</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>10189.665</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.766</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 6.241</td>  <th>  Cond. No.          </th> <td>    77.1</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
reg_mods['model6'] = {'vars': f6, 'r2': 0.780, 's': 0.766, 'k': 6.241}
reg_mods
```




    {'model1': {'vars': 'grade+sqft_living+bathrooms',
      'r2': 0.536,
      's': 3.293,
      'k': 35.828},
     'model2': {'vars': 'C(zipcode)+grade+sqft_living+sqft_living15+condition_2+condition_3+condition_4+condition_5',
      'r2': 0.75,
      's': 5.17,
      'k': 75.467},
     'model3': {'vars': 'C(zipcode)+grade+sqft_living',
      'r2': 0.744,
      's': 4.927,
      'k': 69.417},
     'model4': {'vars': 'C(zipcode)+C(grade)+sqft_living',
      'r2': 0.783,
      's': 4.032,
      'k': 55.222},
     'model5': {'vars': 'C(zipcode)+C(grade)+sqft_living',
      'r2': 0.78,
      's': 0.766,
      'k': 6.241},
     'model6': {'vars': 'C(zipcode)+C(grade)+sca_sqft_living',
      'r2': 0.78,
      's': 0.766,
      'k': 6.241}}




```python
# save final output
df_fin = df_zero_outs.copy()

with open('data.pickle', 'wb') as f:
    pickle.dump(df_fin, f, pickle.HIGHEST_PROTOCOL)
```


```python

df_fin.to_csv('kc_housing_model_df_final_data.csv')
```

# VALIDATION

## K-Fold Validation with OLS


```python
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
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.784</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.782</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   704.4</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 07 Nov 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>04:53:10</td>     <th>  Log-Likelihood:    </th> <td>-1.8305e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 14274</td>      <th>  AIC:               </th>  <td>3.662e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 14200</td>      <th>  BIC:               </th>  <td>3.668e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    73</td>      <th>                     </th>      <td> </td>     
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
  <th>Intercept</th>           <td> 2.512e+05</td> <td> 6131.069</td> <td>   40.970</td> <td> 0.000</td> <td> 2.39e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th> <td> 6908.5365</td> <td> 9352.350</td> <td>    0.739</td> <td> 0.460</td> <td>-1.14e+04</td> <td> 2.52e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th> <td> 2371.7091</td> <td> 8457.202</td> <td>    0.280</td> <td> 0.779</td> <td>-1.42e+04</td> <td> 1.89e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th> <td> 5.306e+05</td> <td> 1.05e+04</td> <td>   50.506</td> <td> 0.000</td> <td>  5.1e+05</td> <td> 5.51e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th> <td> 3.515e+05</td> <td> 1.02e+04</td> <td>   34.589</td> <td> 0.000</td> <td> 3.32e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th> <td> 2.911e+05</td> <td> 7909.032</td> <td>   36.806</td> <td> 0.000</td> <td> 2.76e+05</td> <td> 3.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th> <td> 2.571e+05</td> <td> 1.08e+04</td> <td>   23.754</td> <td> 0.000</td> <td> 2.36e+05</td> <td> 2.78e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th> <td> 2.576e+05</td> <td> 8564.971</td> <td>   30.079</td> <td> 0.000</td> <td> 2.41e+05</td> <td> 2.74e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th> <td> 9.596e+04</td> <td> 1.26e+04</td> <td>    7.641</td> <td> 0.000</td> <td> 7.13e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th> <td> 1.438e+05</td> <td> 9565.386</td> <td>   15.031</td> <td> 0.000</td> <td> 1.25e+05</td> <td> 1.63e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th> <td> 1.325e+05</td> <td>  1.2e+04</td> <td>   11.061</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th> <td> 9.119e+04</td> <td> 9261.733</td> <td>    9.846</td> <td> 0.000</td> <td>  7.3e+04</td> <td> 1.09e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th> <td> 4.119e+04</td> <td> 8791.165</td> <td>    4.685</td> <td> 0.000</td> <td>  2.4e+04</td> <td> 5.84e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th> <td>-1.406e+04</td> <td> 7456.425</td> <td>   -1.886</td> <td> 0.059</td> <td>-2.87e+04</td> <td>  551.979</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th> <td> 1.711e+05</td> <td> 1.33e+04</td> <td>   12.899</td> <td> 0.000</td> <td> 1.45e+05</td> <td> 1.97e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th> <td>  1.97e+05</td> <td> 8009.097</td> <td>   24.592</td> <td> 0.000</td> <td> 1.81e+05</td> <td> 2.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th> <td> 1.382e+05</td> <td> 8374.672</td> <td>   16.504</td> <td> 0.000</td> <td> 1.22e+05</td> <td> 1.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th> <td> 2.186e+05</td> <td> 8291.511</td> <td>   26.361</td> <td> 0.000</td> <td> 2.02e+05</td> <td> 2.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th> <td> 1996.1523</td> <td> 8588.503</td> <td>    0.232</td> <td> 0.816</td> <td>-1.48e+04</td> <td> 1.88e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th> <td> 9574.9402</td> <td> 8616.165</td> <td>    1.111</td> <td> 0.266</td> <td>-7313.872</td> <td> 2.65e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th> <td> -966.5899</td> <td>  1.1e+04</td> <td>   -0.088</td> <td> 0.930</td> <td>-2.25e+04</td> <td> 2.06e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th> <td> 3.287e+05</td> <td> 7871.339</td> <td>   41.763</td> <td> 0.000</td> <td> 3.13e+05</td> <td> 3.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th> <td> 1.819e+05</td> <td> 7185.771</td> <td>   25.317</td> <td> 0.000</td> <td> 1.68e+05</td> <td> 1.96e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th> <td> 3.596e+04</td> <td> 7111.112</td> <td>    5.057</td> <td> 0.000</td> <td>  2.2e+04</td> <td> 4.99e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th> <td>  6.87e+05</td> <td> 5.23e+04</td> <td>   13.148</td> <td> 0.000</td> <td> 5.85e+05</td> <td> 7.89e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th> <td> 4.574e+05</td> <td> 1.02e+04</td> <td>   44.856</td> <td> 0.000</td> <td> 4.37e+05</td> <td> 4.77e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th> <td> 1.065e+04</td> <td> 7146.490</td> <td>    1.491</td> <td> 0.136</td> <td>-3355.086</td> <td> 2.47e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th> <td> 1.129e+05</td> <td> 9403.221</td> <td>   12.009</td> <td> 0.000</td> <td> 9.45e+04</td> <td> 1.31e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th> <td> 2.567e+05</td> <td> 7243.531</td> <td>   35.443</td> <td> 0.000</td> <td> 2.43e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th> <td> 2.437e+05</td> <td> 8002.260</td> <td>   30.450</td> <td> 0.000</td> <td> 2.28e+05</td> <td> 2.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th> <td> 4.272e+04</td> <td> 8714.214</td> <td>    4.902</td> <td> 0.000</td> <td> 2.56e+04</td> <td> 5.98e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th> <td> 1.028e+05</td> <td> 7828.199</td> <td>   13.127</td> <td> 0.000</td> <td> 8.74e+04</td> <td> 1.18e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th> <td> 4.096e+04</td> <td> 7566.253</td> <td>    5.414</td> <td> 0.000</td> <td> 2.61e+04</td> <td> 5.58e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th> <td> 9.627e+04</td> <td> 7577.985</td> <td>   12.704</td> <td> 0.000</td> <td> 8.14e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th> <td> 1.244e+05</td> <td> 8422.276</td> <td>   14.766</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 1.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th> <td> 1.867e+05</td> <td> 1.21e+04</td> <td>   15.414</td> <td> 0.000</td> <td> 1.63e+05</td> <td>  2.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th> <td> 1.808e+05</td> <td> 8846.254</td> <td>   20.439</td> <td> 0.000</td> <td> 1.63e+05</td> <td> 1.98e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th> <td> 2.192e+05</td> <td> 8117.353</td> <td>   27.010</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 2.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th> <td> 2.527e+05</td> <td> 9290.563</td> <td>   27.198</td> <td> 0.000</td> <td> 2.34e+05</td> <td> 2.71e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th> <td> 1.837e+05</td> <td> 1.07e+04</td> <td>   17.127</td> <td> 0.000</td> <td> 1.63e+05</td> <td> 2.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th> <td>-1.077e+04</td> <td> 8060.050</td> <td>   -1.336</td> <td> 0.182</td> <td>-2.66e+04</td> <td> 5030.681</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th> <td> 3.962e+05</td> <td> 1.28e+04</td> <td>   30.910</td> <td> 0.000</td> <td> 3.71e+05</td> <td> 4.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th> <td> 3.173e+05</td> <td> 7077.837</td> <td>   44.831</td> <td> 0.000</td> <td> 3.03e+05</td> <td> 3.31e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th> <td> 3.813e+05</td> <td> 9468.271</td> <td>   40.275</td> <td> 0.000</td> <td> 3.63e+05</td> <td>    4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th> <td> 9.728e+04</td> <td> 8042.246</td> <td>   12.097</td> <td> 0.000</td> <td> 8.15e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th> <td> 3.191e+05</td> <td> 8610.191</td> <td>   37.062</td> <td> 0.000</td> <td> 3.02e+05</td> <td> 3.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th> <td> 1.026e+05</td> <td> 9401.760</td> <td>   10.911</td> <td> 0.000</td> <td> 8.42e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th> <td> 4.335e+05</td> <td> 1.29e+04</td> <td>   33.633</td> <td> 0.000</td> <td> 4.08e+05</td> <td> 4.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th> <td> 4.134e+05</td> <td> 1.01e+04</td> <td>   40.992</td> <td> 0.000</td> <td> 3.94e+05</td> <td> 4.33e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th> <td> 3.233e+05</td> <td> 7095.925</td> <td>   45.564</td> <td> 0.000</td> <td> 3.09e+05</td> <td> 3.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th> <td> 2.969e+05</td> <td> 8094.042</td> <td>   36.682</td> <td> 0.000</td> <td> 2.81e+05</td> <td> 3.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th> <td> 3.127e+05</td> <td> 7191.060</td> <td>   43.478</td> <td> 0.000</td> <td> 2.99e+05</td> <td> 3.27e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th> <td> 1.619e+05</td> <td> 7367.498</td> <td>   21.977</td> <td> 0.000</td> <td> 1.47e+05</td> <td> 1.76e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th> <td>  4.21e+05</td> <td> 1.01e+04</td> <td>   41.849</td> <td> 0.000</td> <td> 4.01e+05</td> <td> 4.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th> <td> 3.081e+05</td> <td> 8402.446</td> <td>   36.667</td> <td> 0.000</td> <td> 2.92e+05</td> <td> 3.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th> <td> 1.929e+05</td> <td> 7649.985</td> <td>   25.210</td> <td> 0.000</td> <td> 1.78e+05</td> <td> 2.08e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th> <td> 1.919e+05</td> <td> 7887.302</td> <td>   24.332</td> <td> 0.000</td> <td> 1.76e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th> <td> 1.514e+05</td> <td> 7301.418</td> <td>   20.731</td> <td> 0.000</td> <td> 1.37e+05</td> <td> 1.66e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th> <td> 2.621e+05</td> <td> 8493.453</td> <td>   30.856</td> <td> 0.000</td> <td> 2.45e+05</td> <td> 2.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th> <td> 2.483e+05</td> <td> 8232.349</td> <td>   30.156</td> <td> 0.000</td> <td> 2.32e+05</td> <td> 2.64e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th> <td> 1.173e+05</td> <td> 8487.585</td> <td>   13.825</td> <td> 0.000</td> <td> 1.01e+05</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th> <td> 4.797e+04</td> <td> 1.43e+04</td> <td>    3.365</td> <td> 0.001</td> <td>    2e+04</td> <td> 7.59e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th> <td> 1.435e+05</td> <td> 7527.316</td> <td>   19.067</td> <td> 0.000</td> <td> 1.29e+05</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th> <td> 1.247e+05</td> <td> 8813.587</td> <td>   14.145</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 1.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th> <td> 3.502e+04</td> <td> 8773.444</td> <td>    3.992</td> <td> 0.000</td> <td> 1.78e+04</td> <td> 5.22e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th> <td> 2.352e+05</td> <td> 8991.768</td> <td>   26.160</td> <td> 0.000</td> <td> 2.18e+05</td> <td> 2.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th> <td> 5.321e+04</td> <td> 8567.429</td> <td>    6.211</td> <td> 0.000</td> <td> 3.64e+04</td> <td>    7e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th> <td>  2.67e+04</td> <td> 1.04e+04</td> <td>    2.568</td> <td> 0.010</td> <td> 6322.089</td> <td> 4.71e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th> <td> 4.376e+04</td> <td> 8428.700</td> <td>    5.192</td> <td> 0.000</td> <td> 2.72e+04</td> <td> 6.03e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th> <td> 3.603e+05</td> <td> 8584.627</td> <td>   41.969</td> <td> 0.000</td> <td> 3.43e+05</td> <td> 3.77e+05</td>
</tr>
<tr>
  <th>C(grade)[T.7]</th>       <td> 1.759e+04</td> <td> 2758.771</td> <td>    6.376</td> <td> 0.000</td> <td> 1.22e+04</td> <td>  2.3e+04</td>
</tr>
<tr>
  <th>C(grade)[T.8]</th>       <td>  5.49e+04</td> <td> 3190.897</td> <td>   17.205</td> <td> 0.000</td> <td> 4.86e+04</td> <td> 6.12e+04</td>
</tr>
<tr>
  <th>C(grade)[T.9]</th>       <td> 1.308e+05</td> <td> 4090.245</td> <td>   31.987</td> <td> 0.000</td> <td> 1.23e+05</td> <td> 1.39e+05</td>
</tr>
<tr>
  <th>sca_sqft_living</th>     <td> 1.382e+05</td> <td> 1646.447</td> <td>   83.948</td> <td> 0.000</td> <td> 1.35e+05</td> <td> 1.41e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2080.969</td> <th>  Durbin-Watson:     </th> <td>   2.002</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>6504.428</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.757</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.940</td>  <th>  Cond. No.          </th> <td>    82.8</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# INTERPRET

> SUMMARY: We can be confident that 78% of the final model (#6) explains the variation in data. Unfortunately, multicollinearity is a significant issue for linear regression and cannot be completely avoided. 


> RECOMMENDATIONS: According to our final model, the best predictors of house prices are sqft-living, zipcode, and grade. 

> * Homes with larger living areas are valued higher than smaller homes. 
> * Houses in certain zip codes are valued at higher prices than other zip codes.
> * Homes that score above at least 8 on Grade will sell higher than those below.

> FUTURE WORK:
* Identify ranking for zip codes by highest home prices (median home value)
 

# Additional Research

> **Do house prices change over time or depending on season?**
This data set was limited to a one-year time-frame. I'd be interested in widening the sample size to investigate how property values fluctuate over time as well as how they are affected by market fluctuations.

> **Can we validate the accuracy of our prediction model by looking specifically at houses that resold for a higher price in a given timeframe?** In other words, try to identify which specific variables changed (e.g. increased grade score after doing renovations) and therefore were determining factors in the increased price of the home when it was resold.


```python
# pypi package for retrieving information based on us zipcodes
from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode=True) # set simple_zipcode=False to use rich info database

# create array of zipcodes
zipcodes = df['zipcode'].unique()
zipcodes
```




    array([98178, 98125, 98028, 98136, 98074, 98053, 98003, 98198, 98146,
           98038, 98007, 98115, 98107, 98126, 98019, 98103, 98002, 98133,
           98040, 98092, 98030, 98119, 98112, 98052, 98027, 98117, 98058,
           98001, 98056, 98166, 98023, 98070, 98148, 98105, 98042, 98008,
           98059, 98122, 98144, 98004, 98005, 98034, 98075, 98116, 98010,
           98118, 98199, 98032, 98045, 98102, 98077, 98108, 98168, 98177,
           98065, 98029, 98006, 98109, 98022, 98033, 98155, 98024, 98011,
           98031, 98106, 98072, 98188, 98014, 98055, 98039])




```python
# create empty dictionary 
dzip = {}

# search pypi uszipcode library to retreive data for each zipcode
for c in zipcodes:
    z = search.by_zipcode(c)
    dzip[c] = z.to_dict()
    
dzip.keys()
```




    dict_keys([98178, 98125, 98028, 98136, 98074, 98053, 98003, 98198, 98146, 98038, 98007, 98115, 98107, 98126, 98019, 98103, 98002, 98133, 98040, 98092, 98030, 98119, 98112, 98052, 98027, 98117, 98058, 98001, 98056, 98166, 98023, 98070, 98148, 98105, 98042, 98008, 98059, 98122, 98144, 98004, 98005, 98034, 98075, 98116, 98010, 98118, 98199, 98032, 98045, 98102, 98077, 98108, 98168, 98177, 98065, 98029, 98006, 98109, 98022, 98033, 98155, 98024, 98011, 98031, 98106, 98072, 98188, 98014, 98055, 98039])




```python
# check information for one of the zipcodes 
# 98032 had the worst p-value (0.838)
dzip[98032]
```




    {'zipcode': '98032',
     'zipcode_type': 'Standard',
     'major_city': 'Kent',
     'post_office_city': 'Kent, WA',
     'common_city_list': ['Kent'],
     'county': 'King County',
     'state': 'WA',
     'lat': 47.4,
     'lng': -122.26,
     'timezone': 'Pacific',
     'radius_in_miles': 5.0,
     'area_code_list': ['253', '425'],
     'population': 33853,
     'population_density': 2024.0,
     'land_area_in_sqmi': 16.72,
     'water_area_in_sqmi': 0.29,
     'housing_units': 14451,
     'occupied_housing_units': 13393,
     'median_home_value': 234700,
     'median_household_income': 48853,
     'bounds_west': -122.309417,
     'bounds_east': -122.217459,
     'bounds_north': 47.441233,
     'bounds_south': 47.34633}




```python
# try retrieving just the median home value for a given zipcode 
dzip[98199]['median_home_value'] #98199 mhv is 3x higher than 98032
```




    606200




```python
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
```


```python
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
```

    [['98039']]
    [['98074'], ['98053'], ['98040'], ['98119'], ['98112'], ['98105'], ['98004'], ['98005'], ['98075'], ['98199'], ['98077'], ['98006'], ['98033'], ['98039']]

