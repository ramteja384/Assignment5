**Pandas Series**


```python
import pandas as pd
import numpy as np

g7_pop = pd.Series([35.467, 63.951, 80.940, 60.665, 127.061, 64.511, 318.523])
g7_pop
```




    0     35.467
    1     63.951
    2     80.940
    3     60.665
    4    127.061
    5     64.511
    6    318.523
    dtype: float64




```python
g7_pop.name = 'G7 Population in millions'
g7_pop
```




    0     35.467
    1     63.951
    2     80.940
    3     60.665
    4    127.061
    5     64.511
    6    318.523
    Name: G7 Population in millions, dtype: float64



Series are pretty similar to numpy arrays:


```python
g7_pop.dtype
```




    dtype('float64')




```python
g7_pop.size
```




    7




```python
type(g7_pop.values)
```




    numpy.ndarray



And they look like simple Python lists or Numpy Arrays. But they're actually more similar to Python dicts.

A Series has an index, that's similar to the automatic index assigned to Python's lists:


```python
g7_pop
```




    0     35.467
    1     63.951
    2     80.940
    3     60.665
    4    127.061
    5     64.511
    6    318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop[0]
```




    35.467




```python
g7_pop[1]
```




    63.951




```python
g7_pop.index
```




    RangeIndex(start=0, stop=7, step=1)



RangeIndex(start=0, stop=7, step=1)


```python
l = ['a', 'b', 'c']
```

But, in contrast to lists, we can explicitly define the index:


```python
g7_pop.index = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]
```


```python
g7_pop
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64



We can say that Series look like "ordered dictionaries". We can actually create Series out of dictionaries:


```python
pd.Series({
    'Canada': 35.467,
    'France': 63.951,
    'Germany': 80.94,
    'Italy': 60.665,
    'Japan': 127.061,
    'United Kingdom': 64.511,
    'United States': 318.523
}, name='G7 Population in millions')
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64




```python
pd.Series(
    [35.467, 63.951, 80.94, 60.665, 127.061, 64.511, 318.523],
    index=['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom',
       'United States'],
    name='G7 Population in millions')
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64



You can also create series out of other series


```python
pd.Series(g7_pop, index=['France', 'Germany', 'Italy', 'Spain'])
```




    France     63.951
    Germany    80.940
    Italy      60.665
    Spain         NaN
    Name: G7 Population in millions, dtype: float64



**Indexing**

Indexing works similarly to lists and dictionaries, you use the index of the element you're looking for:


```python
g7_pop
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop['Canada']
```




    35.467




```python
g7_pop['Japan']
```




    127.061



Numeric positions can also be used, with the iloc attribute:


```python
g7_pop.iloc[0]
```




    35.467




```python
g7_pop.iloc[0:2]
```




    Canada    35.467
    France    63.951
    Name: G7 Population in millions, dtype: float64




```python
g7_pop.iloc[::]
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop.iloc[0:-1]
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    Name: G7 Population in millions, dtype: float64



Selecting multiple elements at once:


```python
g7_pop[['Italy','France']]
```




    Italy     60.665
    France    63.951
    Name: G7 Population in millions, dtype: float64



Slicing also works, but important, in Pandas, the upper limit is also included:


```python
g7_pop['Canada':'Japan']
```




    Canada      35.467
    France      63.951
    Germany     80.940
    Italy       60.665
    Japan      127.061
    Name: G7 Population in millions, dtype: float64



**Conditional selection (boolean arrays)**

The same boolean array techniques we saw applied to numpy arrays can be used for Pandas Series:


```python
g7_pop
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop > 70
```




    Canada            False
    France            False
    Germany            True
    Italy             False
    Japan              True
    United Kingdom    False
    United States      True
    Name: G7 Population in millions, dtype: bool




```python
g7_pop[g7_pop > 70]
```




    Germany           80.940
    Japan            127.061
    United States    318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop.mean()
```




    107.30257142857144




```python
g7_pop[g7_pop > g7_pop.mean()]
```




    Japan            127.061
    United States    318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop.std()
```




    97.24996987121581



**Operations and methods**

Series also support vectorized operations and aggregation functions as Numpy:


```python
g7_pop
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop * 1_000_000
```




    Canada             35467000.0
    France             63951000.0
    Germany            80940000.0
    Italy              60665000.0
    Japan             127061000.0
    United Kingdom     64511000.0
    United States     318523000.0
    Name: G7 Population in millions, dtype: float64




```python
g7_pop.mean()
```




    107.30257142857144




```python
np.log(g7_pop)
```




    Canada            3.568603
    France            4.158117
    Germany           4.393708
    Italy             4.105367
    Japan             4.844667
    United Kingdom    4.166836
    United States     5.763695
    Name: G7 Population in millions, dtype: float64




```python
g7_pop['France': 'Italy'].mean()
```




    68.51866666666666



**Boolean arrays**


```python
g7_pop
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop > 80
```




    Canada            False
    France            False
    Germany            True
    Italy             False
    Japan              True
    United Kingdom    False
    United States      True
    Name: G7 Population in millions, dtype: bool




```python
g7_pop[g7_pop > 80]
```




    Germany           80.940
    Japan            127.061
    United States    318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop[(g7_pop > 80) | (g7_pop < 40)]
```




    Canada            35.467
    Germany           80.940
    Japan            127.061
    United States    318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop[(g7_pop > 80) & (g7_pop < 200)]
```




    Germany     80.940
    Japan      127.061
    Name: G7 Population in millions, dtype: float64



**Modifying Series**


```python
g7_pop['Canada'] = 40.5
g7_pop
```




    Canada             40.500
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: G7 Population in millions, dtype: float64




```python
g7_pop.iloc[-1] = 500
g7_pop
```




    Canada             40.500
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     500.000
    Name: G7 Population in millions, dtype: float64




```python
g7_pop[g7_pop < 70]
```




    Canada            40.500
    France            63.951
    Italy             60.665
    United Kingdom    64.511
    Name: G7 Population in millions, dtype: float64




```python
g7_pop[g7_pop < 70] = 99.99
```


```python
g7_pop
```




    Canada             99.990
    France             99.990
    Germany            80.940
    Italy              99.990
    Japan             127.061
    United Kingdom     99.990
    United States     500.000
    Name: G7 Population in millions, dtype: float64




```python

```

***Pandas - Dataframes***

Creating DataFrames manually can be tedious. 99% of the time you'll be pulling the data from a Database, a csv file or the web. But still, you can create a DataFrame by specifying the columns and values:


```python
df = pd.DataFrame({
    'Population': [35.467, 63.951, 80.94 , 60.665, 127.061, 64.511, 318.523],
    'GDP': [
        1785387,
        2833687,
        3874437,
        2167744,
        4602367,
        2950039,
        17348075
    ],
    'Surface Area': [
        9984670,
        640679,
        357114,
        301336,
        377930,
        242495,
        9525067
    ],
    'HDI': [
        0.913,
        0.888,
        0.916,
        0.873,
        0.891,
        0.907,
        0.915
    ],
    'Continent': [
        'America',
        'Europe',
        'Europe',
        'Europe',
        'Asia',
        'Europe',
        'America'
    ]
}, columns=['Population', 'GDP', 'Surface Area', 'HDI', 'Continent'])
```


```python
df
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>5</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>6</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>



DataFrames also have indexes. As you can see in the "table" above, pandas has assigned a numeric, autoincremental index automatically to each "row" in our DataFrame. In our case, we know that each row represents a country, so we'll just reassign the index:



```python
df.index = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]
```


```python
df
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Population', 'GDP', 'Surface Area', 'HDI', 'Continent'], dtype='object')




```python
df.index
```




    Index(['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom',
           'United States'],
          dtype='object')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 7 entries, Canada to United States
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Population    7 non-null      float64
     1   GDP           7 non-null      int64  
     2   Surface Area  7 non-null      int64  
     3   HDI           7 non-null      float64
     4   Continent     7 non-null      object 
    dtypes: float64(2), int64(2), object(1)
    memory usage: 336.0+ bytes
    


```python
df.size
```




    35




```python
df.shape
```




    (7, 5)




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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.000000</td>
      <td>7.000000e+00</td>
      <td>7.000000e+00</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>107.302571</td>
      <td>5.080248e+06</td>
      <td>3.061327e+06</td>
      <td>0.900429</td>
    </tr>
    <tr>
      <th>std</th>
      <td>97.249970</td>
      <td>5.494020e+06</td>
      <td>4.576187e+06</td>
      <td>0.016592</td>
    </tr>
    <tr>
      <th>min</th>
      <td>35.467000</td>
      <td>1.785387e+06</td>
      <td>2.424950e+05</td>
      <td>0.873000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>62.308000</td>
      <td>2.500716e+06</td>
      <td>3.292250e+05</td>
      <td>0.889500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>64.511000</td>
      <td>2.950039e+06</td>
      <td>3.779300e+05</td>
      <td>0.907000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>104.000500</td>
      <td>4.238402e+06</td>
      <td>5.082873e+06</td>
      <td>0.914000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>318.523000</td>
      <td>1.734808e+07</td>
      <td>9.984670e+06</td>
      <td>0.916000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    Population      float64
    GDP               int64
    Surface Area      int64
    HDI             float64
    Continent        object
    dtype: object




```python
df.dtypes.value_counts()
```




    float64    2
    int64      2
    object     1
    dtype: int64



**Indexing, Selection and Slicing**

Individual columns in the DataFrame can be selected with regular indexing. Each column is represented as a Series:


```python
df
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['Canada']
```




    Population       35.467
    GDP             1785387
    Surface Area    9984670
    HDI               0.913
    Continent       America
    Name: Canada, dtype: object




```python
df.iloc[-1]
```




    Population       318.523
    GDP             17348075
    Surface Area     9525067
    HDI                0.915
    Continent        America
    Name: United States, dtype: object




```python
df['Population']
```




    Canada             35.467
    France             63.951
    Germany            80.940
    Italy              60.665
    Japan             127.061
    United Kingdom     64.511
    United States     318.523
    Name: Population, dtype: float64



Note that the index of the returned Series is the same as the DataFrame one. And its name is the name of the column. If you're working on a notebook and want to see a more DataFrame-like format you can use the to_frame method:



```python
df['Population'].to_frame()

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
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
    </tr>
  </tbody>
</table>
</div>



Multiple columns can also be selected similarly to numpy and Series:



```python
df[['Population', 'GDP']]

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
      <th>Population</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
    </tr>
  </tbody>
</table>
</div>



In this case, the result is another DataFrame. Slicing works differently, it acts at "row level", and can be counter intuitive:



```python
df[1:3]

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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
  </tbody>
</table>
</div>



Row level selection works better with loc and iloc which are recommended over regular "direct slicing" (df[:]).

loc selects rows matching the given index:



```python
df.loc['Italy']
```




    Population       60.665
    GDP             2167744
    Surface Area     301336
    HDI               0.873
    Continent        Europe
    Name: Italy, dtype: object




```python
df.loc['France': 'Italy']
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
  </tbody>
</table>
</div>



As a second "argument", you can pass the column(s) you'd like to select:



```python
df.loc['France': 'Italy', 'Population']
```




    France     63.951
    Germany    80.940
    Italy      60.665
    Name: Population, dtype: float64




```python
df.loc['France': 'Italy', ['Population', 'GDP']]
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
      <th>Population</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
    </tr>
  </tbody>
</table>
</div>



iloc works with the (numeric) "position" of the index:


```python
df
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[ -1]
```




    Population       318.523
    GDP             17348075
    Surface Area     9525067
    HDI                0.915
    Continent        America
    Name: United States, dtype: object




```python
df.iloc[0]
```




    Population       35.467
    GDP             1785387
    Surface Area    9984670
    HDI               0.913
    Continent       America
    Name: Canada, dtype: object




```python
df.iloc[[0, 1, -1]]
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1:3]
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1:3,3]
```




    France     0.888
    Germany    0.916
    Name: HDI, dtype: float64




```python
df.iloc[1:3, [0, 3]]
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
      <th>Population</th>
      <th>HDI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>0.888</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>0.916</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1:3, 1:3]
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
      <th>GDP</th>
      <th>Surface Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>2833687</td>
      <td>640679</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>3874437</td>
      <td>357114</td>
    </tr>
  </tbody>
</table>
</div>



**RECOMMENDED: Always use loc and iloc to reduce ambiguity, specially with DataFrames with numeric indexes.**

**Conditional selection (boolean arrays)**

We saw conditional selection applied to Series and it'll work in the same way for DataFrames. After all, a DataFrame is a collection of Series:


```python
df
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Population'] > 70
```




    Canada            False
    France            False
    Germany            True
    Italy             False
    Japan              True
    United Kingdom    False
    United States      True
    Name: Population, dtype: bool




```python
df.loc[df['Population'] > 70]
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>



The boolean matching is done at Index level, so you can filter by any row, as long as it contains the right indexes. Column selection still works as expected:


```python
df.loc[df['Population'] > 70, 'Population']
```




    Germany           80.940
    Japan            127.061
    United States    318.523
    Name: Population, dtype: float64




```python
df.loc[df['Population'] > 70, ['Population', 'GDP']]
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
      <th>Population</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
    </tr>
  </tbody>
</table>
</div>



**Dropping stuff**

Opposed to the concept of selection, we have "dropping". Instead of pointing out which values you'd like to select you could point which ones you'd like to drop:


```python
df.drop('Canada')
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['Canada', 'Japan'])
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(columns=['Population', 'HDI'])
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
      <th>GDP</th>
      <th>Surface Area</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>1785387</td>
      <td>9984670</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>2833687</td>
      <td>640679</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>3874437</td>
      <td>357114</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>2167744</td>
      <td>301336</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>4602367</td>
      <td>377930</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>2950039</td>
      <td>242495</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>17348075</td>
      <td>9525067</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['Italy', 'Canada'], axis=0)
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['Population', 'HDI'], axis=1)
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
      <th>GDP</th>
      <th>Surface Area</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>1785387</td>
      <td>9984670</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>2833687</td>
      <td>640679</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>3874437</td>
      <td>357114</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>2167744</td>
      <td>301336</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>4602367</td>
      <td>377930</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>2950039</td>
      <td>242495</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>17348075</td>
      <td>9525067</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['Population', 'HDI'], axis=1)
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
      <th>GDP</th>
      <th>Surface Area</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>1785387</td>
      <td>9984670</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>2833687</td>
      <td>640679</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>3874437</td>
      <td>357114</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>2167744</td>
      <td>301336</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>4602367</td>
      <td>377930</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>2950039</td>
      <td>242495</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>17348075</td>
      <td>9525067</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['Population', 'HDI'], axis='columns')
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
      <th>GDP</th>
      <th>Surface Area</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>1785387</td>
      <td>9984670</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>2833687</td>
      <td>640679</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>3874437</td>
      <td>357114</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>2167744</td>
      <td>301336</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>4602367</td>
      <td>377930</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>2950039</td>
      <td>242495</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>17348075</td>
      <td>9525067</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['Canada', 'Germany'], axis='rows')
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>



All these drop methods return a new DataFrame. If you'd like to modify it "in place", you can use the inplace attribute (there's an example below).

**Operations**


```python
df[['Population', 'GDP']]
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
      <th>Population</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Population', 'GDP']]/100
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
      <th>Population</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>0.35467</td>
      <td>17853.87</td>
    </tr>
    <tr>
      <th>France</th>
      <td>0.63951</td>
      <td>28336.87</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>0.80940</td>
      <td>38744.37</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>0.60665</td>
      <td>21677.44</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>1.27061</td>
      <td>46023.67</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>0.64511</td>
      <td>29500.39</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>3.18523</td>
      <td>173480.75</td>
    </tr>
  </tbody>
</table>
</div>



**Operations with Series** work at a column level, broadcasting down the rows (which can be counter intuitive).


```python
crisis = pd.Series([-1_000_000, -0.3], index=['GDP', 'HDI'])
crisis
```




    GDP   -1000000.0
    HDI         -0.3
    dtype: float64




```python
df[['GDP', 'HDI']]
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
      <th>GDP</th>
      <th>HDI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>1785387</td>
      <td>0.913</td>
    </tr>
    <tr>
      <th>France</th>
      <td>2833687</td>
      <td>0.888</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>3874437</td>
      <td>0.916</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>2167744</td>
      <td>0.873</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>4602367</td>
      <td>0.891</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>2950039</td>
      <td>0.907</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>17348075</td>
      <td>0.915</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['GDP', 'HDI']] + crisis
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
      <th>GDP</th>
      <th>HDI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>785387.0</td>
      <td>0.613</td>
    </tr>
    <tr>
      <th>France</th>
      <td>1833687.0</td>
      <td>0.588</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>2874437.0</td>
      <td>0.616</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>1167744.0</td>
      <td>0.573</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>3602367.0</td>
      <td>0.591</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>1950039.0</td>
      <td>0.607</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>16348075.0</td>
      <td>0.615</td>
    </tr>
  </tbody>
</table>
</div>



**Modifying DataFrames**

It's simple and intuitive, You can add columns, or replace values for columns without issues:

 **Adding a new column**


```python
langs = pd.Series(
    ['French', 'German', 'Italian'],
    index=['France', 'Germany', 'Italy'],
    name='Language'
)
```


```python
langs
```




    France      French
    Germany     German
    Italy      Italian
    Name: Language, dtype: object




```python
df['Language'] = langs
```


```python
df
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
      <td>French</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
      <td>German</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
      <td>Italian</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Replacing values per column**


```python
df['Language'] = 'English'
df
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
      <td>English</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
      <td>English</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
      <td>English</td>
    </tr>
  </tbody>
</table>
</div>



**Renaming Columns**


```python
df.rename(
    columns={
        'HDI': 'Human Development Index',
        'Anual Popcorn Consumption': 'APC'
    }, index={
        'United States': 'USA',
        'United Kingdom': 'UK',
        'Argentina': 'AR'
    })
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>Human Development Index</th>
      <th>Continent</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
      <td>English</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
      <td>English</td>
    </tr>
    <tr>
      <th>UK</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
      <td>English</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rename(index=str.upper)
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANADA</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
      <td>English</td>
    </tr>
    <tr>
      <th>FRANCE</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>GERMANY</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>ITALY</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>JAPAN</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
      <td>English</td>
    </tr>
    <tr>
      <th>UNITED KINGDOM</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>UNITED STATES</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
      <td>English</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rename(index=lambda x: x.lower())
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
      <th>Population</th>
      <th>GDP</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
      <th>Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>canada</th>
      <td>35.467</td>
      <td>1785387</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
      <td>English</td>
    </tr>
    <tr>
      <th>france</th>
      <td>63.951</td>
      <td>2833687</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>germany</th>
      <td>80.940</td>
      <td>3874437</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>italy</th>
      <td>60.665</td>
      <td>2167744</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>japan</th>
      <td>127.061</td>
      <td>4602367</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
      <td>English</td>
    </tr>
    <tr>
      <th>united kingdom</th>
      <td>64.511</td>
      <td>2950039</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
      <td>English</td>
    </tr>
    <tr>
      <th>united states</th>
      <td>318.523</td>
      <td>17348075</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
      <td>English</td>
    </tr>
  </tbody>
</table>
</div>



** Dropping columns**


```python
df.drop(columns='GDP', inplace=True)
```

**Adding values**


```python
df.append(pd.Series({
    'Population': 3,
    'GDP': 5
}, name='China'))
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
      <th>Population</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>9984670.0</td>
      <td>0.913</td>
      <td>America</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>640679.0</td>
      <td>0.888</td>
      <td>Europe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>357114.0</td>
      <td>0.916</td>
      <td>Europe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>301336.0</td>
      <td>0.873</td>
      <td>Europe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>377930.0</td>
      <td>0.891</td>
      <td>Asia</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>242495.0</td>
      <td>0.907</td>
      <td>Europe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>9525067.0</td>
      <td>0.915</td>
      <td>America</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>China</th>
      <td>3.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



Append returns a new DataFrame:


```python
df
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
      <th>Population</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>9984670</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>640679</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>357114</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>301336</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>377930</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>242495</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>9525067</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>



You can directly set the new index and values to the DataFrame:


```python
df.loc['China'] = pd.Series({'Population': 1_400_000_000, 'Continent': 'Asia'})
df
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
      <th>Population</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>3.546700e+01</td>
      <td>9984670.0</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>6.395100e+01</td>
      <td>640679.0</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>8.094000e+01</td>
      <td>357114.0</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>6.066500e+01</td>
      <td>301336.0</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>1.270610e+02</td>
      <td>377930.0</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>6.451100e+01</td>
      <td>242495.0</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>3.185230e+02</td>
      <td>9525067.0</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
    <tr>
      <th>China</th>
      <td>1.400000e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Asia</td>
    </tr>
  </tbody>
</table>
</div>



We can use drop to just remove a row by index:


```python
df.drop('China', inplace=True)
df
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
      <th>Population</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>9984670.0</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>640679.0</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>357114.0</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>301336.0</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>377930.0</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>242495.0</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>9525067.0</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>



**More radical index changes**


```python
df.reset_index()
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
      <th>index</th>
      <th>Population</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Canada</td>
      <td>35.467</td>
      <td>9984670.0</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>63.951</td>
      <td>640679.0</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>80.940</td>
      <td>357114.0</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Italy</td>
      <td>60.665</td>
      <td>301336.0</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Japan</td>
      <td>127.061</td>
      <td>377930.0</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>5</th>
      <td>United Kingdom</td>
      <td>64.511</td>
      <td>242495.0</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>6</th>
      <td>United States</td>
      <td>318.523</td>
      <td>9525067.0</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index('Population')
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
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
    </tr>
    <tr>
      <th>Population</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35.467</th>
      <td>9984670.0</td>
      <td>0.913</td>
      <td>America</td>
    </tr>
    <tr>
      <th>63.951</th>
      <td>640679.0</td>
      <td>0.888</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>80.940</th>
      <td>357114.0</td>
      <td>0.916</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>60.665</th>
      <td>301336.0</td>
      <td>0.873</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>127.061</th>
      <td>377930.0</td>
      <td>0.891</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>64.511</th>
      <td>242495.0</td>
      <td>0.907</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>318.523</th>
      <td>9525067.0</td>
      <td>0.915</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>



**Creating columns from other columns**

Altering a DataFrame often involves combining different columns into another. For example, in our Countries analysis, we could try to calculate the "HDI per capita", which is just, HDI / Population.


```python
df[['Population', 'HDI']]
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
      <th>Population</th>
      <th>HDI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>0.913</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>0.888</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>0.916</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>0.873</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>0.891</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>0.907</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>0.915</td>
    </tr>
  </tbody>
</table>
</div>



The regular pandas way of expressing that, is just dividing each series:


```python
df['HDI'] / df['Population']
```




    Canada            0.025742
    France            0.013886
    Germany           0.011317
    Italy             0.014391
    Japan             0.007012
    United Kingdom    0.014060
    United States     0.002873
    dtype: float64



The result of that operation is just another series that you can add to the original DataFrame:


```python
df['HDI Per Capita'] = df['HDI'] / df['Population']
df
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
      <th>Population</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
      <th>HDI Per Capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>9984670.0</td>
      <td>0.913</td>
      <td>America</td>
      <td>0.025742</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>640679.0</td>
      <td>0.888</td>
      <td>Europe</td>
      <td>0.013886</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>357114.0</td>
      <td>0.916</td>
      <td>Europe</td>
      <td>0.011317</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>301336.0</td>
      <td>0.873</td>
      <td>Europe</td>
      <td>0.014391</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>377930.0</td>
      <td>0.891</td>
      <td>Asia</td>
      <td>0.007012</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>64.511</td>
      <td>242495.0</td>
      <td>0.907</td>
      <td>Europe</td>
      <td>0.014060</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>318.523</td>
      <td>9525067.0</td>
      <td>0.915</td>
      <td>America</td>
      <td>0.002873</td>
    </tr>
  </tbody>
</table>
</div>



**Statistical info**

You've already seen the describe method, which gives you a good "summary" of the DataFrame. Let's explore other methods in more detail:


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
      <th>Population</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>Continent</th>
      <th>HDI Per Capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Canada</th>
      <td>35.467</td>
      <td>9984670.0</td>
      <td>0.913</td>
      <td>America</td>
      <td>0.025742</td>
    </tr>
    <tr>
      <th>France</th>
      <td>63.951</td>
      <td>640679.0</td>
      <td>0.888</td>
      <td>Europe</td>
      <td>0.013886</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>80.940</td>
      <td>357114.0</td>
      <td>0.916</td>
      <td>Europe</td>
      <td>0.011317</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>60.665</td>
      <td>301336.0</td>
      <td>0.873</td>
      <td>Europe</td>
      <td>0.014391</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>127.061</td>
      <td>377930.0</td>
      <td>0.891</td>
      <td>Asia</td>
      <td>0.007012</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>Population</th>
      <th>Surface Area</th>
      <th>HDI</th>
      <th>HDI Per Capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.000000</td>
      <td>7.000000e+00</td>
      <td>7.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>107.302571</td>
      <td>3.061327e+06</td>
      <td>0.900429</td>
      <td>0.012754</td>
    </tr>
    <tr>
      <th>std</th>
      <td>97.249970</td>
      <td>4.576187e+06</td>
      <td>0.016592</td>
      <td>0.007153</td>
    </tr>
    <tr>
      <th>min</th>
      <td>35.467000</td>
      <td>2.424950e+05</td>
      <td>0.873000</td>
      <td>0.002873</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>62.308000</td>
      <td>3.292250e+05</td>
      <td>0.889500</td>
      <td>0.009165</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>64.511000</td>
      <td>3.779300e+05</td>
      <td>0.907000</td>
      <td>0.013886</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>104.000500</td>
      <td>5.082873e+06</td>
      <td>0.914000</td>
      <td>0.014225</td>
    </tr>
    <tr>
      <th>max</th>
      <td>318.523000</td>
      <td>9.984670e+06</td>
      <td>0.916000</td>
      <td>0.025742</td>
    </tr>
  </tbody>
</table>
</div>




```python
population = df['Population']
```


```python
population.min(), population.max()
```




    (35.467, 318.523)




```python
population.sum()
```




    751.118




```python
population.sum() / len(population)
```




    107.30257142857144




```python
population.mean()
```




    107.30257142857144




```python
population.std()
```




    97.24996987121581




```python
population.median()
```




    64.511




```python
population.describe()
```




    count      7.000000
    mean     107.302571
    std       97.249970
    min       35.467000
    25%       62.308000
    50%       64.511000
    75%      104.000500
    max      318.523000
    Name: Population, dtype: float64




```python
population.quantile(.25)
```




    62.308




```python
population.quantile([.2, .4, .6, .8, 1])
```




    0.2     61.3222
    0.4     64.1750
    0.6     74.3684
    0.8    117.8368
    1.0    318.5230
    Name: Population, dtype: float64




```python

```

**Reading external data & Plotting**


```python
import matplotlib.pyplot as plt

%matplotlib inline
```

Pandas can easily read data stored in different file formats like CSV, JSON, XML or even Excel. Parsing always involves specifying the correct structure, encoding and other details. The read_csv method reads CSV files and accepts many parameters.


```python
df=pd.read_csv("coin_Aave.csv")
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
      <th>SNo</th>
      <th>Name</th>
      <th>Symbol</th>
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Marketcap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-05 23:59:59</td>
      <td>55.112358</td>
      <td>49.787900</td>
      <td>52.675035</td>
      <td>53.219243</td>
      <td>0.000000e+00</td>
      <td>8.912813e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-06 23:59:59</td>
      <td>53.402270</td>
      <td>40.734578</td>
      <td>53.291969</td>
      <td>42.401599</td>
      <td>5.830915e+05</td>
      <td>7.101144e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-07 23:59:59</td>
      <td>42.408314</td>
      <td>35.970690</td>
      <td>42.399947</td>
      <td>40.083976</td>
      <td>6.828342e+05</td>
      <td>6.713004e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-08 23:59:59</td>
      <td>44.902511</td>
      <td>36.696057</td>
      <td>39.885262</td>
      <td>43.764463</td>
      <td>1.658817e+06</td>
      <td>2.202651e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-09 23:59:59</td>
      <td>47.569533</td>
      <td>43.291776</td>
      <td>43.764463</td>
      <td>46.817744</td>
      <td>8.155377e+05</td>
      <td>2.356322e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['Date', 'Close']
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-148-3e07e2546cb6> in <module>
    ----> 1 df.columns = ['Date', 'Close']
    

    ~\Anaconda3\lib\site-packages\pandas\core\generic.py in __setattr__(self, name, value)
       5150         try:
       5151             object.__getattribute__(self, name)
    -> 5152             return object.__setattr__(self, name, value)
       5153         except AttributeError:
       5154             pass
    

    pandas\_libs\properties.pyx in pandas._libs.properties.AxisProperty.__set__()
    

    ~\Anaconda3\lib\site-packages\pandas\core\generic.py in _set_axis(self, axis, labels)
        562     def _set_axis(self, axis: int, labels: Index) -> None:
        563         labels = ensure_index(labels)
    --> 564         self._mgr.set_axis(axis, labels)
        565         self._clear_item_cache()
        566 
    

    ~\Anaconda3\lib\site-packages\pandas\core\internals\managers.py in set_axis(self, axis, new_labels)
        224 
        225         if new_len != old_len:
    --> 226             raise ValueError(
        227                 f"Length mismatch: Expected axis has {old_len} elements, new "
        228                 f"values have {new_len} elements"
    

    ValueError: Length mismatch: Expected axis has 10 elements, new values have 2 elements



```python
df.shape
```




    (275, 10)




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
      <th>SNo</th>
      <th>Name</th>
      <th>Symbol</th>
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Marketcap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-05 23:59:59</td>
      <td>55.112358</td>
      <td>49.787900</td>
      <td>52.675035</td>
      <td>53.219243</td>
      <td>0.000000e+00</td>
      <td>8.912813e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-06 23:59:59</td>
      <td>53.402270</td>
      <td>40.734578</td>
      <td>53.291969</td>
      <td>42.401599</td>
      <td>5.830915e+05</td>
      <td>7.101144e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-07 23:59:59</td>
      <td>42.408314</td>
      <td>35.970690</td>
      <td>42.399947</td>
      <td>40.083976</td>
      <td>6.828342e+05</td>
      <td>6.713004e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-08 23:59:59</td>
      <td>44.902511</td>
      <td>36.696057</td>
      <td>39.885262</td>
      <td>43.764463</td>
      <td>1.658817e+06</td>
      <td>2.202651e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-09 23:59:59</td>
      <td>47.569533</td>
      <td>43.291776</td>
      <td>43.764463</td>
      <td>46.817744</td>
      <td>8.155377e+05</td>
      <td>2.356322e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(3)
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
      <th>SNo</th>
      <th>Name</th>
      <th>Symbol</th>
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Marketcap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>272</th>
      <td>273</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2021-07-04 23:59:59</td>
      <td>289.001124</td>
      <td>248.285491</td>
      <td>259.399426</td>
      <td>277.038792</td>
      <td>4.275719e+08</td>
      <td>3.555054e+09</td>
    </tr>
    <tr>
      <th>273</th>
      <td>274</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2021-07-05 23:59:59</td>
      <td>317.387234</td>
      <td>263.433881</td>
      <td>277.110533</td>
      <td>307.829079</td>
      <td>7.931409e+08</td>
      <td>3.950269e+09</td>
    </tr>
    <tr>
      <th>274</th>
      <td>275</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2021-07-06 23:59:59</td>
      <td>346.714780</td>
      <td>307.997525</td>
      <td>307.997525</td>
      <td>316.898507</td>
      <td>9.887055e+08</td>
      <td>4.066776e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    SNo            int64
    Name          object
    Symbol        object
    Date          object
    High         float64
    Low          float64
    Open         float64
    Close        float64
    Volume       float64
    Marketcap    float64
    dtype: object




```python
pd.to_datetime(df['Date']).head()
```




    0   2020-10-05 23:59:59
    1   2020-10-06 23:59:59
    2   2020-10-07 23:59:59
    3   2020-10-08 23:59:59
    4   2020-10-09 23:59:59
    Name: Date, dtype: datetime64[ns]




```python
df['Date'] = pd.to_datetime(df['Date'])
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
      <th>SNo</th>
      <th>Name</th>
      <th>Symbol</th>
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Marketcap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-05 23:59:59</td>
      <td>55.112358</td>
      <td>49.787900</td>
      <td>52.675035</td>
      <td>53.219243</td>
      <td>0.000000e+00</td>
      <td>8.912813e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-06 23:59:59</td>
      <td>53.402270</td>
      <td>40.734578</td>
      <td>53.291969</td>
      <td>42.401599</td>
      <td>5.830915e+05</td>
      <td>7.101144e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-07 23:59:59</td>
      <td>42.408314</td>
      <td>35.970690</td>
      <td>42.399947</td>
      <td>40.083976</td>
      <td>6.828342e+05</td>
      <td>6.713004e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-08 23:59:59</td>
      <td>44.902511</td>
      <td>36.696057</td>
      <td>39.885262</td>
      <td>43.764463</td>
      <td>1.658817e+06</td>
      <td>2.202651e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-09 23:59:59</td>
      <td>47.569533</td>
      <td>43.291776</td>
      <td>43.764463</td>
      <td>46.817744</td>
      <td>8.155377e+05</td>
      <td>2.356322e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    SNo                   int64
    Name                 object
    Symbol               object
    Date         datetime64[ns]
    High                float64
    Low                 float64
    Open                float64
    Close               float64
    Volume              float64
    Marketcap           float64
    dtype: object



The timestamp looks a lot like the index of this DataFrame: date > price. We can change the autoincremental ID generated by pandas and use the Timestamp DS column as the Index:


```python
df.set_index('Date', inplace=True)
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
      <th>SNo</th>
      <th>Name</th>
      <th>Symbol</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Marketcap</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2020-10-05 23:59:59</th>
      <td>1</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>55.112358</td>
      <td>49.787900</td>
      <td>52.675035</td>
      <td>53.219243</td>
      <td>0.000000e+00</td>
      <td>8.912813e+07</td>
    </tr>
    <tr>
      <th>2020-10-06 23:59:59</th>
      <td>2</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>53.402270</td>
      <td>40.734578</td>
      <td>53.291969</td>
      <td>42.401599</td>
      <td>5.830915e+05</td>
      <td>7.101144e+07</td>
    </tr>
    <tr>
      <th>2020-10-07 23:59:59</th>
      <td>3</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>42.408314</td>
      <td>35.970690</td>
      <td>42.399947</td>
      <td>40.083976</td>
      <td>6.828342e+05</td>
      <td>6.713004e+07</td>
    </tr>
    <tr>
      <th>2020-10-08 23:59:59</th>
      <td>4</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>44.902511</td>
      <td>36.696057</td>
      <td>39.885262</td>
      <td>43.764463</td>
      <td>1.658817e+06</td>
      <td>2.202651e+08</td>
    </tr>
    <tr>
      <th>2020-10-09 23:59:59</th>
      <td>5</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>47.569533</td>
      <td>43.291776</td>
      <td>43.764463</td>
      <td>46.817744</td>
      <td>8.155377e+05</td>
      <td>2.356322e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['2020-10-09']
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
      <th>SNo</th>
      <th>Name</th>
      <th>Symbol</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Marketcap</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2020-10-09 23:59:59</th>
      <td>5</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>47.569533</td>
      <td>43.291776</td>
      <td>43.764463</td>
      <td>46.817744</td>
      <td>815537.660784</td>
      <td>2.356322e+08</td>
    </tr>
  </tbody>
</table>
</div>



**Putting everything together**

And now, we've finally arrived to the final, desired version of the DataFrame parsed from our CSV file. The steps were:


```python
df = pd.read_csv('data/btc-marke.csv', header=None)
df.columns = ['Timestamp', 'Price']
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SNo</td>
      <td>Name</td>
      <td>Symbol</td>
      <td>Date</td>
      <td>High</td>
      <td>Low</td>
      <td>Open</td>
      <td>Close</td>
      <td>Volume</td>
      <td>Marketcap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-05 23:59:59</td>
      <td>55.11235847</td>
      <td>49.78789992</td>
      <td>52.67503496</td>
      <td>53.21924296</td>
      <td>0.0</td>
      <td>89128128.86084658</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-06 23:59:59</td>
      <td>53.40227002</td>
      <td>40.73457791</td>
      <td>53.29196931</td>
      <td>42.40159861</td>
      <td>583091.4597628</td>
      <td>71011441.25451232</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-07 23:59:59</td>
      <td>42.40831364</td>
      <td>35.97068975</td>
      <td>42.39994711</td>
      <td>40.08397561</td>
      <td>682834.18632335</td>
      <td>67130036.89981823</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-08 23:59:59</td>
      <td>44.90251114</td>
      <td>36.69605677</td>
      <td>39.88526234</td>
      <td>43.76446306</td>
      <td>1658816.92260445</td>
      <td>220265142.10956782</td>
    </tr>
  </tbody>
</table>
</div>



**Plotting basics**

pandas integrates with Matplotlib and creating a plot is as simple as:


```python
f=df.plot
```


```python
f
```




    <pandas.plotting._core.PlotAccessor object at 0x000001BB979694F0>




```python

```


```python

```


```python

```
