**Reading CSV and TXT files**

**The read_csv method**

The first method we'll learn is read_csv, that let us read comma-separated values (CSV) files and raw text (TXT) files into a DataFrame.

The read_csv function is extremely powerful and you can specify a very broad set of parameters at import time that allow us to accurately configure how the data will be read and parsed by specifying the correct structure, enconding and other details. The most common parameters are as follows:

filepath: Path of the file to be read.
sep: Character(s) that are used as a field separator in the file.
header: Index of the row containing the names of the columns (None if none).
index_col: Index of the column or sequence of indexes that should be used as index of rows of the data.
names: Sequence containing the names of the columns (used together with header = None).
skiprows: Number of rows or sequence of row indexes to ignore in the load.
na_values: Sequence of values that, if found in the file, should be treated as NaN.
dtype: Dictionary in which the keys will be column names and the values will be types of NumPy to which their content must be converted.
parse_dates: Flag that indicates if Python should try to parse data with a format similar to dates as dates. You can enter a list of column names that must be joined for the parsing as a date.
date_parser: Function to use to try to parse dates.
nrows: Number of rows to read from the beginning of the file.
skip_footer: Number of rows to ignore at the end of the file.
encoding: Encoding to be expected from the file read.
squeeze: Flag that indicates that if the data read only contains one column the result is a Series instead of a DataFrame.
thousands: Character to use to detect the thousands separator.
decimal: Character to use to detect the decimal separator.
skip_blank_lines: Flag that indicates whether blank lines should be ignored.
Full read_csv documentation can be found here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html.

In this case we'll try to read our btc-market-price.csv CSV file using different parameters to parse it correctly.

This file contains records of the mean price of Bitcoin per date.


```python
import pandas as pd

df = pd.read_csv('btc-market-price.csv')
```

**Save to CSV file**


```python
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv()
```




    ',col1,col2\r\n0,1,3\r\n1,2,4\r\n'



**Reading data from relational databases**

**Read data from SQL database**

Reading data from SQL relational databases is fairly simple and pandas support a variety of methods to deal with it.

We'll start with an example using SQLite, as it's a builtin Python package, and we don't need anything extra installed.


```python
import sqlite3
```


```python
conn = sqlite3.connect('chinook.db')
```

Once we have a Connection object, we can then create a Cursor object. Cursors allow us to execute SQL queries against a database:


```python
cur = conn.cursor()
```

The Cursor created has a method execute, which will receive SQL parameters to run against the database.

The code below will fetch the first 5 rows from the employees table:


```python
results = cur.fetchall()
```


```python
results
```




    []




```python
df = pd.DataFrame(results)
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



**Using pandas read_sql method** 

We can use the pandas read_sql function to read the results of a SQL query directly into a pandas DataFrame. The code below will execute the same query that we just did, but it will return a DataFrame. It has several advantages over the query we did above:

It doesn't require us to create a Cursor object or call fetchall at the end.
It automatically reads in the names of the headers from the table.
It creates a DataFrame, so we can quickly explore the data.


```python
conn = sqlite3.connect('chinook.db')
```

**Using pandas read_sql_query method**

It turns out that the read_sql method we saw above is just a wrapper around read_sql_query and read_sql_table.

We can get the same result using read_sql_query method:


```python
conn = sqlite3.connect('chinook.db')
```

**Using read_sql_table method**

read_sql_table is a useful function, but it works only with SQLAlchemy, a Python SQL Toolkit and Object Relational Mapper.

This is just a demonstration of its usage where we read the whole employees table.


```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///chinook.db')

connection = engine.connect()
```

**Reading HTML Tables**


```python
!pip install lxml
```

    Requirement already satisfied: lxml in c:\users\ramch\anaconda3\lib\site-packages (4.6.1)
    

**Parsing raw HTML strings**

Another useful pandas method is read_html(). This method will read HTML tables from a given URL, a file-like object, or a raw string containing HTML, and return a list of DataFrame objects.

Let's try to read the following html_string into a DataFrame.


```python
html_string = """
<table>
    <thead>
      <tr>
        <th>Order date</th>
        <th>Region</th> 
        <th>Item</th>
        <th>Units</th>
        <th>Unit cost</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1/6/2018</td>
        <td>East</td> 
        <td>Pencil</td>
        <td>95</td>
        <td>1.99</td>
      </tr>
      <tr>
        <td>1/23/2018</td>
        <td>Central</td> 
        <td>Binder</td>
        <td>50</td>
        <td>19.99</td>
      </tr>
      <tr>
        <td>2/9/2018</td>
        <td>Central</td> 
        <td>Pencil</td>
        <td>36</td>
        <td>4.99</td>
      </tr>
      <tr>
        <td>3/15/2018</td>
        <td>West</td> 
        <td>Pen</td>
        <td>27</td>
        <td>19.99</td>
      </tr>
    </tbody>
</table>
"""
```


```python
dfs = pd.read_html(html_string)
```


```python
len(dfs)
```




    1




```python
df = dfs[0]

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
      <th>Order date</th>
      <th>Region</th>
      <th>Item</th>
      <th>Units</th>
      <th>Unit cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/6/2018</td>
      <td>East</td>
      <td>Pencil</td>
      <td>95</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/23/2018</td>
      <td>Central</td>
      <td>Binder</td>
      <td>50</td>
      <td>19.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/9/2018</td>
      <td>Central</td>
      <td>Pencil</td>
      <td>36</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3/15/2018</td>
      <td>West</td>
      <td>Pen</td>
      <td>27</td>
      <td>19.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (4, 5)




```python
df.loc[df['Region'] == 'Central']
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
      <th>Order date</th>
      <th>Region</th>
      <th>Item</th>
      <th>Units</th>
      <th>Unit cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1/23/2018</td>
      <td>Central</td>
      <td>Binder</td>
      <td>50</td>
      <td>19.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/9/2018</td>
      <td>Central</td>
      <td>Pencil</td>
      <td>36</td>
      <td>4.99</td>
    </tr>
  </tbody>
</table>
</div>



**Defining header**

Pandas will automatically find the header to use thanks to the tag.

But in many cases we'll find wrong or incomplete tables that make the read_html method parse the tables in a wrong way without the proper headers.

To fix them we can use the header parameter.


```python
html_string = """
<table>
  <tr>
    <td>Order date</td>
    <td>Region</td> 
    <td>Item</td>
    <td>Units</td>
    <td>Unit cost</td>
  </tr>
  <tr>
    <td>1/6/2018</td>
    <td>East</td> 
    <td>Pencil</td>
    <td>95</td>
    <td>1.99</td>
  </tr>
  <tr>
    <td>1/23/2018</td>
    <td>Central</td> 
    <td>Binder</td>
    <td>50</td>
    <td>19.99</td>
  </tr>
  <tr>
    <td>2/9/2018</td>
    <td>Central</td> 
    <td>Pencil</td>
    <td>36</td>
    <td>4.99</td>
  </tr>
  <tr>
    <td>3/15/2018</td>
    <td>West</td> 
    <td>Pen</td>
    <td>27</td>
    <td>19.99</td>
  </tr>
</table>
"""
```


```python
pd.read_html(html_string)[0]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Order date</td>
      <td>Region</td>
      <td>Item</td>
      <td>Units</td>
      <td>Unit cost</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/6/2018</td>
      <td>East</td>
      <td>Pencil</td>
      <td>95</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/23/2018</td>
      <td>Central</td>
      <td>Binder</td>
      <td>50</td>
      <td>19.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/9/2018</td>
      <td>Central</td>
      <td>Pencil</td>
      <td>36</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/15/2018</td>
      <td>West</td>
      <td>Pen</td>
      <td>27</td>
      <td>19.99</td>
    </tr>
  </tbody>
</table>
</div>



In this case, we'll need to pass the row number to use as header using the header parameter.


```python
pd.read_html(html_string, header=0)[0]
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
      <th>Order date</th>
      <th>Region</th>
      <th>Item</th>
      <th>Units</th>
      <th>Unit cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/6/2018</td>
      <td>East</td>
      <td>Pencil</td>
      <td>95</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/23/2018</td>
      <td>Central</td>
      <td>Binder</td>
      <td>50</td>
      <td>19.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/9/2018</td>
      <td>Central</td>
      <td>Pencil</td>
      <td>36</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3/15/2018</td>
      <td>West</td>
      <td>Pen</td>
      <td>27</td>
      <td>19.99</td>
    </tr>
  </tbody>
</table>
</div>



**Parsing HTML tables from the web**

Now that we know how read_html works, go one step beyond and try to parse HTML tables directly from an URL.

To do that we'll call the read_html method with an URL as paramter.


```python
html_url = "https://www.basketball-reference.com/leagues/NBA_2019_per_game.html"
```


```python
nba_tables = pd.read_html(html_url)
```


```python
len(nba_tables)
```




    1




```python

```

**Reading Excel files**

process for reading excel files is same as the process of reading csv files.


```python
df=pd.read_excel("products.xlsx")
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
