**Syntax Overview**

The following paragraphs will be dedicated to Python's syntax and technical details. There are more to Python than just syntax, as its community, events, email lists, etc. But after all, this is just a technical introduction.

Indentation based
This might feel weird at first, but in Python we do NOT use curly braces to denote blocks of code. We use "indentation" instead. This is similar to Ruby. For example, this is a very simple add_numbers function in javascript:

function add_numbers(x, y){
    return x + y
}
In Python, we'd write it in this way:


```python
def add_numbers(x, y):
    return x + y
```

An if-else block in Javascript:

let language = "Python"

if (language === "Python"){
    console.log("Let the fun begin");
} else {
    console.log("You sure?");
}
In Python:

**Comments**

You've seen comments in the previous block of code: they're prefixed with a pound/hashtag sign:


```python
# this is a comment
```


```python

```

**Variables**

We've defined a variable language in one of our previous examples. In Python, you can set a variable at any time, in any block of code, by just assigning a valid name to any value you want:


```python
name = "Mary"
print(name)
age = 30
print(age)
```

    Mary
    30
    

Variables, once set, will be preserved:


```python
print(name, "is", age, "years old")
```

    Mary is 30 years old
    

**Data Types**

Python supports the most common data types, the usual suspects we could say:

**Integers, type int:**
    
Integers have unlimited magnitude.


```python
age = 30
age
```




    30




```python
type(age)
```




    int



**Float**


```python
b=2.5
type(b)
```




    float



**Boolean**


```python
c=True
type(c)
```




    bool



**String**


```python
type("I am great")
```




    str



**Functions**

We've seen a couple of functions defined already, but let's dig a little bit deeper. Functions in Python are very intuitive. Let's start with an example of a function without parameters:


```python
def hello():
    return "Hello World"

```

The def keyword indicate the definition of a function, followed by a name and a list of arguments (which this function doesn't receive). The return statement is used to break the flow of the function and return a value back to the caller:


```python
result = hello()
result
```




    'Hello World'



If a function doesn't explicitly include a return statement, Python will return None by default:


```python
def empty():
    x = 3
result = empty()
print(result)
```

    None
    

**Operators**

Airthmetic


```python
2+2
```




    4




```python
2*3
```




    6




```python
2/1
```




    2.0




```python

```

Boolean


```python
7>3
```




    True




```python
8>=3
```




    True




```python
True and False
```




    False




```python

```

**Control Flow**

Python supports the most common control flow blocks. Keep in mind they're defined with indentation.

If/else/elif statements


```python
days_subscribed = 28
if days_subscribed >= 30:
    print("Loyal customer")
elif days_subscribed >= 15:
    print("Halfway there")
elif days_subscribed >= 1:
    print("Building confidence")
else:
    print("Too early")
```

    Halfway there
    

**For loops**

For loops in Python are different than other languages, specially those C/Java-inspired languages. In Python, for loops are designed to iterate over collections (we'll see collections later). But keep that in mind.


```python
names = ['Monica', 'Ross', 'Chandler', 'Joey', 'Rachel']
for name in names:
    print(name)
```

    Monica
    Ross
    Chandler
    Joey
    Rachel
    

**While loops**

While loops are seldom used in Python. For loops are the preferred choice 99% of the time. Still, they're available and are useful for some situations:


```python
count = 0
while count < 3:
    print("Counting...")
    count += 1
```

    Counting...
    Counting...
    Counting...
    


```python

```

**Collections**
Python has multiple versatile collection types, each with different features and capabilities. These are the most common collections we'll explore:

Lists
Tuples
Dictionaries
Sets

**Lists**


```python
l = [3, 'Hello World', True]
```


```python
len(l)
```




    3




```python
l[0]
```




    3




```python
l[0:2]
```




    [3, 'Hello World']



**Tuples**

Tuples are very similar to lists, but with a huge difference: they're immutable. That means, once a tuple is created, it can't be further modified:


```python
t = (3, 'Hello World', True)
```


```python
t
```




    (3, 'Hello World', True)




```python
t[::]
```




    (3, 'Hello World', True)




```python
t[-2]
```




    'Hello World'



**Dictionaries**

Dictionaries are map-like collections that store values under a user-defined key. The key must be an immutable object; we usually employ strings for keys. Dictionaries are mutable, and more importantly, unordered.


```python
user = {
    "name": "Mary Smith",
    "email": "mary@example.com",
    "age": 30,
    "subscribed": True
}
user
```




    {'name': 'Mary Smith',
     'email': 'mary@example.com',
     'age': 30,
     'subscribed': True}




```python
user['email']
```




    'mary@example.com'




```python
'last_name' in user
```




    False



**Sets**

Sets are unordered collection which the unique characteristic that they only contain unique elements:


```python
s = {3, 1, 3, 7, 9, 1, 3, 1}
s
```




    {1, 3, 7, 9}



Adding elements is done with the add method:


```python
s.add(10)
s
```




    {3, 7, 9, 10}



Removing elements can be done with pop():


```python
s.pop()
```




    1




```python
s
```




    {3, 7, 9, 10}



**Iterating collections**

As mentioned in the control flow section, Python's for loop is specially designed to iterate over collections:


```python
l = [3, 'Hello World', True]
for elem in l:
    print(elem)
for key in user:
    print(key.title(), '=>', user[key])
```

    3
    Hello World
    True
    Name => Mary Smith
    Email => mary@example.com
    Age => 30
    Subscribed => True
    

**Modules**

One of the best features of Python as a language, is its rich builtin library. To use external modules, you must first import them:


```python
import random
random.randint(0, 99)
```




    67



**Exceptions**

Exceptions are raised at runtime when an abnormal situation is produced in your program. Exceptions can also be constructed and raised by your code. Example of an exception:


```python
age = 30
```


```python
if age > 21:
    print("Allowed entrance")
```

    Allowed entrance
    

Exceptions can be handled at runtime with a try/except block:


```python
try:
    if age > 21:
        print("Allowed entrance")
except:
    print("Something went wrong")
```

    Allowed entrance
    

The except portion can receive also be parametrized with the expected exception:


```python
try:
    if age > 21:
        print("Allowed entrance")
except TypeError:
    print("Age is probably of a wrong type")
```

    Allowed entrance
    
