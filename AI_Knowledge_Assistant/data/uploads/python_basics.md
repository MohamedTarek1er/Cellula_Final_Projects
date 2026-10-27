# Python Basics

## Variables and Data Types

Python variables refer to objects. Common built-in types include
int, float, str, bool, list, tuple, dict, and set.

Example:

```python
name = "Alice"
age = 25
scores = [80, 90, 95]
```

Use descriptive variable names and choose data structures
appropriate to the problem.

## Conditional Statements

Conditional statements select which block of code executes.

```python
age = 20

if age >= 18:
    print("Adult")
else:
    print("Minor")
```

Python uses indentation to define code blocks. The condition
after if must evaluate to a truthy or falsy value.

## Loops

A for loop iterates over an iterable, while a while loop
continues as long as its condition is true.

```python
for number in range(5):
    print(number)
```

```python
count = 0

while count < 5:
    print(count)
    count += 1
```

A loop should have a clear termination condition when required.
An infinite while loop can occur when its condition never becomes false.

## Lists and Dictionaries

Lists store ordered sequences. Dictionaries map keys to values.

```python
numbers = [3, 1, 2]
numbers.sort()

user = {"name": "Alice", "age": 25}
print(user["name"])
```

Use a dictionary when values need to be accessed by meaningful keys.
