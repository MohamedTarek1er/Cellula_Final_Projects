# Common Python Errors and Debugging

## SyntaxError

A SyntaxError occurs when Python cannot parse the source code.

Incorrect:

```python
if x > 5
    print(x)
```

Correct:

```python
if x > 5:
    print(x)
```

## NameError

A NameError can occur when a variable or function name is
referenced before it has been defined, or when its spelling
does not match the intended identifier.

```python
name = "Alice"
print(name)
```

## TypeError

A TypeError often occurs when an operation is incompatible
with the supplied types.

Incorrect:

```python
age = 20
message = "Age: " + age
```

Correct:

```python
age = 20
message = "Age: " + str(age)
```

Alternatively, use an f-string:

```python
message = f"Age: {age}"
```

## IndexError

An IndexError can occur when a sequence is accessed using
an index that is outside its valid range.

```python
items = ["a", "b"]

if items:
    print(items[0])
```

Valid list indices range from zero to len(items) - 1.

## KeyError

A KeyError can occur when a dictionary is accessed with a
key that does not exist.

```python
user = {"name": "Alice"}
name = user.get("name", "Unknown")
```

The dict.get method can provide a default value when a key
is absent.

## Exceptions and Error Handling

Use try and except to handle expected runtime errors.

```python
try:
    number = int("123")
except ValueError:
    print("The input must be an integer.")
```

Catch specific exceptions where possible rather than catching
every exception without a clear recovery strategy.
