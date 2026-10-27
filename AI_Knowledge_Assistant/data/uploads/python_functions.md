# Python Functions

## Defining a Function

Use def to define a reusable block of code.

```python
def add(a, b):
    return a + b
```

Parameters receive input values. The return statement sends a
result back to the caller.

## Default Arguments

A parameter can have a default value.

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```

Avoid mutable objects such as lists as default argument values
when you intend each call to have independent state.

## Type Hints

Type hints communicate expected input and output types.

```python
def multiply(a: float, b: float) -> float:
    return a * b
```

Type hints improve readability and tool support but do not
automatically enforce types at runtime.

## Recursion

A recursive function calls itself. It needs a base case to stop.

```python
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")

    if n <= 1:
        return 1

    return n * factorial(n - 1)
```

For large inputs, an iterative solution may avoid Python's
recursion depth limit.
