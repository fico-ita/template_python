"""Provide several sample math calculations.

This module allows the user to make mathematical calculations.

The module contains the following functions:

- `add(a, b)` - Returns the sum of two numbers.
- `subtract(a, b)` - Returns the difference of two numbers.
- `multiply(a, b)` - Returns the product of two numbers.
- `divide(a, b)` - Returns the quotient of two numbers.

Examples:
    Examples should be written in `doctest` format, and should illustrate how to use the
    function.

    >>> from fico import calculations
    >>> calculations.add(2, 4)
    6.0
    >>> calculations.multiply(2.0, 4.0)
    8.0
    >>> from fico.calculations import divide
    >>> divide(4.0, 2)
    2.0
"""


def add(a: float | int, b: float | int) -> float:
    """The add function adds two numbers together.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Tip:
        Just a tip!

    Args:
        a: First addend in the addition
        b: Second addend in the addition

    Returns:
        The sum of `a` and `b`

    Examples:
        >>> add(4.0, 2.0)
        6.0
        >>> add(4, 2)
        6.0
    """
    return float(a + b)


def subtract(a: float | int, b: float | int) -> float:
    """Calculate the difference of two numbers.

    Args:
        a: Minuend in the subtraction
        b: Subtrahend in the subtraction

    Returns:
        The difference between of `a` and `b`

    Examples:
        >>> subtract(4.0, 2.0)
        2.0
        >>> subtract(4, 2)
        2.0
    """
    return float(a - b)


def multiply(a: float | int, b: float | int) -> float:
    """Calculate the product of two numbers.

    Args:
        a: Multiplicand in the multiplication
        b: Multiplier in the multiplication

    Returns:
        The product of `a` and `b`

    Examples:
        >>> multiply(4.0, 2.0)
        8.0
        >>> multiply(4, 2)
        8.0
    """
    return float(a * b)


def divide(a: float | int, b: float | int) -> float:
    """Calculate the quotient of two numbers.

    Args:
        a: Dividend in the division
        b: Divisor in the division

    Returns:
        The quotient of `a` and `b`

    Raises:
        ZeroDivisionError: Error when divisor is `0`

    Examples:
        >>> divide(4.0, 2.0)
        2.0
        >>> divide(4, 2)
        2.0
        >>> divide(4, 0)
        Traceback (most recent call last):
        ...
        ZeroDivisionError: division by zero
    """
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return float(a / b)
