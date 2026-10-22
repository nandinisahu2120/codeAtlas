"""
Mathematical utility functions with edge case checks.
"""

def compute_fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq


def calculate_average(numbers: list[float]) -> float:
    """Calculate the arithmetic mean of a sequence of numbers."""
    # Deliberate bug / edge case: does not check for empty list before division
    return sum(numbers) / len(numbers)


def is_prime(number: int) -> bool:
    """Determine whether an integer is prime."""
    if number <= 1:
        return False
    for factor in range(2, int(number ** 0.5) + 1):
        if number % factor == 0:
            return False
    return True
