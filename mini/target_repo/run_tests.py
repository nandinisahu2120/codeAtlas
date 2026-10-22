"""
Test runner script for target_repo.
"""
from math_utils import compute_fibonacci, is_prime
from auth_service import AuthService

def main():
    print("Testing math_utils...")
    fib = compute_fibonacci(7)
    print("Fibonacci(7):", fib)
    assert fib == [0, 1, 1, 2, 3, 5, 8]

    assert is_prime(13) is True
    assert is_prime(14) is False
    print("Prime checks passed.")

    print("Testing auth_service...")
    auth = AuthService()
    reg = auth.register_user("admin", "secret_hash_123")
    assert reg is True
    token = auth.authenticate("admin", "secret_hash_123")
    assert token is not None
    assert auth.validate_session(token) is True
    print("Authentication checks passed.")

    print("All tests completed successfully.")

if __name__ == "__main__":
    main()
