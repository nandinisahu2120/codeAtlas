"""
Authentication and session management module.
Handles user credentials, password hashing, and JWT-style tokens.
"""
import hashlib
import time
import secrets


class AuthService:
    def __init__(self, token_expiry_seconds: int = 3600):
        self.token_expiry = token_expiry_seconds
        self.active_sessions: dict[str, dict] = {}
        self.users_db: dict[str, str] = {}

    def register_user(self, username: str, password_hash: str) -> bool:
        if username in self.users_db:
            return False
        self.users_db[username] = password_hash
        return True

    def authenticate(self, username: str, password_hash: str) -> str | None:
        if username not in self.users_db:
            return None
        if self.users_db[username] != password_hash:
            return None

        token = secrets.token_hex(16)
        self.active_sessions[token] = {
            "username": username,
            "expires_at": time.time() + self.token_expiry,
        }
        return token

    def validate_session(self, token: str) -> bool:
        session = self.active_sessions.get(token)
        if not session:
            return False
        if time.time() > session["expires_at"]:
            del self.active_sessions[token]
            return False
        return True

    def revoke_session(self, token: str) -> None:
        if token in self.active_sessions:
            del self.active_sessions[token]
