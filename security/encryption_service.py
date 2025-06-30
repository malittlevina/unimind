from cryptography.fernet import Fernet
import os

class EncryptionService:
    def __init__(self, key_path='encryption.key'):
        self.key_path = key_path
        self.key = self.load_or_generate_key()

    def load_or_generate_key(self):
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as file:
                return file.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as file:
                file.write(key)
            return key

    def encrypt(self, data: str) -> bytes:
        fernet = Fernet(self.key)
        return fernet.encrypt(data.encode())

    def decrypt(self, token: bytes) -> str:
        fernet = Fernet(self.key)
        return fernet.decrypt(token).decode()