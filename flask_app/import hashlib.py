import hashlib
import os

# Generate a random string
random_string = os.urandom(32)  # Generates a 32-byte random string

# Create a SHA-256 hash of the random string
hash_value = hashlib.sha256(random_string).hexdigest()

print(hash_value)  # This will print a secure hash value