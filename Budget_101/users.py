import os
import json
import hashlib
import secrets
import string

# File path for storing user credentials
USER_DB_FILE = os.path.join('data', 'users.json')

def generate_salt(length=16):
    """Generate a random salt for password hashing."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def hash_password(password, salt=None):
    """Hash a password with salt using SHA-256."""
    if salt is None:
        salt = generate_salt()

    # Combine password and salt, then hash
    hash_obj = hashlib.sha256((password + salt).encode())
    password_hash = hash_obj.hexdigest()

    return password_hash, salt

def initialize_user_db():
    """Create the users database file if it doesn't exist."""
    # Make sure the data directory exists
    os.makedirs(os.path.dirname(USER_DB_FILE), exist_ok=True)

    # If file doesn't exist, create it with an admin user
    if not os.path.exists(USER_DB_FILE):
        admin_salt = generate_salt()
        admin_password_hash, _ = hash_password("admin", admin_salt)

        users = {
            "admin": {
                "password_hash": admin_password_hash,
                "salt": admin_salt,
                "is_admin": True,
                "name": "Administrator"
            }
        }

        with open(USER_DB_FILE, 'w') as f:
            json.dump(users, f, indent=2)

        return True

    return False

def get_users():
    """Get all users from the database."""
    try:
        if os.path.exists(USER_DB_FILE):
            with open(USER_DB_FILE, 'r') as f:
                return json.load(f)
        else:
            initialize_user_db()
            with open(USER_DB_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading user database: {e}")
        return {}

def save_users(users):
    """Save users to the database."""
    try:
        with open(USER_DB_FILE, 'w') as f:
            json.dump(users, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving user database: {e}")
        return False

def authenticate_user(username, password):
    """Authenticate a user by username and password."""
    users = get_users()

    if username in users:
        stored_hash = users[username]["password_hash"]
        salt = users[username]["salt"]

        # Hash the provided password with the stored salt
        input_hash, _ = hash_password(password, salt)

        # Compare hashes
        if input_hash == stored_hash:
            return True

    return False

def add_user(username, password, name, is_admin=False):
    """Add a new user to the database."""
    users = get_users()

    # Check if username already exists
    if username in users:
        return False, "Username already exists"

    # Hash the password
    salt = generate_salt()
    password_hash, _ = hash_password(password, salt)

    # Add the new user
    users[username] = {
        "password_hash": password_hash,
        "salt": salt,
        "is_admin": is_admin,
        "name": name
    }

    # Save the updated users
    if save_users(users):
        return True, "User added successfully"
    else:
        return False, "Error saving user"

def delete_user(username):
    """Delete a user from the database."""
    users = get_users()

    # Check if user exists
    if username not in users:
        return False, "User does not exist"

    # Delete the user
    del users[username]

    # Save the updated users
    if save_users(users):
        return True, "User deleted successfully"
    else:
        return False, "Error deleting user"

def change_password(username, new_password):
    """Change a user's password."""
    users = get_users()

    # Check if user exists
    if username not in users:
        return False, "User does not exist"

    # Hash the new password
    salt = generate_salt()
    password_hash, _ = hash_password(new_password, salt)

    # Update the user's password
    users[username]["password_hash"] = password_hash
    users[username]["salt"] = salt

    # Save the updated users
    if save_users(users):
        return True, "Password changed successfully"
    else:
        return False, "Error changing password"

def is_admin(username):
    """Check if a user is an admin."""
    users = get_users()

    if username in users:
        return users[username].get("is_admin", False)

    return False

# Initialize the user database on import
initialize_user_db()
