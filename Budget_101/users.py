import os
import json
import hashlib
import secrets
import string
from s3_utils import read_json_from_s3, write_json_to_s3, create_user_folder_structure, s3_file_exists

# File name for storing user credentials in S3
USER_DB_FILE = 'users.json'

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
    """Create the users database file in S3 if it doesn't exist."""
    # We'll store the main users database in the root of the bucket
    # under a special 'admin' folder
    admin_username = "admin"

    # Check if users.json exists in the admin folder
    admin_users = read_json_from_s3(admin_username, USER_DB_FILE)

    if not admin_users:
        # Create admin folder structure if it doesn't exist
        create_user_folder_structure(admin_username)

        # Create default admin user
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

        # Save to S3
        write_json_to_s3(admin_username, USER_DB_FILE, users)
        return True

    return False

def get_users():
    """Get all users from the S3 database."""
    try:
        admin_username = "admin"
        users = read_json_from_s3(admin_username, USER_DB_FILE)

        if not users:
            initialize_user_db()
            users = read_json_from_s3(admin_username, USER_DB_FILE)

        return users or {}

    except Exception as e:
        print(f"Error reading user database: {e}")
        return {}

def save_users(users):
    """Save users to the S3 database."""
    try:
        admin_username = "admin"
        write_json_to_s3(admin_username, USER_DB_FILE, users)
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
    """Add a new user to the database and create their folder structure in S3."""
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
        # Create the user's folder structure in S3
        create_user_folder_structure(username)
        return True, "User added successfully"
    else:
        return False, "Error saving user"

# Rest of the functions remain mostly the same
def delete_user(username):
    """Delete a user from the database."""
    # Note: We don't delete the user's S3 data folder for data preservation
    # That would require additional implementation if needed
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
