import os
import json
import boto3
from botocore.exceptions import ClientError
import streamlit as st

# Get AWS credentials from Streamlit secrets or environment variables
def get_aws_credentials():
    try:
        # Try to get credentials from Streamlit secrets (for deployed app)
        aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
        aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
    except (KeyError, RuntimeError):
        # Fall back to environment variables (for local development)
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials not found in Streamlit secrets or environment variables")
    
    return aws_access_key_id, aws_secret_access_key

# Initialize S3 client
def get_s3_client():
    aws_access_key_id, aws_secret_access_key = get_aws_credentials()
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

# S3 bucket name
BUCKET_NAME = "familyfinanace-app-data"

# Check if a file exists in S3
def s3_file_exists(s3_path):
    s3_client = get_s3_client()
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise

# Read a file from S3
def read_file_from_s3(username, file_name):
    """
    Read a file from S3

    Args:
        username: The username
        file_name: The file name or path (can include subfolders)

    Returns:
        The file content as string, or None if file doesn't exist
    """
    s3_client = get_s3_client()

    # Handle case where file_name includes a path
    if "/" in file_name and "user_transactions_data" in file_name:
        # This is a full path like "user_transactions_data/discover/file.csv"
        parts = file_name.split("user_transactions_data/", 1)
        if len(parts) > 1:
            s3_path = f"{username}/user_transactions_data/{parts[1]}"
        else:
            s3_path = f"{username}/user_transactions_data/{file_name}"
    else:
        # This is just a file name like "users.json"
        s3_path = f"{username}/user_transactions_data/{file_name}"

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_path)
        content = response['Body'].read().decode('utf-8')
        return content
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found in S3: {s3_path}")
            return None
        else:
            print(f"Error reading from S3: {e}")
            raise

# Write a file to S3
def write_file_to_s3(username, file_name, content):
    s3_client = get_s3_client()
    s3_path = f"{username}/user_transactions_data/{file_name}"
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_path,
        Body=content.encode('utf-8')
    )

# Read JSON from S3
def read_json_from_s3(username, file_name):
    content = read_file_from_s3(username, file_name)
    if content:
        return json.loads(content)
    return None

# Write JSON to S3
def write_json_to_s3(username, file_name, data):
    content = json.dumps(data, indent=2)
    write_file_to_s3(username, file_name, content)

# Create a new user folder structure in S3
def create_user_folder_structure(username):
    s3_client = get_s3_client()
    
    # Create main user directory with an empty placeholder file
    # (S3 doesn't have actual directories, but we can simulate them with keys)
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{username}/.placeholder",
        Body=b""
    )
    
    # Create user_transactions_data directory
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{username}/user_transactions_data/.placeholder",
        Body=b""
    )
    
    # Return True if successful
    return True

# List files in a user's folder
def list_files_in_user_folder(subfolder, username):
    """
    List files in a user's folder on S3

    Args:
        subfolder: The subfolder path (e.g. "user_transactions_data/discover")
        username: The username

    Returns:
        List of filenames in the specified folder
    """
    s3_client = get_s3_client()

    # Handle the case where subfolder already includes user_transactions_data
    if "user_transactions_data" in subfolder:
        # Extract the part after user_transactions_data
        subfolder_parts = subfolder.split("user_transactions_data/")
        if len(subfolder_parts) > 1:
            subfolder_path = subfolder_parts[1]
            prefix = f"{username}/user_transactions_data/{subfolder_path}/"
        else:
            prefix = f"{username}/user_transactions_data/"
    else:
        prefix = f"{username}/{subfolder}/"

    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )

        if 'Contents' not in response:
            print(f"No contents found for prefix: {prefix}")
            return []

        # Extract file names without the full path
        files = []
        for item in response['Contents']:
            key = item['Key']
            file_name = key.split('/')[-1]
            if file_name and file_name != '.placeholder':
                files.append(file_name)

        return files
    except ClientError as e:
        print(f"Error listing files from S3: {e}")
        return []
