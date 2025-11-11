from collections import defaultdict
from itertools import combinations

import streamlit as st

# Import login functionality
from login import login_page
# Import S3 utilities
from s3_utils import read_file_from_s3, list_files_in_user_folder

# Check if user wants to see the main app
if "show_main_app" not in st.session_state:
    st.session_state.show_main_app = False

# Check if user is authenticated
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# If not authenticated or not continuing to main app, show login page
if not st.session_state.authenticated or not st.session_state.show_main_app:
    login_page()
    st.stop()  # Stop execution here if showing login page

# Continue with main app if authenticated
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob
import plotly.express as px
import plotly.graph_objects as go
import calendar
from PIL import Image
import numpy as np
import re
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Family Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to load data from S3
def load_csv_from_s3(username, subfolder, filename):
    """Load a CSV file from S3 and return as pandas DataFrame."""
    file_path = f"{subfolder}/{filename}"
    file_content = read_file_from_s3(username, file_path)

    if file_content is None:
        return None

    # Convert string content to DataFrame using StringIO
    return pd.read_csv(StringIO(file_content))

# Utility functions for data processing
def load_discover_data(username, file_name):
    """Load and process Discover card transaction data from S3."""
    df = load_csv_from_s3(username, "user_transactions_data/discover", file_name)
    if df is None:
        return pd.DataFrame()

    # Convert date columns to datetime
    df['Trans. Date'] = pd.to_datetime(df['Trans. Date'], format='%m/%d/%Y')
    # Ensure amount is treated correctly (positive for expenses, negative for payments/credits)
    df['Amount'] = pd.to_numeric(df['Amount'])

    # Flag returns - for Discover, returns are typically indicated by negative amounts
    # Also check for cases where a merchant might have a matching debit and credit pair
    df['Is_Return'] = False

    # Add a 'Bank' column
    df['Bank'] = 'Discover'
    # Standardize column names
    df = df.rename(columns={
        'Trans. Date': 'Date',
        'Post Date': 'Posted Date',
        'Amount': 'Amount',
        'Category': 'Category'
    })
    return df

def load_bilt_data(username, file_name):
    """Load and process Bilt transaction data from S3.
    Note: For Bilt, credit and debit are reversed from the standard convention.
    Negative values in the 'Debit/Credit' column represent expenses (debits).
    Positive values in the 'Debit/Credit' column represent payments/credits."""
    df = load_csv_from_s3(username, "user_transactions_data/bilt", file_name)
    if df is None:
        return pd.DataFrame()

    # Convert date column to datetime
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='%m/%d/%Y')

    # Convert Debit/Credit to numeric
    df['Debit/Credit'] = pd.to_numeric(df['Debit/Credit'], errors='coerce')

    # For Bilt: Negative values are expenses (debits), positive values are payments (credits)
    # This is the opposite of our standard convention, so we'll multiply by -1
    df['Amount'] = df['Debit/Credit'] * -1

    # Flag returns - for Bilt, returns would typically be indicated by positive values
    df['Is_Return'] = df['Debit/Credit'] > 0

    # Add a 'Bank' column
    df['Bank'] = 'Bilt'

    # Standardize column names
    df = df.rename(columns={
        'Transaction Date': 'Date',
        'Description': 'Description',
    })

    # If we need a Category column but it doesn't exist
    if 'Category' not in df.columns:
        df['Category'] = None

    return df

def load_capital_one_data(username, file_name):
    """Load and process Capital One transaction data from S3."""
    df = load_csv_from_s3(username, "user_transactions_data/Venture_X", file_name)
    if df is None:
        return pd.DataFrame()

    # Print columns for debugging
    print(f"Capital One file columns: {list(df.columns)}")

    # Check if dataframe has content
    if df.empty:
        print(f"CSV file is empty: {file_name}")
        return pd.DataFrame()

    try:
        # Common variations of date column names
        date_col_options = ['Transaction Date', 'Trans Date', 'Date', 'Posted Date', 'Transaction_Date']
        date_col = None

        # Find the first matching date column
        for col in date_col_options:
            if col in df.columns:
                date_col = col
                break

        # If no date column found, create a default one with today's date
        if date_col is None:
            print(f"No date column found in {file_name}. Using today's date.")
            df['Date'] = pd.to_datetime('today')
        else:
            # Convert date column to datetime
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')

        # Common variations of debit/credit column names
        debit_col_options = ['Debit', 'Debit Amount', 'Amount']
        credit_col_options = ['Credit', 'Credit Amount', 'Payment']

        # Find matching debit column
        debit_col = None
        for col in debit_col_options:
            if col in df.columns:
                debit_col = col
                break

        # Find matching credit column
        credit_col = None
        for col in credit_col_options:
            if col in df.columns:
                credit_col = col
                break

        # If we have both debit and credit columns
        if debit_col and credit_col:
            df['Debit'] = pd.to_numeric(df[debit_col], errors='coerce').fillna(0)
            df['Credit'] = pd.to_numeric(df[credit_col], errors='coerce').fillna(0)
            df['Is_Return'] = (df['Debit'] > 0) & (df['Credit'] > 0) & (df['Debit'] == df['Credit'])
            df['Amount'] = df['Debit'] - df['Credit']
        # If we only have one amount column
        elif debit_col:
            df['Amount'] = pd.to_numeric(df[debit_col], errors='coerce').fillna(0)
            df['Is_Return'] = False
        elif credit_col:
            df['Amount'] = pd.to_numeric(df[credit_col], errors='coerce').fillna(0)
            df['Is_Return'] = False
        else:
            # If no amount column found, look for a column with numeric values
            amount_col = None
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    amount_col = col
                    break

            if amount_col:
                df['Amount'] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
            else:
                print(f"No amount column found in {file_name}. Using zeros.")
                df['Amount'] = 0

            df['Is_Return'] = False

        # Check for description column
        desc_col_options = ['Description', 'Desc', 'Transaction Description', 'Merchant', 'Merchant Name']
        desc_col = None

        for col in desc_col_options:
            if col in df.columns:
                desc_col = col
                break

        if desc_col:
            df['Description'] = df[desc_col]
        else:
            df['Description'] = f"Transaction from {file_name}"

        # Check for category column
        cat_col_options = ['Category', 'Transaction Category', 'Type']
        cat_col = None

        for col in cat_col_options:
            if col in df.columns:
                cat_col = col
                break

        if cat_col:
            df['Category'] = df[cat_col]
        else:
            df['Category'] = 'Uncategorized'

        # Add bank column
        df['Bank'] = 'Capital One'

        # Return only the columns we need
        required_cols = ['Date', 'Description', 'Category', 'Amount', 'Bank', 'Is_Return']
        missing_cols = set(required_cols) - set(df.columns)

        # Add any missing columns
        for col in missing_cols:
            if col == 'Is_Return':
                df[col] = False
            else:
                df[col] = None

        return df[required_cols]

    except Exception as e:
        print(f"Error processing Capital One data from {file_name}: {str(e)}")
        # Return empty DataFrame with the expected columns
        return pd.DataFrame(columns=['Date', 'Description', 'Category', 'Amount', 'Bank', 'Is_Return'])

def load_saver_one_data(username, file_name):
    """Load and process Saver One transaction data from S3."""
    df = load_csv_from_s3(username, "user_transactions_data/Saver_one", file_name)
    if df is None:
        return pd.DataFrame()

    # Convert date columns to datetime
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    # Calculate amount (credit is negative, debit is positive for consistency)
    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)

    # Handle returns: if credit exists, mark as return/payment
    df['Is_Return'] = df['Credit'] > 0

    # For regular transactions: debit is positive, credit is negative (for consistency)
    df['Amount'] = df['Debit'] - df['Credit']

    # Add a 'Bank' column
    df['Bank'] = 'Saver One'
    # Standardize column names
    df = df.rename(columns={
        'Transaction Date': 'Date',
        'Category': 'Category'
    })

    return df

def load_all_transaction_data():
    """Load all transaction data from different banks, and find repeated dates by Bank."""
    # Dictionary to hold all dataframes per bank
    bank_data = defaultdict(list)
    all_data = []

    # Process Discover data
    discover_files, _ = list_files_in_user_folder("user_transactions_data/discover", st.session_state.username)
    for file in discover_files:
        df = load_discover_data(st.session_state.username, file)
        bank_data['Discover'].append(df)
        all_data.append(df)

    # Process Capital One data
    capital_one_files, _ = list_files_in_user_folder("user_transactions_data/Venture_X", st.session_state.username)
    for file in capital_one_files:
        df = load_capital_one_data(st.session_state.username, file)
        bank_data['Capital One'].append(df)
        all_data.append(df)

    # Process Saver One data
    saver_one_files, _ = list_files_in_user_folder("user_transactions_data/Saver_one", st.session_state.username)
    for file in saver_one_files:
        df = load_saver_one_data(st.session_state.username, file)
        bank_data['Saver One'].append(df)
        all_data.append(df)

    # Process Bilt data
    bilt_files, _ = list_files_in_user_folder("user_transactions_data/bilt", st.session_state.username)
    for file in bilt_files:
        df = load_bilt_data(st.session_state.username, file)
        bank_data['Bilt'].append(df)
        all_data.append(df)

    # Find repeating dates for each bank
    repeated_dates_by_bank = {}

    for bank, dfs in bank_data.items():
        # Gather unique dates from each file
        file_dates = [set(df['Date'].unique()) for df in dfs]

        # Find dates that appear in more than one file
        repeated_dates = set()
        # Compare all combinations of files within this bank
        for combo in combinations(range(len(file_dates)), 2):
            i, j = combo
            # Find intersection
            common = file_dates[i].intersection(file_dates[j])
            repeated_dates.update(common)
        # Store as list
        repeated_dates_by_bank[bank] = list(repeated_dates)


    # Combine all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        columns_to_keep = ['Date', 'Description', 'Category', 'Amount', 'Bank', 'Is_Return']
        combined_data = combined_data[columns_to_keep]
        combined_data = identify_related_refunds(combined_data)
        combined_data = combined_data.sort_values('Date')

        # Step 1: Identify duplicate rows only for repeating dates
        mask = combined_data.apply(
            lambda row: row['Date'] in repeated_dates_by_bank.get(row['Bank'], []), axis=1
        )

        # Step 2: Separate those rows and remove duplicates
        repeated_date_rows = combined_data[mask]
        unique_repeated_date_rows = repeated_date_rows.drop_duplicates()

        # Step 3: Get non-repeated date rows
        non_repeated_date_rows = combined_data[~mask]

        # Step 4: Concatenate final unique dataframe
        final_unique_data = pd.concat([unique_repeated_date_rows, non_repeated_date_rows], ignore_index=True)

        return final_unique_data
    return pd.DataFrame()


def identify_related_refunds(df):
    """
    Identify transactions that are likely refunds or have associated credits,
    like airline tickets that might be refunded or credited separately.
    """
    # Create a working copy
    processed_df = df.copy()

    # Flag known patterns for airline credits and other special cases
    # Southwest Airlines pattern
    southwest_mask = processed_df['Description'].str.contains('SOUTHWES', case=False, na=False)
    if southwest_mask.any():
        # Mark the Southwest transactions with the exact amount of $301.50
        southwest_301_mask = southwest_mask & (processed_df['Amount'] == 301.50)
        if southwest_301_mask.any():
            processed_df.loc[southwest_301_mask, 'Is_Return'] = True

    # Look for other common airline patterns where we know there are credits
    airline_patterns = ['AIRLINE', 'AIRWAYS', 'AIR TICKET', 'FLIGHT']
    for pattern in airline_patterns:
        pattern_mask = processed_df['Description'].str.contains(pattern, case=False, na=False)

        # If we find matches, check for transactions with the same amount
        if pattern_mask.any():
            # Group by amount and find amounts that appear multiple times (potential refund pairs)
            amount_counts = processed_df[pattern_mask].groupby('Amount').size()
            duplicate_amounts = amount_counts[amount_counts > 1].index.tolist()

            # Mark one of each duplicate amount as a return/credit
            for amount in duplicate_amounts:
                matches = processed_df[(pattern_mask) & (processed_df['Amount'] == amount)]
                if len(matches) >= 2:
                    # Find latest transaction with this amount and mark it
                    latest_index = matches.index[-1]
                    processed_df.loc[latest_index, 'Is_Return'] = True

    # Additional logic for general credit detection
    # Look for descriptions containing common refund/credit terms
    refund_terms = ['REFUND', 'CREDIT', 'RETURN', 'REVERSAL']
    for term in refund_terms:
        processed_df.loc[processed_df['Description'].str.contains(term, case=False, na=False), 'Is_Return'] = True

    return processed_df

def categorize_transactions(df):
    """Group similar categories together for better analysis."""
    # Create a copy to avoid modifying the original
    categorized_df = df.copy()

    # Define category mappings to standardize across banks
    category_map = {
        # Food related
        'Restaurants': 'Food & Dining',
        'Dining': 'Food & Dining',
        'Supermarkets': 'Groceries',
        'Grocery': 'Groceries',

        # Shopping
        'Merchandise': 'Shopping',
        'Shopping': 'Shopping',

        # Utilities and services
        'Utilities': 'Utilities',
        'Services': 'Services',
        'Phone/Cable': 'Utilities',

        # Transportation
        'Gas/Automotive': 'Transportation',
        'Automotive': 'Transportation',
        'Gas': 'Transportation',

        # Housing
        'Lodging': 'Housing',
        'Rent': 'Housing',

        # Insurance
        'Insurance': 'Insurance',

        # Payments and credits (typically not expenses)
        'Payments and Credits': 'Payments',
        'Payment/Credit': 'Payments',
        'Awards and Rebate Credits': 'Payments'
    }

    # Apply the category mapping
    categorized_df['Category Group'] = categorized_df['Category'].map(category_map)

    # Handle unmapped categories
    categorized_df['Category Group'] = categorized_df['Category Group'].fillna('Other')

    return categorized_df

def filter_expenses_only(df):
    """Filter to include only expenses (positive amounts) and exclude returns/credits."""
    # Filter out returns and exclude transactions with zero or negative amounts
    return df[(df['Amount'] > 0) & (~df['Is_Return'])].copy()

def load_vendor_rules():
    """Load vendor pattern rules for automatic categorization from the shared core_app_data folder."""
    try:
        # Since vendor_rules.csv is shared across all users, we'll use a special function to read it
        # or fall back to local file if needed

        # Try to load from S3 first (using a dummy username as this is a shared file)
        file_content = read_file_from_s3("shared", "core_app_data/vendor_rules.csv")

        if file_content is not None:
            # Convert string content to DataFrame using StringIO
            from io import StringIO
            return pd.read_csv(StringIO(file_content))

        # Fallback to local file if S3 retrieval fails
        local_rules_file = os.path.join('core_app_data', 'vendor_rules.csv')
        if os.path.exists(local_rules_file):
            return pd.read_csv(local_rules_file)

    except Exception as e:
        print(f"Error loading vendor rules: {str(e)}")

    # Return empty DataFrame with expected columns if file doesn't exist or error occurs
    return pd.DataFrame(columns=['vendor_pattern', 'category', 'subcategory'])

def load_category_mapping():
    """Load manual category mappings from Spend_categories.csv file."""
    categories_file = os.path.join('user_transactions_data', 'Spend_categories.csv')
    if os.path.exists(categories_file):
        return pd.read_csv(categories_file)
    return pd.DataFrame(columns=['Description', 'Category', 'Changed_Sub-Category', 'Changed_Category'])

def apply_vendor_rules(df):
    """Apply vendor pattern rules to categorize transactions."""
    # Create copy to avoid modifying original
    categorized_df = df.copy()

    # Initialize new columns if they don't exist
    if 'New_Category' not in categorized_df.columns:
        categorized_df['New_Category'] = None
    if 'Sub_Category' not in categorized_df.columns:
        categorized_df['Sub_Category'] = None

    # Load vendor rules
    vendor_rules = load_vendor_rules()

    # If rules exist, apply them
    if not vendor_rules.empty:
        # For each rule, check if description contains the pattern
        for _, rule in vendor_rules.iterrows():
            pattern = rule['vendor_pattern']
            # Apply rule if pattern is found in description
            mask = categorized_df['Description'].str.contains(pattern, case=False, na=False)
            # Only update if not already set by a more specific rule
            categorized_df.loc[mask & categorized_df['New_Category'].isna(), 'New_Category'] = rule['category']
            categorized_df.loc[mask & categorized_df['Sub_Category'].isna(), 'Sub_Category'] = rule['subcategory']

    return categorized_df

def apply_manual_categories(df):
    """Apply manual category mappings from Spend_categories.csv."""
    categorized_df = df.copy()

    # Load manual categories
    manual_categories = load_category_mapping()

    # If manual mappings exist, apply them
    if not manual_categories.empty:
        # Create a mapping dictionary for exact description matches
        category_map = dict(zip(manual_categories['Description'],
                               zip(manual_categories['Changed_Category'],
                                   manual_categories['Changed_Sub-Category'])))

        # Apply exact matches
        for desc, (category, subcategory) in category_map.items():
            mask = categorized_df['Description'] == desc
            categorized_df.loc[mask, 'New_Category'] = category
            categorized_df.loc[mask, 'Sub_Category'] = subcategory

    return categorized_df

def enhanced_categorize_transactions(df):
    """Enhanced categorization function with vendor rules and subcategories."""
    # First apply existing categorization for backward compatibility
    categorized_df = categorize_transactions(df)

    # Apply vendor pattern rules
    categorized_df = apply_vendor_rules(categorized_df)

    # Apply manual categories (these will override pattern-based rules)
    categorized_df = apply_manual_categories(categorized_df)

    # Use New_Category as the main category if it exists, otherwise fall back to Category Group
    categorized_df['Enhanced_Category'] = categorized_df['New_Category'].fillna(categorized_df['Category Group'])

    # Ensure Sub_Category is populated for display
    categorized_df['Sub_Category'] = categorized_df['Sub_Category'].fillna('General')

    return categorized_df

# Function to load income data
def load_income_data():
    # Get username from session state
    username = st.session_state.username
    income_file_path = 'income_data.csv'

    # Try to load from S3 using the username
    try:
        file_content = read_file_from_s3(username, income_file_path)
        if file_content is not None:
            # Convert string content to DataFrame using StringIO
            from io import StringIO
            df = pd.read_csv(StringIO(file_content))

            # Convert date columns to datetime if they exist
            if 'start_month' in df.columns and not df['start_month'].empty:
                df['start_month'] = pd.to_datetime(df['start_month'], errors='coerce').dt.date
            if 'end_month' in df.columns and not df['end_month'].empty:
                df['end_month'] = pd.to_datetime(df['end_month'], errors='coerce').dt.date

            # Add a unique ID if not present
            if 'id' not in df.columns:
                df['id'] = [f"income_{i}" for i in range(len(df))]
                save_income_data(df)
            return df
    except Exception as e:
        print(f"Error loading income data from S3: {str(e)}")

    # If we reach here, either the file doesn't exist or there was an error
    return pd.DataFrame(columns=['person', 'income_source', 'amount', 'frequency', 'start_month', 'end_month', 'id'])

# Function to load recurring payment data
def load_recurring_payments():
    # Get username from session state
    username = st.session_state.username
    payments_file_path = 'recurring_payments.csv'

    # Try to load from S3 using the username
    try:
        file_content = read_file_from_s3(username, payments_file_path)
        if file_content is not None:
            # Convert string content to DataFrame using StringIO
            from io import StringIO
            df = pd.read_csv(StringIO(file_content))

            # Convert date columns to datetime if they exist
            if 'start_month' in df.columns and not df.empty and not df['start_month'].empty:
                df['start_month'] = pd.to_datetime(df['start_month'], errors='coerce').dt.date
            if 'end_month' in df.columns and not df.empty and not df['end_month'].empty:
                df['end_month'] = pd.to_datetime(df['end_month'], errors='coerce').dt.date

            # Add a unique ID if not present
            if 'id' not in df.columns:
                df['id'] = [f"payment_{i}" for i in range(len(df))]
                save_recurring_payments(df)
            return df
    except Exception as e:
        print(f"Error loading recurring payments data from S3: {str(e)}")

    # If we reach here, either the file doesn't exist or there was an error
    return pd.DataFrame(columns=['description', 'amount', 'frequency', 'category', 'start_month', 'end_month', 'sub_category', 'id'])


# Function to save income data
def save_income_data(df):
    # Get username from session state
    username = st.session_state.username
    income_file_path = 'income_data.csv'

    # Create a copy of the dataframe to avoid modifying the original
    save_df = df.copy()

    # Convert date columns to string format without time component
    if 'start_month' in save_df.columns:
        save_df['start_month'] = save_df['start_month'].astype(str).str.split(' ').str[0]
    if 'end_month' in save_df.columns:
        save_df['end_month'] = save_df['end_month'].astype(str).str.split(' ').str[0]
        # Replace 'NaT' with empty string
        save_df['end_month'] = save_df['end_month'].replace('NaT', '')

    try:
        # Convert dataframe to CSV string
        csv_content = save_df.to_csv(index=False)

        # Save to S3
        from s3_utils import write_file_to_s3
        write_file_to_s3(username, income_file_path, csv_content)
    except Exception as e:
        print(f"Error saving income data to S3: {str(e)}")
        # Fallback to local save if S3 fails
        income_file = os.path.join('user_transactions_data', 'income_data.csv')
        save_df.to_csv(income_file, index=False)

# Function to save recurring payment data
def save_recurring_payments(df):
    # Get username from session state
    username = st.session_state.username
    payments_file_path = 'recurring_payments.csv'

    # Create a copy of the dataframe to avoid modifying the original
    save_df = df.copy()

    # Make sure the sub_category column exists
    if 'sub_category' not in save_df.columns and 'Sub_Category' in save_df.columns:
        # Rename from Sub_Category to sub_category for consistency in storage
        save_df['sub_category'] = save_df['Sub_Category']
        save_df = save_df.drop('Sub_Category', axis=1)
    elif 'sub_category' not in save_df.columns:
        # Create the column if it doesn't exist
        save_df['sub_category'] = 'General'

    # Convert date columns to string format without time component if they exist
    if 'start_month' in save_df.columns:
        save_df['start_month'] = save_df['start_month'].astype(str).str.split(' ').str[0]
    if 'end_month' in save_df.columns:
        save_df['end_month'] = save_df['end_month'].astype(str).str.split(' ').str[0]
        # Replace 'NaT' with empty string
        save_df['end_month'] = save_df['end_month'].replace('NaT', '')

    try:
        # Convert dataframe to CSV string
        csv_content = save_df.to_csv(index=False)

        # Save to S3
        from s3_utils import write_file_to_s3
        write_file_to_s3(username, payments_file_path, csv_content)
    except Exception as e:
        print(f"Error saving recurring payments data to S3: {str(e)}")
        # Fallback to local save if S3 fails
        payments_file = os.path.join('user_transactions_data', 'recurring_payments.csv')
        save_df.to_csv(payments_file, index=False)

# Calculate monthly income based on frequency
def calculate_monthly_income(amount, frequency):
    if frequency == 'Monthly':
        return amount
    elif frequency == 'Bi-weekly':
        return amount * 26 / 12
    elif frequency == 'Weekly':
        return amount * 52 / 12
    elif frequency == 'Yearly':
        return amount / 12
    else:
        return amount  # Default to monthly

# Calculate monthly income based on frequency and date range
def calculate_active_monthly_income(income_data, reference_date=None):
    """
    Calculate monthly income considering start and end dates.
    Only includes income sources that are active at the reference_date.

    Args:
        income_data: DataFrame containing income sources
        reference_date: Date to check if income is active (defaults to current date)

    Returns:
        Total monthly income from active sources
    """
    if income_data.empty:
        return 0.0

    # Default to current date if no reference date is provided
    if reference_date is None:
        reference_date = pd.Timestamp(datetime.now().date())
    # Handle Python's datetime.date type
    elif hasattr(reference_date, 'year') and hasattr(reference_date, 'month') and hasattr(reference_date, 'day'):
        # Convert Python date to pandas Timestamp for consistent comparison
        reference_date = pd.Timestamp(reference_date)
    elif isinstance(reference_date, str):
        reference_date = pd.Timestamp(reference_date)

    # Create a copy to avoid modifying the original
    data = income_data.copy()

    # Convert date columns to pandas Timestamp if they aren't already
    if 'start_month' in data.columns:
        data['start_month'] = pd.to_datetime(data['start_month'])

    if 'end_month' in data.columns:
        data['end_month'] = pd.to_datetime(data['end_month'])

    # Filter active income sources based on start_month and end_month
    # For each row, check if:
    # 1. The start_month is before or equal to the reference date
    # 2. The end_month is either NaN/None (meaning no end date) or after the reference date
    active_income = data[
        ((pd.isna(data['start_month'])) | (data['start_month'] <= reference_date)) &
        ((pd.isna(data['end_month'])) | (data['end_month'] >= reference_date))
    ]

    # Calculate monthly income based on frequency
    if not active_income.empty:
        monthly_income = active_income.apply(
            lambda row: calculate_monthly_income(row['amount'], row['frequency']), axis=1
        ).sum()
        return monthly_income

    return 0.0

# Calculate monthly payment based on frequency
def calculate_monthly_payment(amount, frequency):
    if frequency == 'Monthly':
        return amount
    elif frequency == 'Bi-weekly':
        return amount * 26 / 12
    elif frequency == 'Weekly':
        return amount * 52 / 12
    elif frequency == 'Yearly':
        return amount / 12
    elif frequency == 'Quarterly':
        return amount * 4 / 12
    else:
        return amount  # Default to monthly

# Calculate monthly payments considering start and end dates
def calculate_active_monthly_payments(payment_data, reference_date=None):
    """
    Calculate monthly payments considering start and end dates.
    Only includes payment sources that are active at the reference_date.

    Args:
        payment_data: DataFrame containing recurring payments
        reference_date: Date to check if payment is active (defaults to current date)

    Returns:
        Total monthly payment from active sources
    """
    if payment_data.empty:
        return 0.0

    # Default to current date if no reference date is provided
    if reference_date is None:
        reference_date = pd.Timestamp(datetime.now().date())
    # Handle Python's datetime.date type
    elif hasattr(reference_date, 'year') and hasattr(reference_date, 'month') and hasattr(reference_date, 'day'):
        # Convert Python date to pandas Timestamp for consistent comparison
        reference_date = pd.Timestamp(reference_date)
    elif isinstance(reference_date, str):
        reference_date = pd.Timestamp(reference_date)

    # Create a copy to avoid modifying the original
    data = payment_data.copy()

    # If start_month and end_month columns exist, convert to pandas Timestamp
    if 'start_month' in data.columns:
        data['start_month'] = pd.to_datetime(data['start_month'])
    else:
        # If columns don't exist, assume all payments are active (no date restrictions)
        return data.apply(
            lambda row: calculate_monthly_payment(row['amount'], row['frequency']), axis=1
        ).sum()

    if 'end_month' in data.columns:
        data['end_month'] = pd.to_datetime(data['end_month'])

    # Filter active payments based on start_month and end_month
    active_payments = data[
        ((pd.isna(data['start_month'])) | (data['start_month'] <= reference_date)) &
        ((pd.isna(data['end_month'])) | (data['end_month'] >= reference_date))
    ]

    # Calculate monthly payment based on frequency
    if not active_payments.empty:
        monthly_payment = active_payments.apply(
            lambda row: calculate_monthly_payment(row['amount'], row['frequency']), axis=1
        ).sum()
        return monthly_payment

    return 0.0

# Function to load monthly budget data (similar structure to recurring payments)
def load_monthly_budget():
    username = st.session_state.username
    budget_file_path = 'monthly_budget.csv'
    try:
        file_content = read_file_from_s3(username, budget_file_path)
        if file_content is not None:
            from io import StringIO
            df = pd.read_csv(StringIO(file_content))
            # Ensure required columns exist
            required_cols = ['description','amount','frequency','category','sub_category','start_month','end_month','payment_type','id']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'payment_type':
                        df[col] = 'Fixed'
                    elif col == 'id':
                        df[col] = [f"budget_{i}" for i in range(len(df))]
                    else:
                        df[col] = None
            # Parse dates
            if 'start_month' in df.columns:
                df['start_month'] = pd.to_datetime(df['start_month'], errors='coerce').dt.date
            if 'end_month' in df.columns:
                df['end_month'] = pd.to_datetime(df['end_month'], errors='coerce').dt.date
            # Guarantee ids
            if 'id' in df.columns and df['id'].isna().any():
                df.loc[df['id'].isna(),'id'] = [f"budget_{i}" for i in range(len(df[df['id'].isna()]))]
            return df[required_cols]
    except Exception as e:
        print(f"Error loading monthly budget from S3: {e}")
    return pd.DataFrame(columns=['description','amount','frequency','category','sub_category','start_month','end_month','payment_type','id'])

# Function to save monthly budget data
def save_monthly_budget(df):
    username = st.session_state.username
    budget_file_path = 'monthly_budget.csv'
    save_df = df.copy()
    # Normalize date columns
    for col in ['start_month','end_month']:
        if col in save_df.columns:
            save_df[col] = save_df[col].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notna(x) else '')
    try:
        from s3_utils import write_file_to_s3
        csv_content = save_df.to_csv(index=False)
        write_file_to_s3(username, budget_file_path, csv_content)
    except Exception as e:
        print(f"Error saving monthly budget to S3: {e}")

# Helper to render a labeled currency amount with color based on sign (positive green, negative red)
def render_colored_amount(label: str, amount: float):
    color = 'green' if amount >= 0 else 'red'
    st.markdown(
        f"<div style='font-size:20px;font-weight:600;margin-bottom:4px;'>{label}: "
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='font-size:25px;font-weight:600;margin-bottom:4px;'>"
        f"<span style='color:{color};font-size:35px;'>${amount:,.2f}</span></div>",
        unsafe_allow_html=True
    )

# Application title and description
st.title("💰 Family Expense Tracker")
st.markdown("""
    Track, visualize, and analyze your family expenses across different bank accounts.
    Upload your transaction files in the user_transactions_data directory organized by bank name.
""")

# Load all transaction data first, so it's available for the sidebar
with st.spinner("Loading transaction data..."):
    all_transactions = load_all_transaction_data()

# Process transaction data if available
if not all_transactions.empty:
    enhanced_transactions = enhanced_categorize_transactions(all_transactions)
    expenses_only = filter_expenses_only(enhanced_transactions)

# Sidebar for quick access to income and recurring payments
with st.sidebar:
    st.header("Quick Overview")

    # Load income and payment data for sidebar display
    income_data = load_income_data()
    recurring_payments = load_recurring_payments()
    monthly_budget = load_monthly_budget()

    c1 = st.container(border=True)
    with c1:
        # Calculate total monthly income from active sources only
        if not income_data.empty:
            # Use calculate_active_monthly_income which filters based on current date
            current_date = pd.Timestamp(datetime.now().date())
            total_monthly_income = calculate_active_monthly_income(income_data, current_date)
            st.metric("Total Monthly Income", f"${total_monthly_income:.2f}")

            # Show count of active income sources - fix the comparison issue here
            # Convert dates to proper timestamps for comparison
            income_data_copy = income_data.copy()
            if 'start_month' in income_data_copy.columns:
                income_data_copy['start_month'] = pd.to_datetime(income_data_copy['start_month'])
            if 'end_month' in income_data_copy.columns:
                income_data_copy['end_month'] = pd.to_datetime(income_data_copy['end_month'])

            # Now do the comparison with consistent types
            active_sources = len(income_data_copy[
                ((pd.isna(income_data_copy['start_month'])) | (income_data_copy['start_month'] <= current_date)) &
                ((pd.isna(income_data_copy['end_month'])) | (income_data_copy['end_month'] >= current_date))
            ])
            st.caption(f"Based on {active_sources} active income sources")

        # Calculate total monthly recurring payments
        if not monthly_budget.empty:
            # Use calculate_active_monthly_payments which filters based on current date
            total_monthly_budget = calculate_active_monthly_payments(monthly_budget, current_date)
            st.metric("Total Monthly Budget", f"${total_monthly_budget:.2f}")

            # Show count of active payments
            if 'start_month' in monthly_budget.columns and 'end_month' in monthly_budget.columns:
                payments_copy = monthly_budget.copy()
                payments_copy['start_month'] = pd.to_datetime(payments_copy['start_month'])
                payments_copy['end_month'] = pd.to_datetime(payments_copy['end_month'])

                active_budget_items = len(payments_copy[
                                          ((pd.isna(payments_copy['start_month'])) | (
                                                      payments_copy['start_month'] <= current_date)) &
                                          ((pd.isna(payments_copy['end_month'])) | (
                                                      payments_copy['end_month'] >= current_date))
                                          ])
                st.caption(f"Based on {active_budget_items} active recurring budget items")

        # Calculate total monthly recurring payments
        if not recurring_payments.empty:
            # Use calculate_active_monthly_payments which filters based on current date
            recurring_payments_wo_savings = recurring_payments[recurring_payments["category"] != "Savings"]
            total_monthly_payments = calculate_active_monthly_payments(recurring_payments_wo_savings, current_date)
            st.metric("Total Monthly Fixed Expenses", f"${total_monthly_payments:.2f}")

            # Show count of active payments
            if 'start_month' in recurring_payments_wo_savings.columns and 'end_month' in recurring_payments_wo_savings.columns:
                payments_copy = recurring_payments_wo_savings.copy()
                payments_copy['start_month'] = pd.to_datetime(payments_copy['start_month'])
                payments_copy['end_month'] = pd.to_datetime(payments_copy['end_month'])

                active_payments = len(payments_copy[
                    ((pd.isna(payments_copy['start_month'])) | (payments_copy['start_month'] <= current_date)) &
                    ((pd.isna(payments_copy['end_month'])) | (payments_copy['end_month'] >= current_date))
                ])
                st.caption(f"Based on {active_payments} active recurring payments")

        # Get current month's variable expenses
        current_month_start = pd.Timestamp(datetime.now().year, datetime.now().month, 1)
        if datetime.now().month == 12:
            current_month_end = pd.Timestamp(datetime.now().year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            current_month_end = pd.Timestamp(datetime.now().year, datetime.now().month + 1, 1) - pd.Timedelta(days=1)

        # Initialize with zero in case there's no transaction data
        total_variable_expenses = 0.0

        # Only calculate if we have transaction data loaded
        if 'expenses_only' in locals() and not expenses_only.empty:
            # Filter to current month
            current_month_expenses = expenses_only[
                (expenses_only['Date'] >= current_month_start) &
                (expenses_only['Date'] <= current_month_end)
            ]

            # Sum up the variable expenses for current month
            if not current_month_expenses.empty:
                total_variable_expenses = current_month_expenses['Amount'].sum()

        # Show variable expenses in the sidebar
        st.metric("Current Month Variable Expenses", f"${total_variable_expenses:.2f}")

    c2 = st.container(border=True)
    with c2:
        # Show estimated remaining budget if both income and payments exist
        if not monthly_budget.empty:
            # Calculate actual remaining budget including variable expenses
            remaining_budget = total_monthly_budget - total_monthly_payments - total_variable_expenses

            # Colored display replacing st.metric
            render_colored_amount("Remaining Budget (after all expenses)", remaining_budget)

            # Show percent of income spent
            if total_monthly_budget > 0:
                spending_percent = ((total_monthly_payments + total_variable_expenses) / total_monthly_budget) * 100
                savings_percent = 100 - spending_percent
                st.caption(f"You've spent {spending_percent:.1f}% of your budget this month")

                # Add a simple progress bar
                st.progress(min(spending_percent/100, 1.0), f"Saving {savings_percent:.1f}%")
            # Show estimated remaining budget if both income and payments exist

        if not income_data.empty:
            # Calculate actual remaining income including variable expenses
            remaining_income = total_monthly_income - total_monthly_payments - total_variable_expenses
            render_colored_amount("Remaining Income (after all expenses)", remaining_income)

            # Show percent of income spent
            if total_monthly_income > 0:
                spending_percent = ((total_monthly_payments + total_variable_expenses) / total_monthly_income) * 100
                savings_percent = 100 - spending_percent
                st.caption(f"You've spent {spending_percent:.1f}% of your income this month")

                # Add a simple progress bar
                st.progress(min(spending_percent / 100, 1.0), f"Saving {savings_percent:.1f}%")

    # Add some spacing
    st.write("---")

    # Add logout button at the bottom of the sidebar
    if st.button("Log Out"):
        st.session_state.authenticated = False
        st.session_state.show_main_app = False
        st.session_state.username = None
        if "is_admin" in st.session_state:
            st.session_state.is_admin = False
        st.rerun()  # Rerun the app to return to login page

# Load all transaction data
with st.spinner("Loading transaction data..."):
    all_transactions = load_all_transaction_data()

if all_transactions.empty:
    st.warning("No transaction data found. Please make sure CSV files are available in the user_transactions_data directory.")
else:
    # Use the enhanced categorization system
    enhanced_transactions = enhanced_categorize_transactions(all_transactions)
    expenses_only = filter_expenses_only(enhanced_transactions)

    # Create a tab for category management
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Inputs",
        "Financial Overview",
        "Monthly Expense Summary",
        "Expense by Category",
        "Spending Trends",
        "Category Comparison",
        "App Data"
    ])

    with tab1:
        st.header("Inputs")

        # Income data management
        st.subheader("Manage Income Sources")
        income_data = load_income_data()

        # Display current income sources with edit/delete buttons
        if not income_data.empty:
            st.write("### Current Income Sources")

            # Create columns for the data display and action buttons
            col1, col2 = st.columns([4, 1])

            with col1:
                st.dataframe(income_data, use_container_width=True, hide_index=True)

            with col2:
                st.write("### Actions")
                # Get list of income sources for selection
                income_options = []
                for _, row in income_data.iterrows():
                    income_options.append(f"{row['person']} - {row['income_source']} (${row['amount']}) [{row['id']}]")

                selected_income = st.selectbox("Select to edit/delete", options=income_options, key="income_select")

                # Extract ID from the selected option
                selected_income_id = selected_income.split('[')[-1].strip(']') if selected_income else None

                # Delete button
                if st.button("Delete Source", key="income_delete_button", type="primary"):
                    income_data = income_data[income_data['id'] != selected_income_id]
                    save_income_data(income_data)
                    st.success(f"Deleted income source: {selected_income}")
                    st.rerun()

                # Edit button
                if st.button("Edit Source", key="income_edit_button"):
                    # Find the selected income data to pre-fill the form
                    if selected_income_id:
                        selected_income_data = income_data[income_data['id'] == selected_income_id].iloc[0]

                        # Store the values in session state to pre-fill the form
                        st.session_state['edit_income'] = True
                        st.session_state['edit_income_id'] = selected_income_id
                        st.session_state['edit_person'] = selected_income_data['person']
                        st.session_state['edit_income_source'] = selected_income_data['income_source']
                        st.session_state['edit_amount'] = selected_income_data['amount']
                        st.session_state['edit_frequency'] = selected_income_data['frequency']

                        # Handle start_month and end_month
                        if pd.notna(selected_income_data['start_month']):
                            if isinstance(selected_income_data['start_month'], str):
                                st.session_state['edit_start_month'] = pd.to_datetime(selected_income_data['start_month']).date()
                            else:
                                st.session_state['edit_start_month'] = selected_income_data['start_month']

                        # Check if end_month exists and is not NaN/None
                        has_end_date = pd.notna(selected_income_data['end_month']) and selected_income_data['end_month'] not in ['None', '']
                        st.session_state.end_date_enabled = has_end_date

                        if has_end_date:
                            if isinstance(selected_income_data['end_month'], str):
                                st.session_state['edit_end_month'] = pd.to_datetime(selected_income_data['end_month']).date()
                            else:
                                st.session_state['edit_end_month'] = selected_income_data['end_month']

                        # Force a rerun to update the form with these values
                        st.rerun()

        # Income source form
        st.subheader("Add or Edit Income Source")

        # Initialize session state for the end date checkbox if it doesn't exist
        if 'end_date_enabled' not in st.session_state:
            st.session_state.end_date_enabled = False

        # Initialize edit mode flag if it doesn't exist
        if 'edit_income' not in st.session_state:
            st.session_state.edit_income = False

        # End date checkbox OUTSIDE the form
        end_month_provided = st.checkbox("This income source has an end date",
                                         value=st.session_state.end_date_enabled,
                                         key="end_date_checkbox")

        # Store the checkbox value in session state
        st.session_state.end_date_enabled = end_month_provided

        with st.form("income_form"):
            # Pre-fill form values if in edit mode
            person = st.text_input("Person", value=st.session_state.get('edit_person', ''))
            income_source = st.text_input("Source", value=st.session_state.get('edit_income_source', ''))
            amount = st.number_input("Amount", min_value=0.0, format="%.2f", value=st.session_state.get('edit_amount', 0.0))
            frequency = st.selectbox("Frequency", ["Monthly", "Bi-weekly", "Weekly", "Yearly"],
                                    index=["Monthly", "Bi-weekly", "Weekly", "Yearly"].index(
                                        st.session_state.get('edit_frequency', "Monthly")
                                    ))

            # Add date field for start month
            start_month_default = st.session_state.get('edit_start_month', datetime.now().replace(day=1))
            start_month = st.date_input("Start Month", value=start_month_default)

            # Show the end date input field based on the checkbox outside the form
            end_month = None
            if end_month_provided:
                end_month_default = st.session_state.get('edit_end_month', datetime.now().replace(day=1))
                end_month = st.date_input("End Month", value=end_month_default)

            # Add clear form button
            clear_income_form = st.form_submit_button("Clear Form")
            if clear_income_form:
                st.session_state['edit_person'] = ''
                st.session_state['edit_income_source'] = ''
                st.session_state['edit_amount'] = 0.0
                st.session_state['edit_frequency'] = 'Monthly'
                st.session_state['edit_start_month'] = datetime.now().replace(day=1)
                st.session_state['edit_end_month'] = datetime.now().replace(day=1)
                st.session_state['edit_income'] = False
                st.rerun()

            submit_income = st.form_submit_button("Save Income Source")

            if submit_income and person and amount > 0:
                # If source is not provided, use person name as default
                if not income_source:
                    income_source = person

                # Convert dates to appropriate format
                start_month_str = start_month.strftime('%Y-%m-%d') if start_month else None
                end_month_str = end_month.strftime('%Y-%m-%d') if end_month_provided and end_month else None

                # Check if we're editing or adding new
                if st.session_state.get('edit_income', False):
                    # Update the existing record
                    mask = income_data['id'] == st.session_state['edit_income_id']
                    income_data.loc[mask, 'person'] = person
                    income_data.loc[mask, 'income_source'] = income_source
                    income_data.loc[mask, 'amount'] = amount
                    income_data.loc[mask, 'frequency'] = frequency
                    income_data.loc[mask, 'start_month'] = start_month_str
                    income_data.loc[mask, 'end_month'] = end_month_str

                    st.success(f"Updated income source: {income_source} for {person}")

                    # Reset edit mode
                    st.session_state.edit_income = False
                    st.session_state.pop('edit_income_id', None)
                    st.session_state.pop('edit_person', None)
                    st.session_state.pop('edit_income_source', None)
                    st.session_state.pop('edit_amount', None)
                    st.session_state.pop('edit_frequency', None)
                    st.session_state.pop('edit_start_month', None)
                    st.session_state.pop('edit_end_month', None)
                else:
                    # Create new income source entry with a unique ID
                    new_id = f"income_{len(income_data) + 1}" if not income_data.empty else "income_0"

                    new_income = pd.DataFrame({
                        'person': [person],
                        'income_source': [income_source],
                        'amount': [amount],
                        'frequency': [frequency],
                        'start_month': [start_month_str],
                        'end_month': [end_month_str],
                        'id': [new_id]
                    })

                    # Add as a new record
                    income_data = pd.concat([income_data, new_income], ignore_index=True)
                    st.success(f"Added new income source: {income_source} for {person}")

                # Save updated income data
                save_income_data(income_data)
                st.rerun()  # Refresh the page to show updated data

        # Recurring payments management
        st.subheader("Manage Recurring Payments")
        recurring_payments = load_recurring_payments()

        # Display current recurring payments with delete/edit options
        if not recurring_payments.empty:
            st.write("### Current Recurring Payments")

            # Create columns for the data display and action buttons
            col1, col2 = st.columns([4, 1])

            with col1:
                st.dataframe(recurring_payments, use_container_width=True, hide_index=True)

            with col2:
                st.write("### Actions")
                # Get list of payments for selection
                payment_options = []
                for _, row in recurring_payments.iterrows():
                    payment_options.append(f"{row['description']} (${row['amount']}) [{row['id']}]")

                selected_payment = st.selectbox("Select to edit/delete", options=payment_options, key="payment_select")

                # Extract ID from the selected option
                selected_payment_id = selected_payment.split('[')[-1].strip(']') if selected_payment else None

                # Delete button
                if st.button("Delete Payment", key="payment_delete_button", type="primary"):
                    recurring_payments = recurring_payments[recurring_payments['id'] != selected_payment_id]
                    save_recurring_payments(recurring_payments)
                    st.success(f"Deleted recurring payment: {selected_payment}")
                    st.rerun()

                # Edit button
                if st.button("Edit Payment", key="payment_edit_button"):
                    # Find the selected payment data to pre-fill the form
                    if selected_payment_id:
                        selected_payment_data = recurring_payments[recurring_payments['id'] == selected_payment_id].iloc[0]

                        # Store the values in session state to pre-fill the form
                        st.session_state['edit_payment'] = True
                        st.session_state['edit_payment_id'] = selected_payment_id
                        st.session_state['edit_description'] = selected_payment_data['description']
                        st.session_state['edit_amount'] = selected_payment_data['amount']
                        st.session_state['edit_frequency'] = selected_payment_data['frequency']
                        st.session_state['edit_category'] = selected_payment_data['category']
                        st.session_state['edit_sub_category'] = selected_payment_data['sub_category']

                        # Handle start_month and end_month
                        if pd.notna(selected_payment_data['start_month']):
                            try:
                                # Convert to datetime first
                                date_val = pd.to_datetime(selected_payment_data['start_month'])

                                # Then convert to date
                                st.session_state['edit_start_month'] = date_val.date()

                            except Exception as e:
                                st.error(f"Error converting date: {e}")
                                # Use a default date if conversion fails
                                st.session_state['edit_start_month'] = datetime.now().replace(day=1).date()

                        # Check if end_month exists and is not NaN/None
                        has_end_date = pd.notna(selected_payment_data['end_month']) and selected_payment_data['end_month'] not in ['None', '']
                        st.session_state.payment_end_date_enabled = has_end_date

                        if has_end_date:
                            try:
                                # Convert end date the same way
                                date_val = pd.to_datetime(selected_payment_data['end_month'])
                                st.session_state['edit_end_month'] = date_val.date()
                            except Exception as e:
                                st.error(f"Error converting end date: {e}")
                                st.session_state['edit_end_month'] = None

                        # Force a rerun to update the form with these values
                        st.rerun()

        # Recurring payment form
        st.subheader("Add or Edit Recurring Payment")

        # Initialize session state for the payment end date checkbox if it doesn't exist
        if 'payment_end_date_enabled' not in st.session_state:
            st.session_state.payment_end_date_enabled = False

        # Initialize edit mode flag if it doesn't exist
        if 'edit_payment' not in st.session_state:
            st.session_state.edit_payment = False

        # End date checkbox OUTSIDE the form
        payment_end_month_provided = st.checkbox("This recurring payment has an end date",
                                        value=st.session_state.payment_end_date_enabled,
                                        key="payment_end_date_checkbox")

        # Store the checkbox value in session state
        st.session_state.payment_end_date_enabled = payment_end_month_provided

        with st.form("payment_form"):
            # Pre-fill form values if in edit mode
            description = st.text_input("Description", value=st.session_state.get('edit_description', ''))
            amount = st.number_input("Amount", min_value=0.0, format="%.2f", value=st.session_state.get('edit_amount', 0.0))

            frequency_options = ["Monthly", "Bi-weekly", "Weekly", "Yearly", "Quarterly"]
            frequency = st.selectbox("Frequency", frequency_options,
                                    index=frequency_options.index(st.session_state.get('edit_frequency', "Monthly"))
                                    if st.session_state.get('edit_frequency') in frequency_options else 0,
                                     key=f"frequency_{st.session_state.get('edit_payment_id', 'new')}")

            category = st.text_input("Category", value=st.session_state.get('edit_category', ''))
            sub_category = st.text_input("Sub-category", value=st.session_state.get('edit_sub_category', ''))

            # Add payment type dropdown
            payment_type = st.selectbox("Payment Type", ["Fixed", "Variable"],
                                      index=["Fixed", "Variable"].index(st.session_state.get('edit_payment_type', 'Fixed')),
                                        key=f"payment_type_{st.session_state.get('edit_payment_id', 'new')}")

            # Add clear form button
            clear_payment_form = st.form_submit_button("Clear Form")
            if clear_payment_form:
                st.session_state['edit_description'] = ''
                st.session_state['edit_amount'] = 0.0
                st.session_state['edit_frequency'] = 'Monthly'
                st.session_state['edit_category'] = ''
                st.session_state['edit_sub_category'] = ''
                st.session_state['edit_start_month'] = datetime.now().replace(day=1).date()
                st.session_state['edit_end_month'] = datetime.now().replace(day=1).date()
                st.session_state['edit_payment_type'] = 'Fixed'
                st.session_state['edit_payment'] = False
                st.rerun()

            # Check if we're in edit mode
            is_edit_mode = st.session_state.get('edit_payment', False)

            # Get stored date from session state
            session_date = st.session_state.get('edit_start_month')

            if is_edit_mode and session_date is not None:
                # Use the stored date when editing
                start_date_value = session_date
            else:
                # Use current month's first day as default for new entries
                start_date_value = datetime.now().replace(day=1).date()

            # Create the date input with explicit min and max dates
            min_date = datetime(2020, 1, 1).date()  # Reasonable minimum date
            max_date = datetime(2030, 12, 31).date()  # Reasonable maximum date
            payment_start_month = st.date_input(
                "Start Month",
                value=start_date_value,
                min_value=min_date,
                max_value=max_date,
                key=f"payment_start_{st.session_state.get('edit_payment_id', 'new')}"  # Unique key for each edit
            )

            # Show the end date input field based on the checkbox outside the form
            payment_end_month = None
            if payment_end_month_provided:
                end_month_default = st.session_state.get('edit_end_month', datetime.now().replace(day=1) + pd.DateOffset(months=12))
                payment_end_month = st.date_input("End Month", value=end_month_default, key="payment_end")

            submit_payment = st.form_submit_button("Save Recurring Payment")

            if submit_payment and description and amount > 0:
                # Convert dates to appropriate format
                start_month_str = payment_start_month.strftime('%Y-%m-%d') if payment_start_month else None
                end_month_str = payment_end_month.strftime('%Y-%m-%d') if payment_end_month_provided and payment_end_month else None

                # Check if we're editing or adding new
                if st.session_state.get('edit_payment', False):
                    # Update the existing record
                    mask = recurring_payments['id'] == st.session_state['edit_payment_id']
                    recurring_payments.loc[mask, 'description'] = description
                    recurring_payments.loc[mask, 'amount'] = amount
                    recurring_payments.loc[mask, 'frequency'] = frequency
                    recurring_payments.loc[mask, 'category'] = category
                    recurring_payments.loc[mask, 'sub_category'] = sub_category if sub_category else 'General'
                    recurring_payments.loc[mask, 'start_month'] = start_month_str
                    recurring_payments.loc[mask, 'end_month'] = end_month_str
                    recurring_payments.loc[mask, 'payment_type'] = payment_type

                    st.success(f"Updated recurring payment: {description}")

                    # Reset edit mode
                    st.session_state.edit_payment = False
                    st.session_state.pop('edit_payment_id', None)
                    st.session_state.pop('edit_description', None)
                    st.session_state.pop('edit_amount', None)
                    st.session_state.pop('edit_frequency', None)
                    st.session_state.pop('edit_category', None)
                    st.session_state.pop('edit_sub_category', None)
                    st.session_state.pop('edit_start_month', None)
                    st.session_state.pop('edit_end_month', None)
                    st.session_state.pop('edit_payment_type', None)
                else:
                    # Create new payment with a unique ID
                    new_id = f"payment_{len(recurring_payments) + 1}" if not recurring_payments.empty else "payment_0"

                    # Create the new recurring payment
                    new_payment = pd.DataFrame({
                        'description': [description],
                        'amount': [amount],
                        'frequency': [frequency],
                        'category': [category],
                        'sub_category': [sub_category if sub_category else 'General'],
                        'start_month': [start_month_str],
                        'end_month': [end_month_str],
                        'payment_type': [payment_type],
                        'id': [new_id]
                    })

                    # Add as a new record
                    recurring_payments = pd.concat([recurring_payments, new_payment], ignore_index=True)
                    st.success(f"Added new recurring payment: {description}")

                # Save updated recurring payments data
                save_recurring_payments(recurring_payments)
                st.rerun()  # Refresh the page to show updated data
        # ------------------ Monthly Budget Management (new section) ------------------
        st.subheader("Monthly Budget")
        monthly_budget_df = load_monthly_budget()

        if not monthly_budget_df.empty:
            st.write("### Current Monthly Budget Items")
            colb1, colb2 = st.columns([4,1])
            with colb1:
                st.dataframe(monthly_budget_df, use_container_width=True, hide_index=True)
            with colb2:
                st.write("### Actions")
                budget_options = [f"{row['description']} (${row['amount']}) [{row['id']}]" for _, row in monthly_budget_df.iterrows()]
                selected_budget_item = st.selectbox("Select to edit/delete", options=budget_options, key="budget_select")
                selected_budget_id = selected_budget_item.split('[')[-1].strip(']') if selected_budget_item else None
                if st.button("Delete Budget Item", key="budget_delete_button", type="primary") and selected_budget_id:
                    monthly_budget_df = monthly_budget_df[monthly_budget_df['id'] != selected_budget_id]
                    save_monthly_budget(monthly_budget_df)
                    st.success(f"Deleted budget item: {selected_budget_item}")
                    st.rerun()
                if st.button("Edit Budget Item", key="budget_edit_button") and selected_budget_id:
                    selected_budget_row = monthly_budget_df[monthly_budget_df['id'] == selected_budget_id].iloc[0]
                    st.session_state.edit_budget = True
                    st.session_state.edit_budget_id = selected_budget_id
                    st.session_state.edit_budget_description = selected_budget_row['description']
                    st.session_state.edit_budget_amount = selected_budget_row['amount']
                    st.session_state.edit_budget_frequency = selected_budget_row['frequency']
                    st.session_state.edit_budget_category = selected_budget_row['category']
                    st.session_state.edit_budget_sub_category = selected_budget_row['sub_category']
                    st.session_state.edit_budget_payment_type = selected_budget_row['payment_type']
                    if pd.notna(selected_budget_row['start_month']):
                        st.session_state.edit_budget_start_month = pd.to_datetime(selected_budget_row['start_month']).date()
                    if pd.notna(selected_budget_row['end_month']) and selected_budget_row['end_month'] not in ['', 'None']:
                        st.session_state.budget_end_date_enabled = True
                        st.session_state.edit_budget_end_month = pd.to_datetime(selected_budget_row['end_month']).date()
                    else:
                        st.session_state.budget_end_date_enabled = False
                    st.rerun()
        else:
            st.info("No budget items defined yet.")

        # Initialize session state flags
        if 'edit_budget' not in st.session_state:
            st.session_state.edit_budget = False
        if 'budget_end_date_enabled' not in st.session_state:
            st.session_state.budget_end_date_enabled = False

        budget_end_month_toggle = st.checkbox("This budget item has an end date", value=st.session_state.budget_end_date_enabled, key="budget_end_date_checkbox")
        st.session_state.budget_end_date_enabled = budget_end_month_toggle

        with st.form("monthly_budget_form"):
            description_b = st.text_input("Description", value=st.session_state.get('edit_budget_description',''))
            amount_b = st.number_input("Amount", min_value=0.0, format='%.2f', value=float(st.session_state.get('edit_budget_amount', 0.0)), key=f"budget_amount_{st.session_state.get('edit_budget_id', 'new')}")
            freq_opts = ["Monthly","Bi-weekly","Weekly","Yearly","Quarterly"]
            frequency_b = st.selectbox("Frequency", freq_opts, index=freq_opts.index(st.session_state.get('edit_budget_frequency','Monthly')) if st.session_state.get('edit_budget_frequency','Monthly') in freq_opts else 0,
                                       key=f"budget_frequency_{st.session_state.get('edit_budget_id', 'new')}")
            category_b = st.text_input("Category", value=st.session_state.get('edit_budget_category',''))
            sub_category_b = st.text_input("Sub-category", value=st.session_state.get('edit_budget_sub_category',''))
            payment_type_b = st.selectbox("Payment Type", ["Fixed","Variable"], index=["Fixed","Variable"].index(st.session_state.get('edit_budget_payment_type','Fixed')),
                                          key=f"budget_payment_type_{st.session_state.get('edit_budget_id', 'new')}")
            # Dates
            start_default_b = st.session_state.get('edit_budget_start_month', datetime.now().replace(day=1).date())
            start_month_b = st.date_input("Start Month", value=start_default_b,
                                          key=f"budget_start_{st.session_state.get('edit_budget_id', 'new')}" )
            end_month_b = None
            if budget_end_month_toggle:
                end_default_b = st.session_state.get('edit_budget_end_month', datetime.now().replace(day=1).date())
                end_month_b = st.date_input("End Month", value=end_default_b, key=f"budget_end_{st.session_state.get('edit_budget_id', 'new')}")
            clear_budget = st.form_submit_button("Clear Form")
            if clear_budget:
                for k in ['edit_budget','edit_budget_id','edit_budget_description','edit_budget_amount','edit_budget_frequency','edit_budget_category','edit_budget_sub_category','edit_budget_payment_type','edit_budget_start_month','edit_budget_end_month']:
                    if k in st.session_state:
                        st.session_state.pop(k)
                keys_tuple = tuple([
                    'budget_description', 'budget_amount', 'budget_frequency', 'budget_category',
                    'budget_sub_category', 'budget_payment_type', 'budget_start_month', 'budget_end_month',
                    'budget_end_date_enabled'
                ])
                keys_to_delete = [key for key in st.session_state.keys() if key.startswith(keys_tuple)]
                st.write("keys_to_delete", keys_to_delete)
                for k in keys_to_delete:
                    st.session_state.pop(k, None)
                st.session_state.budget_end_date_enabled = False
                st.write("key: ", st.session_state.get('budget_amount_new', 0.0))
                st.rerun()
            submit_budget = st.form_submit_button("Save Budget Item")
            if submit_budget and description_b and amount_b > 0:
                st.write("in here")
                start_str_b = start_month_b.strftime('%Y-%m-%d') if start_month_b else None
                end_str_b = end_month_b.strftime('%Y-%m-%d') if budget_end_month_toggle and end_month_b else None
                if st.session_state.get('edit_budget', False):
                    mask = monthly_budget_df['id'] == st.session_state.edit_budget_id
                    monthly_budget_df.loc[mask,'description'] = description_b
                    monthly_budget_df.loc[mask,'amount'] = amount_b
                    monthly_budget_df.loc[mask,'frequency'] = frequency_b
                    monthly_budget_df.loc[mask,'category'] = category_b
                    monthly_budget_df.loc[mask,'sub_category'] = sub_category_b if sub_category_b else 'General'
                    monthly_budget_df.loc[mask,'start_month'] = start_str_b
                    monthly_budget_df.loc[mask,'end_month'] = end_str_b
                    monthly_budget_df.loc[mask,'payment_type'] = payment_type_b
                    st.success(f"Updated budget item: {description_b}")
                    # reset
                    for k in ['edit_budget','edit_budget_id','edit_budget_description','edit_budget_amount','edit_budget_frequency','edit_budget_category','edit_budget_sub_category','edit_budget_payment_type','edit_budget_start_month','edit_budget_end_month']:
                        if k in st.session_state:
                            st.session_state.pop(k)
                else:
                    new_id_b = f"budget_{len(monthly_budget_df)+1}" if not monthly_budget_df.empty else "budget_0"
                    new_row_b = pd.DataFrame({
                        'description':[description_b],
                        'amount':[amount_b],
                        'frequency':[frequency_b],
                        'category':[category_b],
                        'sub_category':[sub_category_b if sub_category_b else 'General'],
                        'start_month':[start_str_b],
                        'end_month':[end_str_b],
                        'payment_type':[payment_type_b],
                        'id':[new_id_b]
                    })
                    monthly_budget_df = pd.concat([monthly_budget_df, new_row_b], ignore_index=True)
                    st.success(f"Added new budget item: {description_b}")
                    keys_tuple = tuple([
                        'budget_description', 'budget_amount', 'budget_frequency', 'budget_category',
                        'budget_sub_category', 'budget_payment_type', 'budget_start_month', 'budget_end_month',
                        'budget_end_date_enabled'
                    ])
                    for k in [key for key in st.session_state.keys() if key.startswith(keys_tuple)]:
                        st.session_state.pop(k, None)
                    # Edit prefill keys (safe to remove if present)
                    for k in [
                        'edit_budget', 'edit_budget_id', 'edit_budget_description', 'edit_budget_amount',
                        'edit_budget_frequency', 'edit_budget_category', 'edit_budget_sub_category',
                        'edit_budget_payment_type', 'edit_budget_start_month', 'edit_budget_end_month'
                    ]:
                        st.session_state.pop(k, None)
                save_monthly_budget(monthly_budget_df)
                st.rerun()

    with tab2:
        st.header("Financial Overview")

        # Date selection for financial calculations - change to month/year selection
        col1, col2 = st.columns(2)

        with col1:
            # Get available years from the data
            current_year = datetime.now().year
            available_years = list(range(current_year-2, current_year+2))
            selected_year = st.selectbox("Select Year", options=available_years, index=2)  # Default to current year

        with col2:
            # Month selection
            selected_month = st.selectbox(
                "Select Month",
                options=list(range(1, 13)),
                format_func=lambda x: calendar.month_name[x],
                index=datetime.now().month-1  # Default to current month
            )

        # Create a timestamp for the selected month (first day of month)
        selected_date_ts = pd.Timestamp(year=selected_year, month=selected_month, day=1)

        # Get the last day of the selected month for date range filtering
        if selected_month == 12:
            last_day_ts = pd.Timestamp(year=selected_year+1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            last_day_ts = pd.Timestamp(year=selected_year, month=selected_month+1, day=1) - pd.Timedelta(days=1)

        # Calculate active monthly income considering start and end dates
        active_monthly_income = calculate_active_monthly_income(income_data, selected_date_ts)

        # Calculate total monthly recurring payments for the selected month
        recurring_payments_wo_savings = recurring_payments[recurring_payments["category"] != "Savings"]
        active_monthly_payments = calculate_active_monthly_payments(recurring_payments_wo_savings, selected_date_ts)


        # calculate
        active_monthly_budget = calculate_active_monthly_payments(monthly_budget_df, selected_date_ts)

        # Get actual expenses for the selected month from the transaction data
        if not expenses_only.empty:
            monthly_expenses = expenses_only[
                (expenses_only['Date'] >= selected_date_ts) &
                (expenses_only['Date'] <= last_day_ts)
            ]['Amount'].sum()
        else:
            monthly_expenses = 0.0

        # Calculate remaining budget after actual expenses
        remaining_income = active_monthly_income - active_monthly_payments - monthly_expenses
        remaining_budget = active_monthly_budget - active_monthly_payments - monthly_expenses

        # Display financial metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monthly Income", f"${active_monthly_income:.2f}")

            # Create a copy and convert dates to proper timestamps for comparison
            income_data_copy = income_data.copy()
            if 'start_month' in income_data_copy.columns:
                income_data_copy['start_month'] = pd.to_datetime(income_data_copy['start_month'])
            if 'end_month' in income_data_copy.columns:
                income_data_copy['end_month'] = pd.to_datetime(income_data_copy['end_month'])

            # Now do the comparison with consistent types
            active_sources = len(income_data_copy[
                ((pd.isna(income_data_copy['start_month'])) | (income_data_copy['start_month'] <= selected_date_ts)) &
                ((pd.isna(income_data_copy['end_month'])) | (income_data_copy['end_month'] >= selected_date_ts))
            ])
            st.caption(f"Based on {active_sources} active income sources")
        with col2:
            st.metric("Fixed Expenses", f"${active_monthly_payments:.2f}")

            # Show variable expenses pulled from Monthly Expense Summary
            st.metric("Variable Expenses", f"${monthly_expenses:.2f}")

        with col3:
            # Colored remaining budget instead of st.metric
            render_colored_amount("Remaining Budget", remaining_budget)

            # Colored remaining budget instead of st.metric
            render_colored_amount("Remaining Income", remaining_income)
            st.caption(f"For {calendar.month_name[selected_month]} {selected_year}")

        # Monthly income, expense, and savings breakdown
        st.subheader("Monthly Financial Breakdown")

        # Prepare data for plotting by month
        months_to_display = 12  # Show 12 months of data

        # Create a range of months centered on selected month
        start_range_month = selected_date_ts - pd.DateOffset(months=months_to_display-1)
        month_range = [start_range_month + pd.DateOffset(months=i) for i in range(months_to_display)]

        # Initialize data arrays for the chart
        month_labels = [d.strftime('%b %Y') for d in month_range]
        income_values = []
        budget_values = []
        fixed_expense_values = []
        variable_expense_values = []
        savings_values = []

        # Calculate values for each month in the range
        for month_date in month_range:
            # Last day of current month for filtering
            if month_date.month == 12:
                month_end = pd.Timestamp(year=month_date.year+1, month=1, day=1) - pd.Timedelta(days=1)
            else:
                month_end = pd.Timestamp(year=month_date.year, month=month_date.month+1, day=1) - pd.Timedelta(days=1)

            # Calculate income for this month
            month_income = calculate_active_monthly_income(income_data, month_date)
            income_values.append(month_income)

            # Calculate fixed expenses for this month
            month_fixed_expenses = calculate_active_monthly_payments(recurring_payments, month_date)
            fixed_expense_values.append(month_fixed_expenses)

            # Calculate fixed budget for this month
            month_budget = calculate_active_monthly_payments(monthly_budget_df, month_date)
            budget_values.append(month_budget)

            # Calculate variable expenses from transaction data
            if not expenses_only.empty:
                month_variable_expenses = expenses_only[
                    (expenses_only['Date'] >= month_date) &
                    (expenses_only['Date'] <= month_end)
                ]['Amount'].sum()
            else:
                month_variable_expenses = 0.0
            variable_expense_values.append(month_variable_expenses)

            # Calculate savings
            month_savings = month_income - month_fixed_expenses - month_variable_expenses
            savings_values.append(month_savings)

        # Create data for the stacked bar chart
        fig = go.Figure()

        # Add fixed expenses (bottom bar)
        fig.add_trace(go.Bar(
            x=month_labels,
            y=fixed_expense_values,
            name='Fixed Expenses',
            marker_color='#ff7f0e'
        ))

        # Add variable expenses (stacked on fixed)
        fig.add_trace(go.Bar(
            x=month_labels,
            y=variable_expense_values,
            name='Variable Expenses',
            marker_color='#d62728'
        ))

        # Add income line
        fig.add_trace(go.Scatter(
            x=month_labels,
            y=income_values,
            name='Income',
            line=dict(color='#2ca02c', width=3)
        ))

        # Add Budget line
        fig.add_trace(go.Scatter(
            x=month_labels,
            y=budget_values,
            name='Budget',
            line=dict(color='#000000', width=3)
        ))

        # Add savings line
        fig.add_trace(go.Scatter(
            x=month_labels,
            y=savings_values,
            name='Savings',
            line=dict(color='#1f77b4', width=2, dash='dash')
        ))

        # Update layout
        fig.update_layout(
            title='Monthly Financial Summary',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            barmode='stack',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Income vs Expense Ratio Analysis
        st.subheader("Income vs Expense Ratio Analysis")

        # Calculate ratios
        fixed_ratio = (active_monthly_payments / active_monthly_income * 100) if active_monthly_income > 0 else 0
        variable_ratio = (monthly_expenses / active_monthly_income * 100) if active_monthly_income > 0 else 0
        savings_ratio = 100 - fixed_ratio - variable_ratio

        # Create a pie chart
        labels = ['Fixed Expenses', 'Variable Expenses', 'Savings']
        values = [fixed_ratio, variable_ratio, savings_ratio]

        fig_pie = px.pie(
            names=labels,
            values=values,
            title=f'Income Distribution for {calendar.month_name[selected_month]} {selected_year}',
            color_discrete_sequence=['#ff7f0e', '#d62728', '#1f77b4'],
            hole=0.4
        )

        fig_pie.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig_pie, use_container_width=True)

        # Expense Breakdown by Category for the selected month
        if not expenses_only.empty:
            st.subheader("Expense Breakdown by Category")

            # Filter expenses for the selected month
            month_expenses = expenses_only[
                (expenses_only['Date'] >= selected_date_ts) &
                (expenses_only['Date'] <= last_day_ts)
            ]

            if not month_expenses.empty:
                # ---------------- New logic: Compute Budget vs Actual by Category ----------------
                # Group actual (variable) expenses by Enhanced_Category
                category_expenses = month_expenses.groupby('Enhanced_Category')['Amount'].sum().reset_index()
                category_expenses.rename(columns={'Enhanced_Category': 'Category', 'Amount': 'Expenses'}, inplace=True)

                # Merge in active recurring payments (frequency-normalized) excluding category 'Savings'
                if 'recurring_payments' in locals() and not recurring_payments.empty and 'category' in recurring_payments.columns:
                    rp_active = recurring_payments.copy()
                    # Date parsing (coerce invalid to NaT)
                    if 'start_month' in rp_active.columns:
                        rp_active['start_month'] = pd.to_datetime(rp_active['start_month'], errors='coerce')
                    if 'end_month' in rp_active.columns:
                        rp_active['end_month'] = pd.to_datetime(rp_active['end_month'], errors='coerce')

                    # Filter to payments active for selected month
                    rp_active = rp_active[
                        ((rp_active['start_month'].isna()) | (rp_active['start_month'] <= selected_date_ts)) &
                        ((rp_active['end_month'].isna()) | (rp_active['end_month'] >= selected_date_ts))
                    ]

                    if not rp_active.empty:
                        # Exclude Savings (case-insensitive, handle NaNs)
                        rp_active['category'] = rp_active['category'].fillna('')
                        rp_active = rp_active[rp_active['category'].str.lower() != 'savings']

                        if not rp_active.empty:
                            # Normalize amount per month using existing helper
                            rp_active['Monthly_Amount'] = rp_active.apply(lambda r: calculate_monthly_payment(r['amount'], r['frequency']), axis=1)
                            rp_by_cat = rp_active.groupby('category')['Monthly_Amount'].sum().reset_index()
                            rp_by_cat.rename(columns={'category': 'Category', 'Monthly_Amount': 'Recurring'}, inplace=True)

                            # Outer merge to include categories that only have recurring payments
                            category_expenses = category_expenses.merge(rp_by_cat, on='Category', how='outer')
                            category_expenses['Expenses'] = category_expenses['Expenses'].fillna(0.0) + category_expenses['Recurring'].fillna(0.0)
                            category_expenses.drop(columns=['Recurring'], inplace=True)
                # ---------------- End merge of recurring payments ----------------

                # Prepare monthly budget by category for the selected month
                category_budgets = pd.DataFrame(columns=['Category', 'Budget'])
                if 'category' in monthly_budget_df.columns and not monthly_budget_df.empty:
                    budget_active = monthly_budget_df.copy()
                    # Convert dates to datetime for comparison
                    if 'start_month' in budget_active.columns:
                        budget_active['start_month'] = pd.to_datetime(budget_active['start_month'], errors='coerce')
                    if 'end_month' in budget_active.columns:
                        budget_active['end_month'] = pd.to_datetime(budget_active['end_month'], errors='coerce')

                    # Filter active budget items for the selected month
                    budget_active = budget_active[
                        ((budget_active['start_month'].isna()) | (budget_active['start_month'] <= selected_date_ts)) &
                        ((budget_active['end_month'].isna()) | (budget_active['end_month'] >= selected_date_ts))
                    ]

                    if not budget_active.empty:
                        # Calculate the effective monthly amount for each budget line using existing helper
                        budget_active['Monthly_Amount'] = budget_active.apply(
                            lambda row: calculate_monthly_payment(row['amount'], row['frequency']), axis=1
                        )
                        category_budgets = budget_active.groupby('category')['Monthly_Amount'].sum().reset_index()
                        category_budgets.rename(columns={'category': 'Category', 'Monthly_Amount': 'Budget'}, inplace=True)

                # Merge budgets and expenses (union of categories)
                combined = pd.merge(category_budgets, category_expenses, on='Category', how='outer')
                combined['Budget'] = combined['Budget'].fillna(0.0)
                combined['Expenses'] = combined['Expenses'].fillna(0.0)
                combined['Variance'] = combined['Budget'] - combined['Expenses']  # Positive = Under budget

                # Sort categories by highest Expenses (fallback to Budget if equal)
                combined.sort_values(['Expenses', 'Budget'], ascending=[False, False], inplace=True)

                # Build grouped bar chart Budget vs Expenses
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=combined['Category'],
                    y=combined['Budget'],
                    name='Budget',
                    marker_color='#1f77b4',
                    hovertemplate='<b>%{x}</b><br>Budget: $%{y:.2f}<extra></extra>'
                ))
                fig_bar.add_trace(go.Bar(
                    x=combined['Category'],
                    y=combined['Expenses'],
                    name='Expenses',
                    marker_color='#d62728',
                    hovertemplate='<b>%{x}</b><br>Expenses: $%{y:.2f}<extra></extra>'
                ))

                # Add variance annotations (optional small text above bars)
                annotations = []
                for idx, row in combined.iterrows():
                    variance_text = f"Δ ${(row['Variance']):.0f}" if row['Variance'] != 0 else ""
                    if variance_text:
                        annotations.append(dict(
                            x=row['Category'],
                            y=max(row['Budget'], row['Expenses']) * 1.02,
                            text=variance_text,
                            showarrow=False,
                            font=dict(size=10, color='#333')
                        ))

                fig_bar.update_layout(
                    title=f'Budget vs Actual Expenses by Category - {calendar.month_name[selected_month]} {selected_year}',
                    xaxis_title='Category',
                    yaxis_title='Amount ($)',
                    barmode='group',
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    annotations=annotations,
                    margin=dict(t=80)
                )

                # Add variance hover info via customdata
                fig_bar.update_traces(
                    selector=dict(name='Expenses'),
                    hovertemplate='<b>%{x}</b><br>Expenses: $%{y:.2f}<br>' +
                                  'Budget: $%{customdata[0]:.2f}<br>Variance: $%{customdata[1]:.2f}<extra></extra>',
                    customdata=np.stack((combined['Budget'], combined['Variance']), axis=-1)
                )
                fig_bar.update_traces(
                    selector=dict(name='Budget'),
                    hovertemplate='<b>%{x}</b><br>Budget: $%{y:.2f}<br>' +
                                  'Expenses: $%{customdata[0]:.2f}<br>Variance: $%{customdata[1]:.2f}<extra></extra>',
                    customdata=np.stack((combined['Expenses'], combined['Variance']), axis=-1)
                )

                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption('Variance = Budget - Expenses (positive means under budget)')
                # ------------------------------------------------------------------------------
            else:
                st.info(f"No expense data available for {calendar.month_name[selected_month]} {selected_year}")
        else:
            st.info("No expense data available for analysis")

    with tab3:
        st.header("Monthly Expense Summary")

        # Date range filter
        min_date = expenses_only['Date'].min()
        max_date = expenses_only['Date'].max()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)

        # Convert to datetime for comparison
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter data based on selected date range
        filtered_data = expenses_only[
            (expenses_only['Date'] >= start_date) &
            (expenses_only['Date'] <= end_date)
        ]

        # Group by month and enhanced category
        filtered_data['Month'] = filtered_data['Date'].dt.to_period('M')
        monthly_category_expenses = filtered_data.groupby(['Month', 'Enhanced_Category'])['Amount'].sum().reset_index()
        monthly_category_expenses['Month'] = monthly_category_expenses['Month'].astype(str)

        # Create a pivot table for visualization
        pivot_data = monthly_category_expenses.pivot(index='Month', columns='Enhanced_Category', values='Amount').fillna(0)

        # Plot the data using Plotly
        fig = px.bar(
            monthly_category_expenses,
            x='Month',
            y='Amount',
            color='Enhanced_Category',
            title='Monthly Expenses by Category',
            barmode='stack',
            height=500
        )

        # Customize the layout
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            legend_title='Category',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Expenses", f"${filtered_data['Amount'].sum():.2f}")
        with col2:
            st.metric("Average Monthly Expense", f"${filtered_data.groupby('Month')['Amount'].sum().mean():.2f}")
        with col3:
            st.metric("Number of Transactions", f"{len(filtered_data)}")

    with tab4:
        st.header("Expense by Category")

        # Time period selection options
        time_filter_option = st.radio(
            "Select Time Filter Type",
            ["Preset Periods", "Custom Date Range", "Specific Month/Year"]
        )

        # Filter based on selected time filter option
        today = pd.to_datetime(datetime.now())

        if time_filter_option == "Preset Periods":
            # Original preset time periods
            time_period = st.selectbox(
                "Select Time Period",
                ["All Time", "Last 30 Days", "Last 3 Months", "Last 6 Months", "Last Year"]
            )

            # Apply preset filters
            if time_period == "Last 30 Days":
                category_data = expenses_only[expenses_only['Date'] >= (today - pd.Timedelta(days=30))]
            elif time_period == "Last 3 Months":
                category_data = expenses_only[expenses_only['Date'] >= (today - pd.Timedelta(days=90))]
            elif time_period == "Last 6 Months":
                category_data = expenses_only[expenses_only['Date'] >= (today - pd.Timedelta(days=180))]
            elif time_period == "Last Year":
                category_data = expenses_only[expenses_only['Date'] >= (today - pd.Timedelta(days=365))]
            else:
                category_data = expenses_only

            filter_description = time_period

        elif time_filter_option == "Custom Date Range":
            # Custom date range selection
            col1, col2 = st.columns(2)
            with col1:
                custom_start_date = st.date_input(
                    "Start Date",
                    value=pd.to_datetime(today - pd.Timedelta(days=30)).date(),
                    min_value=expenses_only['Date'].min().date(),
                    max_value=today.date()
                )
            with col2:
                custom_end_date = st.date_input(
                    "End Date",
                    value=today.date(),
                    min_value=expenses_only['Date'].min().date(),
                    max_value=today.date()
                )

            # Convert to datetime for filtering
            custom_start = pd.to_datetime(custom_start_date)
            custom_end = pd.to_datetime(custom_end_date)

            # Apply custom date range filter
            category_data = expenses_only[
                (expenses_only['Date'] >= custom_start) &
                (expenses_only['Date'] <= custom_end)
            ]

            filter_description = f"{custom_start_date.strftime('%b %d, %Y')} - {custom_end_date.strftime('%b %d, %Y')}"

        else:  # Specific Month/Year
            # Get available months and years from the data
            expenses_only['Year'] = expenses_only['Date'].dt.year
            expenses_only['Month'] = expenses_only['Date'].dt.month

            available_years = sorted(expenses_only['Year'].unique())

            # Year and month selection
            col1, col2 = st.columns(2)
            with col1:
                selected_year = st.selectbox("Select Year", options=available_years, index=len(available_years)-1)

            # Filter months available for the selected year
            available_months = sorted(expenses_only[expenses_only['Year'] == selected_year]['Month'].unique())

            with col2:
                selected_month = st.selectbox(
                    "Select Month",
                    options=available_months,
                    format_func=lambda x: calendar.month_name[x],
                    index=min(len(available_months)-1, max(0, datetime.now().month-1) if selected_year == datetime.now().year else 0)
                )

            # Apply month/year filter
            category_data = expenses_only[
                (expenses_only['Year'] == selected_year) &
                (expenses_only['Month'] == selected_month)
            ]

            filter_description = f"{calendar.month_name[selected_month]} {selected_year}"

        # Group by category group
        if len(category_data) > 0:
            # View selector for category breakdown
            view_type = st.radio(
                "Select View",
                ["Categories Only", "Categories with Subcategories"],
                horizontal=True
            )

            if view_type == "Categories Only":
                # Group by main category
                category_summary = category_data.groupby('Enhanced_Category')['Amount'].sum().sort_values(ascending=False).reset_index()

                col1, col2 = st.columns([3, 2])

                with col1:
                    # Pie chart for category breakdown
                    fig_pie = px.pie(
                        category_summary,
                        values='Amount',
                        names='Enhanced_Category',
                        title=f'Expense Distribution by Category ({filter_description})',
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Table showing category totals
                    st.subheader("Category Totals")
                    category_summary['Percentage'] = (category_summary['Amount'] / category_summary['Amount'].sum() * 100)
                    category_summary_display = category_summary.copy()
                    category_summary_display['Amount'] = category_summary_display['Amount'].map('${:,.2f}'.format)
                    category_summary_display['Percentage'] = category_summary_display['Percentage'].map('{:.1f}%'.format)
                    st.table(category_summary_display)

                    # Total for the period
                    st.metric("Total Expenses", f"${category_data['Amount'].sum():.2f}")
                    st.metric("Number of Transactions", f"{len(category_data)}")
            else:
                # Group by main category and subcategory
                subcategory_summary = category_data.groupby(['Enhanced_Category', 'Sub_Category'])['Amount'].sum().reset_index()

                # Create a combined name for the display
                subcategory_summary['Category_SubCategory'] = subcategory_summary['Enhanced_Category'] + ' > ' + subcategory_summary['Sub_Category']
                subcategory_summary = subcategory_summary.sort_values('Amount', ascending=False)

                # Main visualization
                fig_sunburst = px.sunburst(
                    subcategory_summary,
                    path=['Enhanced_Category', 'Sub_Category'],
                    values='Amount',
                    title=f'Expense Distribution by Category and Subcategory ({filter_description})',
                    height=600
                )

                fig_sunburst.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_sunburst, use_container_width=True)

                # Table with details
                st.subheader("Category and Subcategory Details")

                # Calculate total amount and percentages
                total_amount = subcategory_summary['Amount'].sum()
                subcategory_summary['Percentage'] = subcategory_summary['Amount'] / total_amount * 100

                # Format for display
                display_df = subcategory_summary.copy()
                display_df['Amount'] = display_df['Amount'].map('${:,.2f}'.format)
                display_df['Percentage'] = display_df['Percentage'].map('{:.1f}%'.format)

                # Show the table
                st.dataframe(
                    display_df[['Enhanced_Category', 'Sub_Category', 'Amount', 'Percentage']],
                    use_container_width=True,
                    hide_index=True
                )

                # Total for the period
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Expenses", f"${category_data['Amount'].sum():.2f}")
                with col2:
                    st.metric("Number of Transactions", f"{len(category_data)}")

            # Top merchants by category
            st.subheader("Top Merchants by Selected Category")
            selected_category = st.selectbox("Select Category", category_data['Enhanced_Category'].unique())

            # Show sub-categories if available for this category
            selected_data = category_data[category_data['Enhanced_Category'] == selected_category]
            subcats = selected_data['Sub_Category'].unique()

            if len(subcats) > 1:  # Only show if there are multiple subcategories
                selected_subcat = st.selectbox(
                    "Select Subcategory",
                    options=['All Subcategories'] + list(subcats)
                )

                if selected_subcat != 'All Subcategories':
                    selected_data = selected_data[selected_data['Sub_Category'] == selected_subcat]

            # Group by description
            top_merchants = selected_data.groupby('Description')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()

            fig_merchants = px.bar(
                top_merchants,
                x='Amount',
                y='Description',
                orientation='h',
                title=f'Top 10 Merchants in {selected_category}',
                height=500
            )

            fig_merchants.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_merchants, use_container_width=True)
        else:
            st.info("No transactions found for the selected time period.")

    with tab5:
        st.header("Spending Trends")

        # Spending over time
        expenses_only['Month'] = expenses_only['Date'].dt.to_period('M')
        monthly_expenses = expenses_only.groupby('Month')['Amount'].sum().reset_index()
        monthly_expenses['Month'] = monthly_expenses['Month'].astype(str)

        # Line chart of monthly spending
        fig_trend = px.line(
            monthly_expenses,
            x='Month',
            y='Amount',
            markers=True,
            title='Monthly Spending Trend',
            height=400
        )

        fig_trend.update_layout(
            xaxis_title='Month',
            yaxis_title='Total Expenses ($)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_trend, use_container_width=True)

        # Year-over-year comparison if enough data is available
        expenses_only['Year'] = expenses_only['Date'].dt.year
        expenses_only['Month_Num'] = expenses_only['Date'].dt.month

        # Check if we have data for multiple years
        years = expenses_only['Year'].unique()
        if len(years) > 1:
            st.subheader("Year-over-Year Comparison")

            yearly_monthly_expenses = expenses_only.groupby(['Year', 'Month_Num'])['Amount'].sum().reset_index()
            yearly_monthly_expenses['Month_Name'] = yearly_monthly_expenses['Month_Num'].apply(lambda x: calendar.month_abbr[x])

            fig_yoy = px.line(
                yearly_monthly_expenses,
                x='Month_Num',
                y='Amount',
                color='Year',
                markers=True,
                labels={'Month_Num': 'Month', 'Amount': 'Expenses ($)'},
                title='Year-over-Year Monthly Expenses',
                height=400
            )

            fig_yoy.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(1, 13)),
                    ticktext=[calendar.month_abbr[i] for i in range(1, 13)]
                ),
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_yoy, use_container_width=True)

        # Day of week analysis
        st.subheader("Spending by Day of Week")
        expenses_only['Day_of_Week'] = expenses_only['Date'].dt.day_name()

        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_expenses = expenses_only.groupby('Day_of_Week')['Amount'].sum().reindex(day_order).reset_index()

        fig_dow = px.bar(
            dow_expenses,
            x='Day_of_Week',
            y='Amount',
            title='Total Spending by Day of Week',
            height=400,
            color='Day_of_Week'
        )

        st.plotly_chart(fig_dow, use_container_width=True)

    with tab6:
        st.header("Category Comparison")

        # Add description of this tab's purpose
        st.markdown("""
        This tab helps you understand month-by-month category changes and identifies what's causing increases in your spending.
        Select categories and months to analyze spending patterns and get insights on significant changes.
        """)

        # Month selection
        expenses_only['Month'] = expenses_only['Date'].dt.to_period('M')
        all_months = sorted(expenses_only['Month'].unique())

        if len(all_months) >= 2:
            # Convert to strings for display
            month_strings = [str(m) for m in all_months]

            col1, col2 = st.columns(2)
            with col1:
                compare_month1 = st.selectbox(
                    "Select First Month",
                    options=month_strings,
                    index=len(month_strings)-2  # Default to second-to-last month
                )
            with col2:
                compare_month2 = st.selectbox(
                    "Select Second Month",
                    options=month_strings,
                    index=len(month_strings)-1  # Default to most recent month
                )

            # Category selection (multi-select or show all)
            show_all_categories = st.checkbox("Show All Categories", value=True)

            if not show_all_categories:
                compare_categories = st.multiselect(
                    "Select Categories to Compare",
                    options=expenses_only['Enhanced_Category'].unique(),
                    default=expenses_only['Enhanced_Category'].unique()[:3]  # Default to first three categories
                )
            else:
                compare_categories = expenses_only['Enhanced_Category'].unique()

            # Filter data for the selected months
            month1_data = expenses_only[expenses_only['Month'].astype(str) == compare_month1]
            month2_data = expenses_only[expenses_only['Month'].astype(str) == compare_month2]

            # Group by category
            month1_by_category = month1_data[month1_data['Enhanced_Category'].isin(compare_categories)].groupby('Enhanced_Category')['Amount'].sum()
            month2_by_category = month2_data[month2_data['Enhanced_Category'].isin(compare_categories)].groupby('Enhanced_Category')['Amount'].sum()

            # Combine into a DataFrame for comparison
            comparison_df = pd.DataFrame({
                compare_month1: month1_by_category,
                compare_month2: month2_by_category
            }).fillna(0)

            # Calculate the difference and percentage change
            comparison_df['Difference'] = comparison_df[compare_month2] - comparison_df[compare_month1]
            comparison_df['% Change'] = ((comparison_df[compare_month2] - comparison_df[compare_month1]) / comparison_df[compare_month1] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

            # Sort by absolute difference for better visualization
            comparison_df = comparison_df.sort_values('Difference', key=abs, ascending=False)

            # Display the comparison table
            st.subheader(f"Category Comparison: {compare_month1} vs {compare_month2}")

            # Format for display
            display_df = comparison_df.copy()
            display_df[compare_month1] = display_df[compare_month1].map('${:,.2f}'.format)
            display_df[compare_month2] = display_df[compare_month2].map('${:,.2f}'.format)
            display_df['Difference'] = display_df['Difference'].map('${:,.2f}'.format)
            display_df['% Change'] = display_df['% Change'].map('{:+.1f}%'.format)

            st.dataframe(display_df, use_container_width=True)

            # Prepare data for visualization with color coding
            plot_data = comparison_df.reset_index()
            # Add a color column to explicitly set colors based on difference
            plot_data['Color'] = plot_data['Difference'].apply(lambda x: '#FF9B99' if x > 0 else '#96D6B0')

            # Visualize the comparison with color-coded bars
            fig_bar_compare = px.bar(
                plot_data,
                x='Enhanced_Category',
                y='Difference',
                title=f'Change in Spending by Category ({compare_month1} to {compare_month2})',
                color='Color',  # Use the color column for coloring
                color_discrete_map="identity",  # Use the exact colors provided
                height=400,
                labels={'Difference': 'Change in Amount ($)', 'Enhanced_Category': 'Category'}
            )

            # Improve the appearance by hiding the color legend which is redundant
            fig_bar_compare.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_bar_compare, use_container_width=True)

            # Significant changes analysis
            st.subheader("What Caused The Changes?")

            # Find categories with significant changes
            significant_increases = comparison_df[comparison_df['Difference'] > 0].sort_values('Difference', ascending=False)
            significant_decreases = comparison_df[comparison_df['Difference'] < 0].sort_values('Difference', ascending=True)

            # Show insights for top increases
            if not significant_increases.empty:
                st.markdown("### 📈 Notable Spending Increases")

                for category, row in significant_increases.head(3).iterrows():
                    st.markdown(f"#### {category}: ${row['Difference']:.2f} increase ({row['% Change']:.1f}%)")

                    # Get transactions for this category in the two months
                    category_month2 = month2_data[month2_data['Enhanced_Category'] == category].sort_values('Amount', ascending=False)
                    category_month1 = month1_data[month1_data['Enhanced_Category'] == category].sort_values('Amount', ascending=False)

                    # Find new merchants that weren't in the previous month
                    month1_merchants = set(category_month1['Description'].str.lower())
                    month2_merchants = set(category_month2['Description'].str.lower())
                    new_merchants = month2_merchants - month1_merchants

                    # Find merchants with increased spending
                    common_merchants = month1_merchants.intersection(month2_merchants)

                    # Group by merchant and compare
                    if len(common_merchants) > 0:
                        m1_by_merchant = category_month1.groupby('Description')['Amount'].sum()
                        m2_by_merchant = category_month2.groupby('Description')['Amount'].sum()

                        # Combine and calculate differences
                        merchant_comparison = pd.DataFrame({
                            'Month1': m1_by_merchant,
                            'Month2': m2_by_merchant
                        }).fillna(0)
                        merchant_comparison['Difference'] = merchant_comparison['Month2'] - merchant_comparison['Month1']
                        merchant_comparison = merchant_comparison.sort_values('Difference', ascending=False)

                        # Show top merchants with increased spending
                        increased_merchants = merchant_comparison[merchant_comparison['Difference'] > 0]
                        if not increased_merchants.empty:
                            st.markdown("**Merchants with increased spending:**")
                            for merchant, values in increased_merchants.head(3).iterrows():
                                st.markdown(f"- **{merchant}**: ${values['Month1']:.2f} → ${values['Month2']:.2f} " +
                                          f"(+${values['Difference']:.2f})")

                    # Show new merchants
                    if new_merchants:
                        st.markdown("**New spending at:**")
                        new_merchants_df = category_month2[category_month2['Description'].str.lower().isin(new_merchants)]
                        new_merchants_grouped = new_merchants_df.groupby('Description')['Amount'].sum().sort_values(ascending=False)

                        for merchant, amount in new_merchants_grouped.head(3).items():
                            st.markdown(f"- **{merchant}**: ${amount:.2f}")

                    # Show transaction frequency
                    m1_count = len(category_month1)
                    m2_count = len(category_month2)
                    st.markdown(f"**Transaction frequency**: {m1_count} → {m2_count} transactions")

            # Show insights for top decreases
            if not significant_decreases.empty:
                st.markdown("### 📉 Notable Spending Decreases")

                for category, row in significant_decreases.head(3).iterrows():
                    st.markdown(f"#### {category}: ${-row['Difference']:.2f} decrease ({row['% Change']:.1f}%)")

                    # Get transactions for this category in the two months
                    category_month2 = month2_data[month2_data['Enhanced_Category'] == category].sort_values('Amount', ascending=False)
                    category_month1 = month1_data[month1_data['Enhanced_Category'] == category].sort_values('Amount', ascending=False)

                    # Find merchants that disappeared
                    month1_merchants = set(category_month1['Description'].str.lower())
                    month2_merchants = set(category_month2['Description'].str.lower())
                    disappeared_merchants = month1_merchants - month2_merchants

                    # Find merchants with decreased spending
                    common_merchants = month1_merchants.intersection(month2_merchants)

                    # Group by merchant and compare
                    if len(common_merchants) > 0:
                        m1_by_merchant = category_month1.groupby('Description')['Amount'].sum()
                        m2_by_merchant = category_month2.groupby('Description')['Amount'].sum()

                        # Combine and calculate differences
                        merchant_comparison = pd.DataFrame({
                            'Month1': m1_by_merchant,
                            'Month2': m2_by_merchant
                        }).fillna(0)
                        merchant_comparison['Difference'] = merchant_comparison['Month2'] - merchant_comparison['Month1']
                        merchant_comparison = merchant_comparison.sort_values('Difference', ascending=True)

                        # Show top merchants with decreased spending
                        decreased_merchants = merchant_comparison[merchant_comparison['Difference'] < 0]
                        if not decreased_merchants.empty:
                            st.markdown("**Merchants with decreased spending:**")
                            for merchant, values in decreased_merchants.head(3).iterrows():
                                st.markdown(f"- **{merchant}**: ${values['Month1']:.2f} → ${values['Month2']:.2f} " +
                                          f"(${values['Difference']:.2f})")

                    # Show disappeared merchants
                    if disappeared_merchants:
                        st.markdown("**No longer spending at:**")
                        disappeared_merchants_df = category_month1[category_month1['Description'].str.lower().isin(disappeared_merchants)]
                        disappeared_merchants_grouped = disappeared_merchants_df.groupby('Description')['Amount'].sum().sort_values(ascending=False)

                        for merchant, amount in disappeared_merchants_grouped.head(3).items():
                            st.markdown(f"- **{merchant}**: ${amount:.2f} in {compare_month1}")

                    # Show transaction frequency
                    m1_count = len(category_month1)
                    m2_count = len(category_month2)
                    st.markdown(f"**Transaction frequency**: {m1_count} → {m2_count} transactions")

            # Monthly category trends
            st.subheader("Category Spending Trends")

            # Allow selecting a specific category to see its trend
            selected_trend_category = st.selectbox(
                "Select a Category to View Trend",
                options=compare_categories
            )

            # Create a monthly trend for the selected category
            category_trend_data = expenses_only[expenses_only['Enhanced_Category'] == selected_trend_category]
            monthly_category_trend = category_trend_data.groupby('Month')['Amount'].sum().reset_index()
            monthly_category_trend['Month'] = monthly_category_trend['Month'].astype(str)

            fig_trend = px.line(
                monthly_category_trend,
                x='Month',
                y='Amount',
                markers=True,
                title=f'Monthly Spending Trend: {selected_trend_category}',
                height=400
            )

            fig_trend.update_layout(
                xaxis_title='Month',
                yaxis_title='Amount ($)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_trend, use_container_width=True)

        else:
            st.info("You need at least two months of transaction data for comparison. Please add more transaction data.")
    with tab7:
        tab7a, tab7b, tab7c = st.tabs([
            "Transaction Details",
            "Category Management",
            "Transactions Files Upload"
        ])

        with tab7a:
            st.header("Transaction Details")

            # Filters for the transaction table
            col1, col2, col3 = st.columns(3)

            with col1:
                filter_bank = st.multiselect(
                    "Filter by Bank",
                    options=enhanced_transactions['Bank'].unique(),
                    default=enhanced_transactions['Bank'].unique()
                )

            with col2:
                filter_category = st.multiselect(
                    "Filter by Category",
                    options=enhanced_transactions['Enhanced_Category'].unique(),
                    default=enhanced_transactions['Enhanced_Category'].unique()
                )

            with col3:
                filter_amount = st.slider(
                    "Amount Range",
                    min_value=float(enhanced_transactions['Amount'].min()),
                    max_value=float(enhanced_transactions['Amount'].max()),
                    value=(float(enhanced_transactions['Amount'].min()), float(enhanced_transactions['Amount'].max()))
                )

            # Apply filters
            filtered_transactions = enhanced_transactions[
                (enhanced_transactions['Bank'].isin(filter_bank)) &
                (enhanced_transactions['Enhanced_Category'].isin(filter_category)) &
                (enhanced_transactions['Amount'] >= filter_amount[0]) &
                (enhanced_transactions['Amount'] <= filter_amount[1])
            ]

            # Search field
            search_term = st.text_input("Search Transactions", "")
            if search_term:
                filtered_transactions = filtered_transactions[
                    filtered_transactions['Description'].str.contains(search_term, case=False) |
                    filtered_transactions['Category'].str.contains(search_term, case=False)
                ]

            # Display the filtered transactions
            st.dataframe(
                filtered_transactions[['Date', 'Description', 'Category', 'Enhanced_Category', 'Sub_Category', 'Amount', 'Bank']],
                use_container_width=True,
                hide_index=True
            )

            # Export functionality
            if not filtered_transactions.empty:
                csv = filtered_transactions.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Transactions as CSV",
                    data=csv,
                    file_name="filtered_transactions.csv",
                    mime="text/csv",
                )

        with tab7b:
            st.header("Category Management")

            st.markdown("""
            This tab allows you to manage how transactions are categorized, including:
            - Creating pattern-based rules for automatic categorization
            - Managing your transaction categories and subcategories
            - Viewing which rules are applied to your transactions
            """)

            # Show tabs for different management functions
            cat_tab1, cat_tab2, cat_tab3 = st.tabs([
                "Vendor Rules",
                "Manual Categories",
                "Test Categorization"
            ])

            with cat_tab1:
                st.subheader("Vendor Pattern Rules")
                st.markdown("""
                These rules automatically categorize transactions based on patterns in the description.
                For example, all transactions containing "LYFT" can be categorized as Transportation > Rideshare.
                """)

                # Load existing rules
                vendor_rules = load_vendor_rules()

                # Display existing rules
                if not vendor_rules.empty:
                    st.dataframe(
                        vendor_rules,
                        use_container_width=True,
                        hide_index=True
                    )

                # Form to add new rule
                st.subheader("Add New Vendor Rule")

                with st.form("add_vendor_rule"):
                    pattern = st.text_input("Vendor Pattern (e.g., LYFT)", "")
                    cat_col1, cat_col2 = st.columns(2)

                    with cat_col1:
                        category = st.text_input("Category", "")

                    with cat_col2:
                        subcategory = st.text_input("Subcategory", "")

                    submitted = st.form_submit_button("Add Rule")

                    if submitted and pattern and category:
                        # Create new rule dataframe
                        new_rule = pd.DataFrame({
                            'vendor_pattern': [pattern],
                            'category': [category],
                            'subcategory': [subcategory if subcategory else 'General']
                        })

                        # Combine with existing rules
                        if vendor_rules.empty:
                            updated_rules = new_rule
                        else:
                            # Check for duplicates
                            if pattern.upper() in vendor_rules['vendor_pattern'].str.upper().values:
                                st.error(f"A rule for '{pattern}' already exists!")
                            else:
                                updated_rules = pd.concat([vendor_rules, new_rule], ignore_index=True)

                        # Save updated rules
                        try:
                            # Convert dataframe to CSV string for S3 storage
                            csv_content = updated_rules.to_csv(index=False)

                            # Save to S3 in shared core_app_data folder
                            from s3_utils import write_file_to_s3
                            write_file_to_s3("shared", "core_app_data/vendor_rules.csv", csv_content)

                            # Also update local file for fallback
                            rules_file = os.path.join('core_app_data', 'vendor_rules.csv')
                            if os.path.exists(os.path.dirname(rules_file)):  # Make sure directory exists
                                updated_rules.to_csv(rules_file, index=False)

                            st.success(f"Added rule for '{pattern}'")
                            st.rerun()  # Refresh to show the new rule
                        except Exception as e:
                            st.error(f"Error saving rule: {e}")

                # Feature to find common patterns in uncategorized transactions
                st.subheader("Find Common Patterns in Uncategorized Transactions")

                if st.button("Analyze Transactions for Common Vendors"):
                    # Get transactions that don't have custom categories
                    uncategorized = enhanced_transactions[
                        enhanced_transactions['New_Category'].isna()
                    ]

                    if not uncategorized.empty:
                        # Extract vendor names from descriptions using common patterns
                        vendors = []
                        for desc in uncategorized['Description']:
                            # Try to extract vendor name - usually the first word or first few characters
                            parts = re.split(r'[^A-Za-z0-9]', desc)
                            parts = [p for p in parts if p]  # Remove empty strings
                            if parts:
                                vendors.append(parts[0])

                        # Count vendors
                        vendor_counts = pd.Series(vendors).value_counts()
                        top_vendors = vendor_counts.head(10)

                        # Show top uncategorized vendors
                        st.write("Top 10 vendors without specific rules:")

                        for vendor, count in top_vendors.items():
                            st.write(f"- **{vendor}**: {count} transactions")

                        st.info("You can create rules for these vendors to automatically categorize similar transactions.")
                    else:
                        st.info("All transactions have been categorized!")

            with cat_tab2:
                st.subheader("Manual Category Assignments")
                st.markdown("""
                These are exact matches for specific transaction descriptions.
                They override any pattern-based rules.
                """)

                # Load existing manual categories
                manual_categories = load_category_mapping()

                if not manual_categories.empty:
                    # Display table with manual mappings
                    st.dataframe(
                        manual_categories,
                        use_container_width=True,
                        hide_index=True
                    )

                # Form to add new manual mapping
                st.subheader("Add New Manual Mapping")

                # First get a list of unique descriptions for selection
                all_descriptions = sorted(enhanced_transactions['Description'].unique())

                with st.form("add_manual_category"):
                    selected_desc = st.selectbox("Select transaction description:", all_descriptions)
                    cat_col1, cat_col2 = st.columns(2)

                    with cat_col1:
                        manual_category = st.text_input("Category:", "")

                    with cat_col2:
                        manual_subcategory = st.text_input("Subcategory:", "")

                    manual_submitted = st.form_submit_button("Add Manual Mapping")

                    if manual_submitted and selected_desc and manual_category:
                        # Create new mapping dataframe
                        new_mapping = pd.DataFrame({
                            'Description': [selected_desc],
                            'Category': [''], # Original category from bank (leave blank)
                            'Changed_Sub-Category': [manual_subcategory if manual_subcategory else "General"],
                            'Changed_Category': [manual_category]
                        })

                        # Combine with existing mappings
                        if manual_categories.empty:
                            updated_mappings = new_mapping
                        else:
                            # Check for duplicates
                            if selected_desc in manual_categories['Description'].values:
                                st.error(f"A mapping for '{selected_desc}' already exists!")
                            else:
                                updated_mappings = pd.concat([manual_categories, new_mapping], ignore_index=True)

                        # Save updated mappings
                        try:
                            categories_file = os.path.join('user_transactions_data', 'Spend_categories.csv')
                            updated_mappings.to_csv(categories_file, index=False)
                            st.success(f"Added mapping for '{selected_desc}'")
                            st.rerun()  # Refresh to show the new mapping
                        except Exception as e:
                            st.error(f"Error saving mapping: {e}")

        with cat_tab3:
            st.subheader("Test Categorization")
            st.markdown("""
            This tool helps you test how a transaction would be categorized based on your current rules.
            Enter a transaction description to see which category and subcategory would be assigned.
            """)

            test_desc = st.text_input("Enter a transaction description:", "LYFT *1 RIDE 03-25")

            if st.button("Test Categorization") and test_desc:
                # Create a test dataframe with the single description
                test_df = pd.DataFrame({
                    'Description': [test_desc],
                    'Category': ['Uncategorized'],
                    'Amount': [10.0],
                    'Date': [datetime.now()],
                    'Bank': ['Test'],
                    'Is_Return': [False]
                })

                # Apply our categorization logic
                categorized_test = categorize_transactions(test_df)
                vendor_categorized = apply_vendor_rules(categorized_test)
                final_test = apply_manual_categories(vendor_categorized)

                # Display results
                st.subheader("Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Original Category", categorized_test['Category Group'].iloc[0])

                with col2:
                    if final_test['New_Category'].iloc[0]:
                        st.metric("Enhanced Category", final_test['New_Category'].iloc[0])
                    else:
                        st.metric("Enhanced Category", "(No rule match)")

                with col3:
                    if final_test['Sub_Category'].iloc[0]:
                        st.metric("Subcategory", final_test['Sub_Category'].iloc[0])
                    else:
                        st.metric("Subcategory", "(None)")

                # Show which rule was applied
                if final_test['New_Category'].iloc[0]:
                    st.success("✅ Transaction was successfully categorized")

                    # Determine if it was a vendor rule or manual category
                    vendor_rules = load_vendor_rules()
                    manual_cats = load_category_mapping()

                    if not manual_cats.empty and test_desc in manual_cats['Description'].values:
                        st.info("✳️ Categorized by: **Manual category mapping**")
                    elif not vendor_rules.empty:
                        matched_rule = None
                        for _, rule in vendor_rules.iterrows():
                            if rule['vendor_pattern'].upper() in test_desc.upper():
                                matched_rule = rule['vendor_pattern']
                                break

                        if matched_rule:
                            st.info(f"✳️ Categorized by: **Vendor rule** matching '{matched_rule}'")
                else:
                    st.warning("⚠️ No specific rule matched this transaction")
                    st.info("This transaction would use the default category from your bank.")

                # Show similar transactions in your data
                st.subheader("Similar Transactions in Your Data")

                # Find transactions with similar descriptions
                if not enhanced_transactions.empty:
                    similar_mask = enhanced_transactions['Description'].str.contains(
                        '|'.join(test_desc.split()[:1]),
                        case=False,
                        na=False
                    )
                    similar_trans = enhanced_transactions[similar_mask].head(5)

                    if not similar_trans.empty:
                        st.write("These existing transactions have similar descriptions:")

                        # Show similar transactions
                        similar_display = similar_trans[['Description', 'Enhanced_Category', 'Sub_Category', 'Amount', 'Date']]
                        similar_display = similar_display.rename(columns={'Enhanced_Category': 'Category'})
                        st.dataframe(similar_display, use_container_width=True, hide_index=True)
                    else:
                        st.info("No similar transactions found in your data.")
        with tab7c:
            st.header("Transactions Files Upload")

            st.markdown("""
            Upload your transaction CSV files to the appropriate bank folder.
            This will make your transaction data available for analysis in the application.
            """)

            # Get list of existing folders in user_transactions_data directory via S3
            username = st.session_state.username
            data_dir = 'user_transactions_data'

            try:
                # List all files (keys) in the user's transaction data directory
                _, all_s3_files = list_files_in_user_folder(data_dir, username)

                # Extract unique folder names from file paths
                folders = set()
                for file_path in all_s3_files:
                    # Expect relative paths like 'discover/file.csv'
                    parts = file_path.split('/')
                    if len(parts) > 1:  # has a subfolder
                        folders.add(parts[0])
                folders = sorted(list(folders))
            except Exception as e:
                st.error(f"Error retrieving folders: {str(e)}")
                folders = []

            # Option to create a new folder
            create_new_folder = st.checkbox("Create a new bank folder")

            if create_new_folder:
                new_folder_name = st.text_input("Enter new bank/folder name:")
                if new_folder_name:
                    if new_folder_name not in folders and new_folder_name.strip() != '':
                        if st.button("Create Folder"):
                            try:
                                # Create placeholder file to realize the folder in S3
                                from s3_utils import write_file_to_s3
                                write_file_to_s3(username, f"{new_folder_name}/.placeholder", "")
                                st.success(f"Successfully created folder: {new_folder_name}")
                                folders.append(new_folder_name)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error creating folder: {str(e)}")
                    else:
                        st.warning("Folder already exists or invalid name provided.")

            # Folder selection for upload
            if folders:
                selected_folder = st.selectbox(
                    "Select folder for upload:",
                    options=folders,
                    format_func=lambda x: x.capitalize()
                )
            else:
                st.warning("No folders found. Please create a folder first.")
                selected_folder = None

            # File uploader
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:
                if not selected_folder:
                    st.warning("Please select or create a folder before uploading a file.")
                else:
                    file_details = {
                        "Filename": uploaded_file.name,
                        "File size": f"{uploaded_file.size / 1024:.2f} KB"
                    }
                    st.write(file_details)

                    # Preview
                    try:

                        df_preview = pd.read_csv(uploaded_file)
                        if selected_folder.lower() == "bilt":
                            df_preview = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")), header=None,
                                                     names=["Transaction Date",
                                                                                          "Debit/Credit",
                                                                                          "Reference Number",
                                                                                          "Card No.",
                                                                                          "Description"])

                        st.write("Preview of the uploaded file:")
                        st.dataframe(df_preview.head(5), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error reading the CSV file: {str(e)}")
                        st.stop()

                    # Check if file exists in S3
                    from s3_utils import s3_file_exists
                    s3_key = f"{username}/{data_dir}/{selected_folder}/{uploaded_file.name}"
                    file_exists = s3_file_exists(s3_key)

                    if file_exists:
                        overwrite = st.checkbox("A file with this name already exists. Overwrite?")
                        if not overwrite:
                            st.stop()

                    if st.button("Save File"):
                        try:
                            from s3_utils import write_file_to_s3
                            # write_file_to_s3 expects (username, relative_path, content_string)
                            # Relative path inside user_transactions_data/<folder>/<file>
                            content_str = uploaded_file.getvalue().decode('utf-8')
                            # If Bilt folder, enforce headers when saving
                            if selected_folder and selected_folder.lower() == "bilt":
                                try:
                                    # df_preview already has correct headers assigned above
                                    content_str = df_preview.to_csv(index=False)
                                except Exception:
                                    # Fallback: re-read and assign headers explicitly
                                    bilt_df = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")), header=None,
                                                          names=["Transaction Date",
                                                                 "Debit/Credit",
                                                                 "Reference Number",
                                                                 "Card No.",
                                                                 "Description"])
                                    content_str = bilt_df.to_csv(index=False)
                            write_file_to_s3(username, f"{selected_folder}/{uploaded_file.name}", content_str)
                            st.success(f"Successfully saved {uploaded_file.name} to {selected_folder} folder in S3!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving file to S3: {str(e)}")

            # Display existing files in folders from S3
            st.subheader("Existing Transaction Files")
            if not folders:
                st.info("No folders available.")
            else:
                for folder in folders:
                    with st.expander(f"{folder.capitalize()} Files"):
                        # List files in this specific folder
                        files, rel_paths = list_files_in_user_folder(f"user_transactions_data/{folder}", username)
                        # Filter out placeholder
                        display_files = [f for f in files if f != '.placeholder']
                        if display_files:
                            st.write(display_files)
                        else:
                            st.info(f"No files in {folder} folder.")
