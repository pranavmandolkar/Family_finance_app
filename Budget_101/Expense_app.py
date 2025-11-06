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
import os
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import calendar
import numpy as np
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
    """Load all transaction data from different banks."""
    all_data = []

    # Process Discover data
    discover_files = list_files_in_user_folder("user_transactions_data/discover", st.session_state.username)
    for file in discover_files:
        df = load_discover_data(st.session_state.username, file)
        all_data.append(df)

    # Process Capital One data
    capital_one_files = list_files_in_user_folder("user_transactions_data/Venture_X", st.session_state.username)
    for file in capital_one_files:
        df = load_capital_one_data(st.session_state.username, file)
        all_data.append(df)

    # Process Saver One data
    saver_one_files = list_files_in_user_folder("user_transactions_data/Saver_one", st.session_state.username)
    for file in saver_one_files:
        df = load_saver_one_data(st.session_state.username, file)
        all_data.append(df)

    # Process Bilt data
    bilt_files = list_files_in_user_folder("user_transactions_data/bilt", st.session_state.username)
    for file in bilt_files:
        df = load_bilt_data(st.session_state.username, file)
        all_data.append(df)

    # Combine all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Keep only essential columns for analysis
        columns_to_keep = ['Date', 'Description', 'Category', 'Amount', 'Bank', 'Is_Return']
        combined_data = combined_data[columns_to_keep]

        # Post-processing to identify refunds and credits across different rows
        combined_data = identify_related_refunds(combined_data)

        # Sort by date
        combined_data = combined_data.sort_values('Date')
        return combined_data
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
            from io import StringIO
            df = pd.read_csv(StringIO(file_content))

            # Add payment_type column if it doesn't exist
            if 'payment_type' not in df.columns:
                df['payment_type'] = 'Fixed'  # Set default value for existing records

            if 'start_month' in df.columns and not df.empty and not df['start_month'].empty:
                # Convert to datetime first, handling any format
                df['start_month'] = pd.to_datetime(df['start_month'], format='%Y-%m-%d', errors='coerce')
                # Then convert to date objects
                df['start_month'] = df['start_month'].dt.date

            if 'end_month' in df.columns and not df.empty and not df['end_month'].empty:
                # Do the same for end_month
                df['end_month'] = pd.to_datetime(df['end_month'], format='%Y-%m-%d', errors='coerce').dt.date

            # Add a unique ID if not present
            if 'id' not in df.columns:
                df['id'] = [f"payment_{i}" for i in range(len(df))]
                save_recurring_payments(df)
            return df
    except Exception as e:
        print(f"Error loading recurring payments data from S3: {str(e)}")

    # If we reach here, either the file doesn't exist or there was an error
    return pd.DataFrame(columns=['description', 'amount', 'frequency', 'category', 'start_month', 'end_month', 'sub_category', 'payment_type', 'id'])

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

    # Make sure payment_type exists
    if 'payment_type' not in save_df.columns:
        save_df['payment_type'] = 'Fixed'

    # Debug info for saving
    if 'start_month' in save_df.columns:
        # Convert to string format YYYY-MM-DD
        def format_date(x):
            try:
                if isinstance(x, (datetime, date)):
                    return x.strftime('%Y-%m-%d')
                elif pd.notna(x):
                    return pd.to_datetime(x).strftime('%Y-%m-%d')
                return None
            except:
                return None

        save_df['start_month'] = save_df['start_month'].apply(format_date)

    if 'end_month' in save_df.columns:
        # Handle end_month the same way
        save_df['end_month'] = save_df['end_month'].apply(format_date)
        # Replace None with empty string
        save_df['end_month'] = save_df['end_month'].fillna('')

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
    st.header("Quick Actions")

    # Load income and payment data for sidebar display
    income_data = load_income_data()
    recurring_payments = load_recurring_payments()

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
    if not recurring_payments.empty:
        # Use calculate_active_monthly_payments which filters based on current date
        total_monthly_payments = calculate_active_monthly_payments(recurring_payments, current_date)
        st.metric("Total Monthly Fixed Expenses", f"${total_monthly_payments:.2f}")

        # Show count of active payments
        if 'start_month' in recurring_payments.columns and 'end_month' in recurring_payments.columns:
            payments_copy = recurring_payments.copy()
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

    # Show estimated remaining budget if both income and payments exist
    if not income_data.empty:
        # Calculate actual remaining budget including variable expenses
        remaining_budget = total_monthly_income - total_monthly_payments - total_variable_expenses
        st.metric("Remaining Budget (after all expenses)",
                 f"${remaining_budget:.2f}",
                 delta=None)

        # Show percent of income spent
        if total_monthly_income > 0:
            spending_percent = ((total_monthly_payments + total_variable_expenses) / total_monthly_income) * 100
            savings_percent = 100 - spending_percent
            st.caption(f"You've spent {spending_percent:.1f}% of your income this month")

            # Add a simple progress bar
            st.progress(min(spending_percent/100, 1.0), f"Saving {savings_percent:.1f}%")

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
                            st.session_state['edit_start_month'] = pd.to_datetime(selected_income_data['start_month']).date()

                        # Check if end_month exists and is not NaN/None
                        has_end_date = pd.notna(selected_income_data['end_month']) and selected_income_data['end_month'] not in ['None', '']
                        st.session_state.end_date_enabled = has_end_date

                        if has_end_date:
                            st.session_state['edit_end_month'] = pd.to_datetime(selected_income_data['end_month']).date()

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

                    st.success(f"Updated income source: {income_source}")

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
                        st.write(selected_payment_data)
                        # Store the values in session state to pre-fill the form
                        st.session_state['edit_payment'] = True
                        st.session_state['edit_payment_id'] = selected_payment_id
                        st.session_state['edit_description'] = selected_payment_data['description']
                        st.session_state['edit_amount'] = selected_payment_data['amount']
                        st.session_state['edit_frequency'] = selected_payment_data['frequency']
                        st.session_state['edit_category'] = selected_payment_data['category']
                        st.session_state['edit_sub_category'] = selected_payment_data['sub_category']
                        st.session_state['edit_payment_type'] = selected_payment_data['payment_type']

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
            amount_b = st.number_input("Amount", min_value=0.0, format='%.2f', value=float(st.session_state.get('edit_budget_amount', 0.0)))
            freq_opts = ["Monthly","Bi-weekly","Weekly","Yearly","Quarterly"]
            frequency_b = st.selectbox("Frequency", freq_opts, index=freq_opts.index(st.session_state.get('edit_budget_frequency','Monthly')) if st.session_state.get('edit_budget_frequency','Monthly') in freq_opts else 0)
            category_b = st.text_input("Category", value=st.session_state.get('edit_budget_category',''))
            sub_category_b = st.text_input("Sub-category", value=st.session_state.get('edit_budget_sub_category',''))
            payment_type_b = st.selectbox("Payment Type", ["Fixed","Variable"], index=["Fixed","Variable"].index(st.session_state.get('edit_budget_payment_type','Fixed')))
            # Dates
            start_default_b = st.session_state.get('edit_budget_start_month', datetime.now().replace(day=1).date())
            start_month_b = st.date_input("Start Month", value=start_default_b, key="budget_start_month_input")
            end_month_b = None
            if budget_end_month_toggle:
                end_default_b = st.session_state.get('edit_budget_end_month', datetime.now().replace(day=1).date())
                end_month_b = st.date_input("End Month", value=end_default_b, key="budget_end_month_input")
            clear_budget = st.form_submit_button("Clear Form")
            if clear_budget:
                for k in ['edit_budget','edit_budget_id','edit_budget_description','edit_budget_amount','edit_budget_frequency','edit_budget_category','edit_budget_sub_category','edit_budget_payment_type','edit_budget_start_month','edit_budget_end_month']:
                    if k in st.session_state:
                        st.session_state.pop(k)
                st.session_state.budget_end_date_enabled = False
                st.rerun()
            submit_budget = st.form_submit_button("Save Budget Item")
            if submit_budget and description_b and amount_b > 0:
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
                save_monthly_budget(monthly_budget_df)
                st.rerun()
