import streamlit as st

# Import login functionality
from login import login_page

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
# Import required libraries for the main app
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

# Set page configuration
st.set_page_config(
    page_title="Family Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add sidebar with logout functionality
with st.sidebar:
    st.title("Family Expense Tracker")
    st.write(f"Welcome, {st.session_state.username}")

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
