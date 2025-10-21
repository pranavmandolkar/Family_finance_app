import streamlit as st
import time
from users import authenticate_user, get_users, change_password, add_user, delete_user, is_admin

def login_page():
    st.set_page_config(
        page_title="Family Expense Tracker - Login",
        page_icon="🔒",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Title and description
    st.title("🔒 Family Expense Tracker")
    st.subheader("Login to access your financial data")

    # Check if already logged in
    if "authenticated" in st.session_state and st.session_state.authenticated:
        st.success(f"Logged in as {st.session_state.username}")

        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.is_admin = False
            st.rerun()

        # User settings section
        with st.expander("User Settings"):
            # Change password form
            st.subheader("Change Password")
            with st.form("change_password_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_new_password = st.text_input("Confirm New Password", type="password")
                submit_password = st.form_submit_button("Change Password")

                if submit_password:
                    # Verify current password first
                    if authenticate_user(st.session_state.username, current_password):
                        if new_password == confirm_new_password:
                            if new_password.strip():  # Ensure password is not empty
                                success, message = change_password(st.session_state.username, new_password)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                            else:
                                st.error("New password cannot be empty")
                        else:
                            st.error("New passwords don't match")
                    else:
                        st.error("Current password is incorrect")

        # Admin section
        if st.session_state.get("is_admin", False):
            with st.expander("Admin Settings", expanded=False):
                st.subheader("Manage Users")

                # User list
                users = get_users()
                st.write("Current users:")
                user_data = []
                for username, user_info in users.items():
                    user_data.append({
                        "Username": username,
                        "Name": user_info.get("name", ""),
                        "Admin": "Yes" if user_info.get("is_admin", False) else "No"
                    })

                st.table(user_data)

                # Add new user form
                st.subheader("Add New User")
                with st.form("add_user_form"):
                    new_username = st.text_input("Username").lower().strip()
                    new_password = st.text_input("Password", type="password")
                    new_name = st.text_input("Full Name")
                    new_is_admin = st.checkbox("Admin privileges")

                    add_user_submit = st.form_submit_button("Add User")

                    if add_user_submit:
                        if new_username and new_password and new_name:
                            success, message = add_user(new_username, new_password, new_name, new_is_admin)
                            if success:
                                st.success(message)
                                time.sleep(1)
                                st.rerun()  # Refresh to show updated user list
                            else:
                                st.error(message)
                        else:
                            st.error("All fields are required")

                # Delete user
                st.subheader("Delete User")

                # Don't allow deleting the logged-in user or the last admin
                deletable_users = [user for user in users.keys()
                                  if user != st.session_state.username and
                                  not (users[user].get("is_admin", False) and
                                       sum(1 for u in users.values() if u.get("is_admin", False)) <= 1)]

                if deletable_users:
                    with st.form("delete_user_form"):
                        user_to_delete = st.selectbox("Select User", options=deletable_users)
                        delete_confirm = st.text_input("Type DELETE to confirm")

                        delete_submit = st.form_submit_button("Delete User")

                        if delete_submit:
                            if delete_confirm == "DELETE":
                                success, message = delete_user(user_to_delete)
                                if success:
                                    st.success(message)
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(message)
                            else:
                                st.error("Please type DELETE to confirm")
                else:
                    st.info("No users available to delete (can't delete yourself or the last admin)")

        # Button to continue to the app
        st.markdown("---")
        if st.button("Continue to Expense Tracker", type="primary", use_container_width=True):
            st.session_state.show_main_app = True
            st.rerun()

    else:
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username").lower().strip()
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if username and password:
                    if authenticate_user(username, password):
                        # Set authenticated flag in session state
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.is_admin = is_admin(username)
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")

        # Default credentials hint
        st.info("Default admin credentials: username = 'admin', password = 'admin'. Remember to change this password after your first login!")
