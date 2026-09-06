import os

import streamlit as st
from supabase import Client, create_client
from supabase.client import ClientOptions


# ============================================================
# SUPABASE CLIENT
# ============================================================

def get_supabase_client() -> Client:
    """Create an isolated Supabase client for the current app request."""

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured in Streamlit secrets.")

    if not supabase_key:
        raise RuntimeError("SUPABASE_KEY is not configured in Streamlit secrets.")

    options = ClientOptions(flow_type="pkce")
    return create_client(supabase_url, supabase_key, options=options)


def get_current_user():
    """Return the user stored in this Streamlit browser session."""

    return st.session_state.get("supabase_user")


def _friendly_auth_error(exc: Exception) -> str:
    """Turn common Supabase auth errors into user-friendly messages."""

    message = str(exc).strip()
    lowered = message.lower()

    if "invalid login credentials" in lowered:
        return "Invalid email or password."
    if "email not confirmed" in lowered:
        return "Please confirm your email address before logging in."
    if "user already registered" in lowered:
        return "An account with this email already exists. Please log in."
    if "password should be at least" in lowered:
        return "Password must be at least 8 characters."
    if "rate limit" in lowered or "too many requests" in lowered:
        return "Too many authentication attempts. Please wait a moment and try again."

    return message or "Authentication failed. Please try again."


def clear_auth_state() -> None:
    """Clear only this browser session's authentication state."""

    for key in (
        "supabase_user",
        "password_recovery",
        "recovery_code",
    ):
        st.session_state.pop(key, None)


def sign_in(email, password):
    """Sign in using Supabase email/password authentication."""

    client = get_supabase_client()
    response = client.auth.sign_in_with_password(
        {
            "email": email.strip(),
            "password": password,
        }
    )

    user = response.user
    if user is None:
        raise RuntimeError("Login did not return an authenticated user.")

    st.session_state["supabase_user"] = user
    return user


def sign_up(email, password):
    """Create a new Supabase Auth user."""

    client = get_supabase_client()
    response = client.auth.sign_up(
        {
            "email": email.strip(),
            "password": password,
        }
    )

    user = response.user

    if user is not None and response.session is not None:
        st.session_state["supabase_user"] = user

    return user, response.session is not None


def reset_password(email):
    """Send a password recovery email."""

    client = get_supabase_client()
    clean_email = email.strip()

    if clean_email.startswith("[") and "](" in clean_email:
        clean_email = clean_email.split("](", 1)[0].lstrip("[").strip()

    redirect_url = os.getenv("SUPABASE_REDIRECT_URL", "http://localhost:8501")

    return client.auth.reset_password_for_email(
        clean_email,
        options={"redirect_to": redirect_url},
    )


def update_password(new_password):
    """Update the password for the authenticated recovery session."""

    client = get_supabase_client()
    response = client.auth.update_user({"password": new_password})
    return response.user


def sign_out():
    """Sign out the current user and clear this browser session."""

    try:
        client = get_supabase_client()
        client.auth.sign_out()
    finally:
        clear_auth_state()


def handle_password_recovery():
    """Process a Supabase PKCE password-recovery callback."""

    code = st.query_params.get("code")
    if not code:
        return False

    if st.session_state.get("recovery_code") == code:
        return st.session_state.get("password_recovery", False)

    try:
        client = get_supabase_client()
        response = client.auth.exchange_code_for_session(
            {"auth_code": code}
        )
        user = response.user

        if user is None:
            raise RuntimeError("Recovery session could not be created.")

        st.session_state["supabase_user"] = user
        st.session_state["password_recovery"] = True
        st.session_state["recovery_code"] = code
        st.query_params.clear()
        return True

    except Exception as exc:
        clear_auth_state()
        st.error(
            "Password recovery link could not be processed: "
            + _friendly_auth_error(exc)
        )
        return False


def render_password_recovery():
    """Render the set-new-password page."""

    st.title("Rocket Guardian AI")
    st.subheader("Set New Password")
    st.write("Enter a new password for your account.")

    new_password = st.text_input(
        "New Password",
        type="password",
        key="recovery_new_password",
    )
    confirm_password = st.text_input(
        "Confirm New Password",
        type="password",
        key="recovery_confirm_password",
    )

    if st.button(
        "Update Password",
        type="primary",
        use_container_width=True,
    ):
        if len(new_password) < 8:
            st.error("Password must be at least 8 characters.")
            return

        if new_password != confirm_password:
            st.error("Passwords do not match.")
            return

        try:
            update_password(new_password)
            st.success("Password updated successfully. Please continue to the application.")
            clear_auth_state()
            st.rerun()
        except Exception as exc:
            st.error("Password update failed: " + _friendly_auth_error(exc))


def render_login():
    """Render login and account creation UI."""

    if handle_password_recovery():
        render_password_recovery()
        return False

    st.title("Rocket Guardian AI")
    st.subheader("Customer Login")

    tab_login, tab_signup = st.tabs(["Login", "Create Account"])

    with tab_login:
        st.text_input("Email", key="login_email")
        st.text_input("Password", type="password", key="login_password")

        if st.button(
            "Login",
            type="primary",
            use_container_width=True,
        ):
            login_email = st.session_state.get("login_email", "").strip()
            login_password = st.session_state.get("login_password", "")

            if not login_email:
                st.error("Please enter your email.")
                return False
            if not login_password:
                st.error("Please enter your password.")
                return False

            try:
                sign_in(login_email, login_password)
                st.success("Login successful.")
                st.rerun()
            except Exception as exc:
                clear_auth_state()
                st.error("Login failed: " + _friendly_auth_error(exc))

        st.divider()

        if st.button("Forgot Password?", use_container_width=True):
            reset_email = st.session_state.get("login_email", "").strip()

            if not reset_email:
                st.error("Please enter your email first.")
            else:
                try:
                    reset_password(reset_email)
                    st.success(
                        "Password reset email sent. Check your email and follow the reset link."
                    )
                except Exception as exc:
                    st.error(
                        "Password reset failed: "
                        + _friendly_auth_error(exc)
                    )

    with tab_signup:
        st.text_input("Email", key="signup_email")
        st.text_input("Password", type="password", key="signup_password")

        if st.button("Create Account", use_container_width=True):
            signup_email = st.session_state.get("signup_email", "").strip()
            signup_password = st.session_state.get("signup_password", "")

            if not signup_email:
                st.error("Please enter your email.")
                return False

            if len(signup_password) < 8:
                st.error("Password must be at least 8 characters.")
                return False

            try:
                user, has_session = sign_up(signup_email, signup_password)

                if user is None:
                    raise RuntimeError("Account creation did not return a user.")

                if has_session:
                    st.success("Account created successfully.")
                    st.rerun()
                else:
                    st.success(
                        "Account created. Please confirm your email, then log in."
                    )
            except Exception as exc:
                st.error(
                    "Account creation failed: "
                    + _friendly_auth_error(exc)
                )

    return False
