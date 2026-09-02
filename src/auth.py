import os

import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    """
    Create and cache the Supabase client.
    """

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured.")

    if not supabase_key:
        raise RuntimeError("SUPABASE_KEY is not configured.")

    return create_client(
        supabase_url,
        supabase_key,
    )


def get_current_user():
    """
    Return the currently authenticated Supabase user,
    or None when nobody is logged in.
    """

    return st.session_state.get("supabase_user")


def sign_in(email, password):
    """
    Sign in using Supabase email/password authentication.
    """

    client = get_supabase_client()

    response = client.auth.sign_in_with_password(
        {
            "email": email.strip(),
            "password": password,
        }
    )

    user = response.user

    if user is not None:
        st.session_state["supabase_user"] = user

    return user


def sign_up(email, password):
    """
    Create a new Supabase Auth user.
    """

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

    return user


def reset_password(email):
    """
    Send password recovery email.
    """

    client = get_supabase_client()

    redirect_url = os.getenv(
        "SUPABASE_REDIRECT_URL",
        "http://localhost:8501",
    )

    return client.auth.reset_password_for_email(
        email.strip(),
        options={
            "redirect_to": redirect_url,
        },
    )


def update_password(new_password):
    """
    Update the password for the authenticated recovery session.
    """

    client = get_supabase_client()

    response = client.auth.update_user(
        {
            "password": new_password,
        }
    )

    return response.user


def sign_out():
    """
    Sign out the current Supabase user.
    """

    try:
        client = get_supabase_client()
        client.auth.sign_out()
    finally:
        st.session_state.pop(
            "supabase_user",
            None,
        )


def handle_password_recovery():
    """
    Detect a Supabase password-recovery callback.

    For PKCE flows, Supabase can return a `code` query parameter.
    """

    query_params = st.query_params

    code = query_params.get("code")

    if not code:
        return False

    if st.session_state.get("recovery_code") == code:
        return st.session_state.get(
            "password_recovery",
            False,
        )

    try:
        client = get_supabase_client()

        response = client.auth.exchange_code_for_session(
            {
                "auth_code": code,
            }
        )

        user = response.user

        if user is not None:
            st.session_state["supabase_user"] = user

        st.session_state["password_recovery"] = True
        st.session_state["recovery_code"] = code

        st.query_params.clear()

        return True

    except Exception as exc:
        st.error(
            f"Password recovery link could not be processed: {exc}"
        )
        return False


def render_password_recovery():
    """
    Render the set-new-password page.
    """

    st.title("Rocket Guardian AI")
    st.subheader("Set New Password")

    st.write(
        "Enter a new password for your account."
    )

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
            st.error(
                "Password must be at least 8 characters."
            )
            return

        if new_password != confirm_password:
            st.error(
                "Passwords do not match."
            )
            return

        try:

            update_password(new_password)

            st.success(
                "Password updated successfully. "
                "You can now continue to the application."
            )

            st.session_state.pop(
                "password_recovery",
                None,
            )

            st.session_state.pop(
                "recovery_code",
                None,
            )

            st.rerun()

        except Exception as exc:

            st.error(
                f"Password update failed: {exc}"
            )


def render_login():
    """
    Render the login/sign-up screen.

    Returns True when a user is authenticated.
    """

    if handle_password_recovery():

        render_password_recovery()

        return False

    st.title("Rocket Guardian AI")
    st.subheader("Customer Login")

    tab_login, tab_signup = st.tabs(
        [
            "Login",
            "Create Account",
        ]
    )

    with tab_login:

        st.text_input(
            "Email",
            key="login_email",
        )

        st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True,
        ):

            login_email = (
                st.session_state.get(
                    "login_email",
                    "",
                )
                .strip()
            )

            login_password = st.session_state.get(
                "login_password",
                "",
            )

            if not login_email:

                st.error(
                    "Please enter your email."
                )

            elif not login_password:

                st.error(
                    "Please enter your password."
                )

            else:

                try:

                    user = sign_in(
                        login_email,
                        login_password,
                    )

                    if user is not None:

                        st.success(
                            "Login successful."
                        )

                        st.rerun()

                except Exception as exc:

                    st.error(
                        f"Login failed: {exc}"
                    )

        st.divider()

        if st.button(
            "Forgot Password?",
            use_container_width=True,
        ):

            reset_email = (
                st.session_state.get(
                    "login_email",
                    "",
                )
                .strip()
            )

            if not reset_email:

                st.error(
                    "Please enter your email first."
                )

            else:

                try:

                    reset_password(
                        reset_email
                    )

                    st.success(
                        "Password reset email sent. "
                        "Check your email and follow the reset link."
                    )

                except Exception as exc:

                    st.error(
                        f"Password reset failed: {exc}"
                    )

    with tab_signup:

        st.text_input(
            "Email",
            key="signup_email",
        )

        st.text_input(
            "Password",
            type="password",
            key="signup_password",
        )

        if st.button(
            "Create Account",
            use_container_width=True,
        ):

            signup_email = (
                st.session_state.get(
                    "signup_email",
                    "",
                )
                .strip()
            )

            signup_password = st.session_state.get(
                "signup_password",
                "",
            )

            if not signup_email:

                st.error(
                    "Please enter your email."
                )

            elif len(signup_password) < 8:

                st.error(
                    "Password must be at least 8 characters."
                )

            else:

                try:

                    user = sign_up(
                        signup_email,
                        signup_password,
                    )

                    if user is not None:

                        st.success(
                            "Account created. "
                            "Check your email if confirmation is required."
                        )

                        if (
                            st.session_state.get(
                                "supabase_user"
                            )
                            is not None
                        ):

                            st.rerun()

                except Exception as exc:

                    st.error(
                        f"Account creation failed: {exc}"
                    )

    return False