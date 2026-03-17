"""
Authentication helpers for Enterprise AI Assistant.
Uses simple username/password from environment variables.
"""
import hashlib
import streamlit as st
from src.config import get_users


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# Pre-hash the stored passwords on first load
_USERS_HASHED = {u: _hash(p) for u, p in get_users().items()}


def check_credentials(username: str, password: str) -> bool:
    return _USERS_HASHED.get(username) == _hash(password)


def is_logged_in() -> bool:
    return st.session_state.get("authenticated", False)


def get_current_user() -> str:
    return st.session_state.get("current_user", "guest")


def login(username: str, password: str) -> bool:
    if check_credentials(username, password):
        st.session_state.authenticated = True
        st.session_state.current_user  = username
        return True
    return False


def logout():
    st.session_state.authenticated  = False
    st.session_state.current_user   = ""
    st.session_state.messages       = []
    st.session_state.active_conv_id = None


def render_login_page():
    """Render a styled login page. Returns True if user just logged in."""
    st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    .login-box {
        max-width: 420px; margin: 80px auto; padding: 48px 40px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    }
    .login-title { font-size: 28px; font-weight: 800;
                   text-align: center; margin-bottom: 4px; }
    .login-sub   { text-align: center; color: #888; margin-bottom: 32px; }
</style>
""", unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown('<div class="login-title">💼 Enterprise AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Sign in to access your documents</div>', unsafe_allow_html=True)
        st.markdown("")

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        st.markdown("")
        if st.button("Sign In →", use_container_width=True, type="primary"):
            if login(username, password):
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        st.markdown(
            '<div style="text-align:center;color:#555;font-size:12px;margin-top:24px">'
            'Secured · Citations enforced · Audit logged</div>',
            unsafe_allow_html=True,
        )
