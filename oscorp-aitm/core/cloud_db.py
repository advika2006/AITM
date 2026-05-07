# core/cloud_db.py
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

@st.cache_resource
def get_db():
    """Initialise and return the Firestore client using Streamlit secrets."""
    try:
        if not firebase_admin._apps:
            # Assumes you have your Firebase JSON credentials stored in st.secrets["firebase"]
            if "firebase" in st.secrets:
                dict_creds = dict(st.secrets["firebase"])
                cred = credentials.Certificate(dict_creds)
                firebase_admin.initialize_app(cred)
            else:
                return None
        return firestore.client()
    except Exception as e:
        print(f"Firebase not configured: {e}")
        return None