import time
import streamlit as st


def get_cached_df(cache_name, cache_time_name, ttl_seconds, fetch_func):
    """Get DataFrame from cache or fetch if expired"""
    current_time = time.time()
    if (st.session_state[cache_name] is None or 
        st.session_state[cache_time_name] is None or 
        current_time - st.session_state[cache_time_name] > ttl_seconds):
        # Cache expired or doesn't exist, fetch new data
        st.session_state[cache_name] = fetch_func()
        st.session_state[cache_time_name] = current_time
    return st.session_state[cache_name]
