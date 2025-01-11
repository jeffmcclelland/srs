import random
import streamlit as st

def get_random_boost_gif(boost_type, df_boosters, DEBUG=False):
    if DEBUG:
        st.write(f"Debug - Looking for boost_type:", boost_type)
    
    filtered_boosts = df_boosters[df_boosters['Boost type'] == boost_type]
    
    if DEBUG:
        st.write(f"Debug - Filtered boosts:", filtered_boosts)
        st.write(f"Debug - Number of matching boosts:", len(filtered_boosts))
    
    if not filtered_boosts.empty:
        url = random.choice(filtered_boosts['Boost URL'].tolist())
        if DEBUG:
            st.write(f"Debug - Selected URL:", url)
        return url
    
    if DEBUG:
        st.write("Debug - No matching boosts found")
    return None
