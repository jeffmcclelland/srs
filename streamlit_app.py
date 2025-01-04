import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Initialize canvas key in session state if it doesn't exist
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# Create a canvas component with fixed parameters
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color="#37384c",
    background_color="#F9FBFD",
    background_image=None,
    update_streamlit=True,
    height=200,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)

# Reset button
if st.button('Clear Writing Area'):
    st.session_state.canvas_key += 1
    st.rerun()
