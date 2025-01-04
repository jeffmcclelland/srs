import os
import requests
import streamlit as st
import sys
import yaml
import aisuite as ai
import base64
import io
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Configure Streamlit to use wide mode and hide the top streamlit menu
st.set_page_config(layout="wide", menu_items={})

# Add heading with padding
st.markdown(
    "<div style='padding-top: 1rem;'><h2 style='text-align: center; color: #ffffff;'>Siiri SRS</h2></div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        /* Hide Streamlit's default top bar */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Remove top padding/margin */
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            margin-top: 0rem;
        }

        /* Remove padding from the app container */
        .appview-container {
            padding-top: 0rem;
        }
        
        /* Custom CSS for scrollable chat container */
        .chat-container {
            height: 650px;
            overflow-y: auto !important;
            background-color: #1E1E1E;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        
        /* Ensure the container takes full width */
        .stMarkdown {
            width: 100%;
        }
        
        /* Style for chat messages to ensure they're visible */
        .chat-message {
            margin: 10px 0;
            padding: 10px;
        }
        
        #text_area_1 {
            min-height: 20px !important;
        } 
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants
PROMPT = "What word or letters are in this drawing?"

# Load configuration and initialize aisuite client
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
configured_llms = config["llms"]

# Create model selection dropdown
model_names = [f"{llm['provider']}:{llm['model']}" for llm in configured_llms]
selected_model = st.selectbox("Select Model", model_names)

# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

client = ai.Client()

# Initialize canvas key in session state if it doesn't exist
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

if 'llm_response' not in st.session_state:
    st.session_state.llm_response = None

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

# Add Check button
check_button = st.button("Check")

# Only query LLM if there's actually a drawing and the Check button is pressed
if check_button and canvas_result.image_data is not None and canvas_result.json_data["objects"]:
    # Create status box for LLM response
    with st.status("Analyzing your drawing...", expanded=True) as status:
        try:
            # Debug: Print selected model
            st.write("Debug - Selected model:", selected_model)
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Debug: Print first few chars of base64 string
            st.write("Debug - Base64 string starts with:", img_str[:50])
            
            # Prepare message based on provider
            provider = selected_model.split(":")[0]
            
            if provider == "openai":
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", 
                         "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                         }
                        }
                    ]}
                ]
            else:  # anthropic
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_str
                            }
                        }
                    ]}
                ]
            
            # Debug: Print the message structure
            st.write("Debug - Message structure:", messages)
            st.write("Debug - Provider:", provider)
            
            # Query the LLM
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=0.0
            )
            
            # Store the response
            st.session_state.llm_response = response.choices[0].message.content
            status.update(label="Analysis complete!", state="complete")
        except Exception as e:
            st.error(f"Error analyzing drawing: {str(e)}")
            status.update(label="Analysis failed", state="error")

# Display the LLM response if available
if st.session_state.llm_response:
    st.write("Response:", st.session_state.llm_response)

    # Add a "Clear" button to reset the canvas and response
    if st.button("Clear"):
        st.session_state.canvas_key += 1
        st.session_state.llm_response = None
        st.rerun()
