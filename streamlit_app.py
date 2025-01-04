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
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import gspread
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request

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
DEBUG = False
PROMPT = "What word or letters are in this drawing?"
DATA_RANGE = "A:D"  # Range of columns to read/write in Google Sheet

# Google sheets config
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

skey = st.secrets["gcp_service_account"]
googlecreds = Credentials.from_service_account_info(skey, scopes=SCOPES)

client = gspread.authorize(googlecreds)

# Create a new request object, capable of making HTTP requests
request = Request()
# Use it to refresh the access token
googlecreds.refresh(request)

# Spreadsheet configuration
spreadsheet_url = "https://docs.google.com/spreadsheets/d/1NyaBvbHef_eX1lBYtTPzZJ2fBSPRG2yYxitTeEoJy-M/edit?gid=324250006#gid=324250006"
sheet_name_SRSNext = "SRSNext"

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
            if DEBUG:
                st.write("Debug - Selected model:", selected_model)
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            if DEBUG:
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
            
            if DEBUG:
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

# Function to read Google Sheet and display as dataframe
def read_sheet_to_df(googlecreds, spreadsheet_url, spreadsheet_sheet_name, data_range):
    # Set up the credentials
    client = gspread.authorize(googlecreds)

    # Open the spreadsheet
    sh = client.open_by_url(spreadsheet_url)

    # Select worksheet
    worksheet = sh.worksheet(spreadsheet_sheet_name)

    # Fetch data from the defined range
    data = worksheet.get(data_range)

    # Create a DataFrame from the fetched data
    df = pd.DataFrame(data[1:], columns=data[0])

    return df

# Function to write dataframe to Google Sheet
def write_df_to_google_sheet(googlecreds, 
                            spreadsheet_url, 
                            spreadsheet_sheet_name, 
                            df, 
                            start_cell='A1', 
                            clear_sheet=False, 
                            flag_append=False):
    # Setup the credentials
    client = gspread.authorize(googlecreds)
    
    # Open the spreadsheet
    sh = client.open_by_url(spreadsheet_url)
    
    # Select worksheet
    worksheet = sh.worksheet(spreadsheet_sheet_name)
    
    # Fill NaN values in DataFrame with empty strings and downcast types
    df = df.fillna('').infer_objects(copy=False)

    # Convert Timestamp columns to string
    for column in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[column] = df[column].astype(str)
    
    # Prepare data from DataFrame to write
    if flag_append:
        data = df.values.tolist()  # Exclude header for appending
    else:
        data = [df.columns.values.tolist()] + df.values.tolist()  # Include header
    
    # Optionally clear the existing data
    if clear_sheet:
        worksheet.clear()
        
    if not flag_append:  # If not appending, write data starting from start_cell
        worksheet.update(start_cell, data)
    else:
        # To append data, find the first empty row
        all_values = worksheet.get_all_values()
        first_empty_row = len(all_values) + 1  # Next empty row after the last filled row
        start_cell = f"A{first_empty_row}"

        # Update starting at the next empty cell calculated
        worksheet.update(start_cell, data)
    
    if DEBUG:
        print(f"Finished writing to sheet '{spreadsheet_sheet_name}' - Append Mode: {'Yes' if flag_append else 'No'}")

# Add a separator before the dataframe
st.markdown("---")
st.subheader("SRS Data")

# Read and display the sheet data
try:
    df = read_sheet_to_df(googlecreds, spreadsheet_url, sheet_name_SRSNext, DATA_RANGE)
    st.dataframe(df)
except Exception as e:
    if DEBUG:
        st.error(f"Error reading spreadsheet: {str(e)}")
    else:
        st.error("Error reading spreadsheet data")
