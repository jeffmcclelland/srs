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
from datetime import datetime, timedelta
import json
import pytz

# Configure Streamlit page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Siiri SRS",
    page_icon="🖍️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={},
)

# Constants
DEBUG = False
DATA_RANGE = "A:E"  # Updated to include Confidence Level column
SELECTED_MODEL = None  # Use the model from config
TIMEZONE = pytz.timezone('EET')  # Add timezone constant
ACTIVE_THEME = "theme1"  # Can be "theme1" or "theme2"

# Theme definitions
THEMES = {
    "theme1": {
        "primaryColor": "#eb5e28",
        "backgroundColor": "#fffcf2",
        "secondaryBackgroundColor": "#fff",
        "textColor": "#403d39"
    },
    "theme2": {
        "primaryColor": "#ff6700",
        "backgroundColor": "#fff",
        "secondaryBackgroundColor": "#ebebeb",
        "textColor": "#004e98"
    }
}

# SRS time delay configuration
srs_time_delays = [
    {"confidence_level": 0, "delay_quantity": 10, "delay_time_unit": 'minutes'},
    {"confidence_level": 1, "delay_quantity": 8,  "delay_time_unit": 'hours'},
    {"confidence_level": 2, "delay_quantity": 3,  "delay_time_unit": 'days'},
    {"confidence_level": 3, "delay_quantity": 5,  "delay_time_unit": 'days'},
    {"confidence_level": 4, "delay_quantity": 10, "delay_time_unit": 'days'},
    {"confidence_level": 5, "delay_quantity": 20, "delay_time_unit": 'days'},
]

# Get current theme
current_theme = THEMES[ACTIVE_THEME]

# Apply theme
st.markdown(
    f"""
    <style>
        /* Theme colors */
        :root {{
            --primary-color: {current_theme["primaryColor"]};
            --background-color: {current_theme["backgroundColor"]};
            --secondary-background-color: {current_theme["secondaryBackgroundColor"]};
            --text-color: {current_theme["textColor"]};
        }}
        
        /* Hide Streamlit's default top bar */
        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Apply theme colors */
        .stApp {{
            background-color: var(--background-color);
            color: var(--text-color);
        }}
        
        /* Primary button style */
        .stButton > button[kind="primary"] {{
            background-color: var(--primary-color);
            color: var(--background-color);
            border-radius: 20px;
            padding: 0.2rem 1rem;
        }}
        
        /* Secondary button style */
        .stButton > button[kind="secondary"] {{
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid var(--text-color);
            border-radius: 20px;
            padding: 0.2rem 1rem;
        }}
        
        /* Remove top padding/margin */
        .block-container {{
            padding-top: 0rem;
            padding-bottom: 0rem;
            margin-top: 0rem;
        }}

        /* Remove padding from the app container */
        .appview-container {{
            padding-top: 0rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get Haiku model from config
HAIKU_MODEL = next(
    (f"{model['provider']}:{model['model']}" 
     for model in config['llms'] 
     if model['name'] == "Anthropic Claude 3.5 Sonnet"),
    None
)

if DEBUG:
    st.write("Debug - Selected model:", HAIKU_MODEL)

SELECTED_MODEL = HAIKU_MODEL  # Use the model from config

# Template for LLM prompt
LLM_PROMPT_TEMPLATE = """Review the image. It contains text that was handwritten. 

The user was issued the following prompt: '{prompt}'

The correct answer is: '{correct_answer}'

This is a test of their spelling abilities. Capitalisation doesn't matter but the spelling should be exactly correct. 
- Congratulate the user if the writing in the image matches. 
- If it does not match, then indicate the errors that the user made. For instance which letters were missing or incorrect. 
- If it's not possible to read any letters from the image, indicate that the user should try writing it again. 

Respond in valid json with the following keys:
- "image_answer": string - the words detected in the image
- "correct": true or false
- "try_again": true or false - true if it's not possible to read any letters from the image; false in all other cases
- "user_message": string - a nice congratulations if they got it right, or a concise, clear message about what they got wrong. if it's not possible to read any letters from the image, then indicate the user should try again.

"""

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

# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

client = ai.Client()

# Initialize session state variables
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'active_questions' not in st.session_state:
    st.session_state.active_questions = None
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = None
if 'full_df' not in st.session_state:
    st.session_state.full_df = None
if 'current_timestamp' not in st.session_state:
    st.session_state.current_timestamp = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'show_next_button' not in st.session_state:
    st.session_state.show_next_button = False

def calculate_next_timestamp(confidence_level, current_time):
    """Calculate next ask timestamp based on confidence level"""
    delay_config = next(d for d in srs_time_delays if d['confidence_level'] == confidence_level)
    
    if delay_config['delay_time_unit'] == 'minutes':
        delta = timedelta(minutes=delay_config['delay_quantity'])
    elif delay_config['delay_time_unit'] == 'hours':
        delta = timedelta(hours=delay_config['delay_quantity'])
    else:  # days
        delta = timedelta(days=delay_config['delay_quantity'])
    
    return current_time + delta

def update_confidence_level(current_level, is_correct):
    """Update confidence level based on answer correctness"""
    if not is_correct:
        return 0
    return min(current_level + 1, 5)

def update_next_ask_timestamp(googlecreds, spreadsheet_url, sheet_name, row_index, next_ask_timestamp, confidence_level):
    """Update both Next Ask Timestamp and Confidence Level for a specific row"""
    try:
        if DEBUG:
            print(f"Debug - Updating row {row_index} with timestamp {next_ask_timestamp} and confidence {confidence_level}")
            
        client = gspread.authorize(googlecreds)
        sh = client.open_by_url(spreadsheet_url)
        worksheet = sh.worksheet(sheet_name)
        
        # Update Next Ask Timestamp (Column E) and Confidence Level (Column D)
        worksheet.batch_update([
            {'range': f'E{row_index}', 'values': [[next_ask_timestamp]]},
            {'range': f'D{row_index}', 'values': [[confidence_level]]}
        ])
        
        if DEBUG:
            print("Debug - Update successful")
            
    except Exception as e:
        if DEBUG:
            print(f"Debug - Error updating row: {str(e)}")
            st.error(f"Error updating row: {str(e)}")

def log_response_to_sheet(question_data, llm_response, timestamp):
    """Log the response to the SRSLog sheet"""
    try:
        # Parse LLM response as JSON
        result = json.loads(llm_response)
        
        # Get current confidence level
        current_confidence = int(question_data['Confidence Level'])
        
        # Calculate new confidence level
        new_confidence = update_confidence_level(current_confidence, result['correct'])
        
        # Calculate next ask timestamp based on new confidence
        next_ask = calculate_next_timestamp(new_confidence, timestamp)
        next_ask_str = next_ask.strftime('%Y-%m-%d %H:%M:%S')
            
        # Prepare the log entry
        log_data = pd.DataFrame({
            'Prompt ID': [question_data['Prompt ID']],
            'Prompt': [question_data['Prompt']],
            'Correct Answer': [question_data['Correct Answer']],
            'Asked Timestamp': [timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            'Image Answer': [result.get('image_answer', '')],  # Get image_answer from LLM response
            'Result': ['Correct' if result['correct'] else 'Incorrect'],
            'Result Details': [result.get('user_message', 'No details provided')],
            'Confidence Level Before': [current_confidence],
            'Confidence Level After': [new_confidence],
            'Next Ask Timestamp': [next_ask_str]
        })
        
        if DEBUG:
            print("Debug - Log data:")
            print(log_data)
            print("Debug - Image Answer from LLM:", result.get('image_answer', ''))
            print("Debug - Result Details:", result.get('user_message', 'No details provided'))
        
        # Write to SRSLog sheet
        write_df_to_google_sheet(googlecreds, 
                               spreadsheet_url, 
                               'SRSLog',
                               log_data,
                               flag_append=True)
        
        # Update the Next Ask Timestamp and Confidence Level in SRSNext
        row_index = st.session_state.current_question_index + 2
        update_next_ask_timestamp(
            googlecreds,
            spreadsheet_url,
            sheet_name_SRSNext,
            row_index,
            next_ask_str,
            new_confidence
        )
        
        if DEBUG:
            print(f"Debug - Updated confidence level from {current_confidence} to {new_confidence}")
            
    except Exception as e:
        if DEBUG:
            print(f"Debug - Error logging response: {str(e)}")
            st.error(f"Error logging response: {str(e)}")

def move_to_next_question():
    """Move to the next question and reset the canvas"""
    if DEBUG:
        print("Debug - Moving to next question")
        print(f"Debug - Current index: {st.session_state.current_question_index}")
        
    # Reset all relevant session state
    st.session_state.current_question_index += 1
    st.session_state.canvas_key += 1
    st.session_state.llm_response = None
    st.session_state.current_image = None
    st.session_state.show_next_button = False
    
    if DEBUG:
        print(f"Debug - New index: {st.session_state.current_question_index}")
    
    # Force a complete rerun of the app
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
        worksheet.update(values=data, range_name=start_cell)
    else:
        # To append data, find the first empty row
        all_values = worksheet.get_all_values()
        first_empty_row = len(all_values) + 1  # Next empty row after the last filled row
        start_cell = f"A{first_empty_row}"

        # Update starting at the next empty cell calculated
        worksheet.update(values=data, range_name=start_cell)
    
    if DEBUG:
        print(f"Finished writing to sheet '{spreadsheet_sheet_name}' - Append Mode: {'Yes' if flag_append else 'No'}")

# Add heading with padding
st.markdown(
    "<div style='padding-top: 1rem;'><h2 style='text-align: center;'>Siiri SRS</h2></div>",
    unsafe_allow_html=True,
)

# Read and display the sheet data
try:
    # Only read the sheet if we haven't loaded questions yet
    if st.session_state.active_questions is None:
        # Read the sheet data
        df = read_sheet_to_df(googlecreds, spreadsheet_url, sheet_name_SRSNext, DATA_RANGE)
        
        # Get current time in EET
        timestamp = datetime.now(TIMEZONE)
        
        # Convert timestamps in dataframe to EET timezone
        df['Next Ask Timestamp'] = pd.to_datetime(df['Next Ask Timestamp']).dt.tz_localize('EET')
        
        # Filter for questions that are due
        mask = df['Next Ask Timestamp'] <= timestamp
        df_filtered = df[mask].copy()
        
        st.session_state.full_df = df  # Store the full df in session state
        st.session_state.active_questions = df_filtered
        st.session_state.current_timestamp = timestamp
    
    active_questions = st.session_state.active_questions
    
    # Debug information - outside the initialization block so it stays visible
    if DEBUG:
        st.write("### Debug Information")
        st.write("Full SRSNext dataframe before filtering:")
        st.write(st.session_state.full_df)
        st.write("DataFrame info:")
        st.write(st.session_state.full_df.info())
        st.write("Current time:", st.session_state.current_timestamp)
        st.write(f"Number of questions after filtering: {len(active_questions)}")
        if len(active_questions) > 0:
            st.write("First filtered question:")
            st.write(active_questions.iloc[0])
    
    # Display number of prompts at the top
    num_prompts = len(active_questions)
    if num_prompts > 0:
        st.markdown(f"You have {num_prompts} word{'s' if num_prompts > 1 else ''} to practice today!")
    
    if len(active_questions) > 0 and st.session_state.current_question_index < len(active_questions):
        if DEBUG:
            st.write("Debug - Active questions:", active_questions)
            st.write("Debug - Current question index:", st.session_state.current_question_index)
        
        # Get the current question
        current_question = active_questions.iloc[st.session_state.current_question_index]
        prompt = current_question['Prompt']
        correct_answer = current_question['Correct Answer']
        
        # Display progress
        if active_questions is not None and len(active_questions) > 0:
            progress = st.progress(st.session_state.current_question_index / len(active_questions))
            st.write(f"Question {st.session_state.current_question_index + 1} of {len(active_questions)}")

        # Display the prompt
        st.write(f"### {current_question['Prompt']}")
        
        # Get theme colors from our theme dictionary
        theme_secondary_bg = current_theme["secondaryBackgroundColor"]
        theme_primary_color = current_theme["primaryColor"]
        
        # Create canvas
        canvas_result = st_canvas(
            stroke_width=6,
            stroke_color=theme_primary_color,  
            background_color=theme_secondary_bg,  
            height=300,
            width=800,
            display_toolbar=False,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )
        
        # Create two columns for buttons
        col1, col2 = st.columns([1, 8])
        
        # Clear button in left column
        if col1.button("Clear",type="secondary"):
            st.session_state.canvas_key += 1
            st.rerun()
            
        # Check button in right column
        if col2.button("Check",type="primary"):
            # Create status box for LLM response
            with st.status("Analyzing your writing...", expanded=True) as status:
                try:
                    # Convert numpy array to PIL Image
                    image = Image.fromarray(canvas_result.image_data.astype('uint8'))
                    
                    # Convert to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Prepare message based on provider
                    provider = SELECTED_MODEL.split(":")[0]
                    prompt_text = LLM_PROMPT_TEMPLATE.format(
                        prompt=current_question['Prompt'],
                        correct_answer=current_question['Correct Answer']
                    )
                    
                    if DEBUG:
                        st.write("Debug - Prompt text:", prompt_text)
                    
                    if provider == "openai":
                        messages = [
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt_text},
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
                                {"type": "text", "text": prompt_text},
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
                        st.write("Debug - Model:", SELECTED_MODEL)
                    
                    # Query the LLM
                    response = client.chat.completions.create(
                        model=SELECTED_MODEL,
                        messages=messages,
                        temperature=0.5
                    )
                    
                    # Store the response
                    llm_response = response.choices[0].message.content
                    st.session_state.llm_response = llm_response
                    
                    if DEBUG:
                        st.write("Debug - Raw LLM response:", llm_response)
                    
                    # Parse the JSON response
                    result = json.loads(llm_response)
                    
                    if DEBUG:
                        st.write("Debug - Parsed JSON response:", result)
                    
                    if result['try_again']:
                        st.write("Response:", result['user_message'])
                        # Clear canvas for retry
                        st.session_state.canvas_key += 1
                        st.rerun()
                    else:
                        # Log the response
                        current_time = datetime.now(TIMEZONE)
                        log_response_to_sheet(current_question, llm_response, current_time)
                        
                        # Display the response
                        st.write("Response:", result['user_message'])
                        
                        # If this was the last question, show completion message
                        if st.session_state.current_question_index == len(active_questions) - 1:
                            st.success("Congratulations! You've completed all your practice questions!")
                        else:
                            # Show next button if not the last question
                            if st.button("Next Question",type="primary"):
                                move_to_next_question()
                                st.rerun()
                    
                    status.update(label="Analysis complete!", state="complete", expanded=True)
                
                except Exception as e:
                    if DEBUG:
                        st.error(f"Error: {str(e)}")
                    st.error("An error occurred while processing your response")
                    status.update(label="Error occurred!", state="error", expanded=True)
    else:
        if len(active_questions) == 0:
            st.info("No questions are due at this time.")
        elif st.session_state.current_question_index >= len(active_questions):
            st.success("Congratulations! You've completed all your practice questions!")
    
    if DEBUG:
        st.subheader("All Questions")
        st.dataframe(active_questions)
        
    if len(active_questions) > 0 and st.session_state.current_question_index == len(active_questions):
        st.success("Congratulations! You've completed all your practice questions!")

except Exception as e:
    if DEBUG:
        st.error(f"Error reading spreadsheet: {str(e)}")
    else:
        st.error("Error reading spreadsheet data")
