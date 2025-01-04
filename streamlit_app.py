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

# Configure Streamlit to use wide mode and hide the top streamlit menu
st.set_page_config(layout="wide", menu_items={})

# Constants
DEBUG = True
DATA_RANGE = "A:D"  # Range of columns to read/write in Google Sheet
SELECTED_MODEL = "anthropic:claude-3-opus-20240229"  # Hardcoded to Claude model

# Template for LLM prompt
LLM_PROMPT_TEMPLATE = """Review the image. It contains text that was handwritten. 

The user was issued the following prompt: {prompt}
The correct answer is: {correct_answer}

This is a test of their spelling abilities. Capitalisation doesn't matter but the spelling should be exactly correct. 
- Congratulate the user if the writing in the image matches. 
- If it does not match, then indicate the errors that the user made. For instance which letters were missing or incorrect. 
- If it's not possible to read any letters from the image, indicate that the user should try writing it again. 

Respond in valid json with the following keys:
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

# Load configuration and initialize aisuite client
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

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

def log_response_to_sheet(question_data, llm_response, timestamp):
    """Log the response to the SRSLog sheet"""
    try:
        # Parse LLM response as JSON
        result = json.loads(llm_response)
        
        # Calculate next ask timestamp
        if result['correct']:
            next_ask = timestamp + timedelta(minutes=20)
        else:
            next_ask = timestamp + timedelta(minutes=10)
            
        next_ask_str = next_ask.strftime('%Y-%m-%d %H:%M:%S')
            
        # Prepare the log entry
        log_data = pd.DataFrame({
            'Prompt ID': [question_data['Prompt ID']],
            'Prompt': [question_data['Prompt']],
            'Correct Answer': [question_data['Correct Answer']],
            'Asked Timestamp': [timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            'Result': ['Correct' if result['correct'] else 'Incorrect'],
            'Result Details': ['Correct' if result['correct'] else result['user_message']],
            'Next Ask Timestamp': [next_ask_str]
        })
        
        # Write to SRSLog sheet
        write_df_to_google_sheet(
            googlecreds,
            spreadsheet_url,
            'SRSLog',
            log_data,
            flag_append=True
        )
        
        # Update the Next Ask Timestamp in SRSNext
        # Add 2 to account for header row and 0-based index
        row_index = st.session_state.current_question_index + 2
        update_next_ask_timestamp(
            googlecreds,
            spreadsheet_url,
            sheet_name_SRSNext,
            row_index,
            next_ask_str
        )
        
        if DEBUG:
            st.write("Debug - Logged response to SRSLog and updated SRSNext")
            
    except Exception as e:
        if DEBUG:
            st.error(f"Error logging response: {str(e)}")

def update_next_ask_timestamp(googlecreds, spreadsheet_url, sheet_name, row_index, next_ask_timestamp):
    """Update the Next Ask Timestamp for a specific row in SRSNext sheet.
    
    Args:
        googlecreds: Google credentials
        spreadsheet_url: URL of the spreadsheet
        sheet_name: Name of the sheet (SRSNext)
        row_index: The row number to update (1-based)
        next_ask_timestamp: The new timestamp value
    """
    try:
        # Setup the credentials and get worksheet
        client = gspread.authorize(googlecreds)
        sh = client.open_by_url(spreadsheet_url)
        worksheet = sh.worksheet(sheet_name)
        
        # Column D is the Next Ask Timestamp column
        # Convert row_index to A1 notation for column D
        cell = f'D{row_index}'
        
        # Update the cell
        worksheet.update_acell(cell, next_ask_timestamp)
        
        if DEBUG:
            print(f"Updated Next Ask Timestamp in {sheet_name} at {cell}")
            
    except Exception as e:
        if DEBUG:
            st.error(f"Error updating Next Ask Timestamp: {str(e)}")

def move_to_next_question():
    """Move to the next question and reset the canvas"""
    st.session_state.current_question_index += 1
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

# Read and display the sheet data
try:
    # Only read the sheet if we haven't loaded questions yet
    if st.session_state.active_questions is None:
        # Read the sheet
        df = read_sheet_to_df(googlecreds, spreadsheet_url, sheet_name_SRSNext, DATA_RANGE)
        
        # Convert Next Ask Timestamp to datetime
        df['Next Ask Timestamp'] = pd.to_datetime(df['Next Ask Timestamp'])
        
        # Filter for questions that should be asked now
        current_time = datetime.now()
        st.session_state.active_questions = df[df['Next Ask Timestamp'] <= current_time].copy()
    
    active_questions = st.session_state.active_questions
    
    # Display number of prompts at the top
    num_prompts = len(active_questions)
    if num_prompts > 0:
        st.info(f"You have {num_prompts} word{'s' if num_prompts > 1 else ''} to practice today!")
    
    if len(active_questions) > 0 and st.session_state.current_question_index < len(active_questions):
        if DEBUG:
            st.write("Debug - Active questions:", active_questions)
            st.write("Debug - Current question index:", st.session_state.current_question_index)
        
        # Get the current question
        current_question = active_questions.iloc[st.session_state.current_question_index]
        prompt = current_question['Prompt']
        correct_answer = current_question['Correct Answer']
        
        # Show progress
        st.progress((st.session_state.current_question_index) / num_prompts, 
                   text=f"Progress: {st.session_state.current_question_index + 1}/{num_prompts}")
        
        # Display the prompt
        st.header("Practice Question")
        st.write(prompt)
        
        # Drawing canvas for user input
        canvas_result = st_canvas(
            stroke_width=4,
            stroke_color="#37384c",
            background_color="#F9FBFD",
            height=300,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        # Create a "Check" button to trigger LLM analysis
        if st.button("Check"):
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
                        prompt=prompt,
                        correct_answer=correct_answer
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
                        current_time = datetime.now()
                        log_response_to_sheet(current_question, llm_response, current_time)
                        
                        # Display the response
                        st.write("Response:", result['user_message'])
                        
                        # If this was the last question, show completion message
                        if st.session_state.current_question_index == len(active_questions) - 1:
                            st.success("Congratulations! You've completed all your practice questions!")
                        else:
                            # Show next button if not the last question
                            if st.button("Next Question"):
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
