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
import random
from urllib.parse import urlencode
import time
import tempfile
from openai import OpenAI
from src.config.settings import *
from src.services.boost import get_random_boost_gif
from src.services.tts import generate_tts_audio
from src.utils.time_utils import calculate_next_timestamp
from src.utils.cache import get_cached_df
from src.models.srs import toggle_study_mode, get_available_sets, update_confidence_level, increment_question, log_response_to_sheet
from src.utils.sheets import write_df_to_google_sheet, update_next_ask_timestamp

# Configure Streamlit page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Siiri SRS",
    page_icon="🖍️",
    initial_sidebar_state="collapsed",
    menu_items={},
)

# Apply theme
st.markdown(
    f"""
    <style>
        /* Theme colors */
        :root {{
            --primary-color: {CURRENT_THEME["primaryColor"]};
            --background-color: {CURRENT_THEME["backgroundColor"]};
            --secondary-background-color: {CURRENT_THEME["secondaryBackgroundColor"]};
            --text-color: {CURRENT_THEME["textColor"]};
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

The user was asked to translate following word from Estonian into English: '{prompt}'

The correct answer in English is: '{correct_answer}'

This is a test of their spelling abilities. 
- Carefully check letter-by-letter between what the user wrote and the correct answer.

Notes on what to consider correct: 
  - Capitalisation doesn't matter but the spelling should be exactly correct. 
  - Punctuation like commas and exclamation marks don't matter. 
  - If there are multiple correct answers provided (e.g. father / dad) and the USER WRITES **ANY OF THOSE ANSWERS** Then consider it CORRECT.

Notes on responding to the user: 
- Congratulate the user if they get the correct answer. 
- If they do not get the correct answer, indicate the errors that the user made. For instance which letters were missing or incorrect. 
- If it's not possible to read any letters from the image, indicate that the user should try writing it again. 

Respond in valid json with the following keys:
- "image_answer": string - the words detected in the image
- "user_message": string - a nice congratulations if they got it right, or a concise, clear message about what they got wrong. if it's not possible to read any letters from the image, then indicate the user should try again.
- "correct": true or false
- "try_again": true or false - true if it's not possible to read any letters from the image; false in all other cases

"""

# Google sheets config
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

googlecreds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)

client = gspread.authorize(googlecreds)

# Create a new request object, capable of making HTTP requests
request = Request()
# Use it to refresh the access token
googlecreds.refresh(request)

# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

ai_client = ai.Client()

# Initialize session state variables
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0
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
if 'selected_sets' not in st.session_state:
    st.session_state.selected_sets = {}
if 'study_mode' not in st.session_state:
    st.session_state.study_mode = False
if 'retry_count' not in st.session_state:
    st.session_state.retry_count = 0
if 'boost_gif' not in st.session_state:
    st.session_state.boost_gif = None

# Cache the DataFrames
if 'df_boosters_cache' not in st.session_state:
    st.session_state.df_boosters_cache = None
    st.session_state.df_boosters_cache_time = None

if 'df_srsnext_cache' not in st.session_state:
    st.session_state.df_srsnext_cache = None
    st.session_state.df_srsnext_cache_time = None

# Function to read Google Sheet and display as dataframe
def read_sheet_to_df(googlecreds, spreadsheet_url, sheet_name, data_range):
    """Read a Google Sheet into a pandas DataFrame"""
    try:
        # Set up the credentials
        client = gspread.authorize(googlecreds)
        
        # Open the spreadsheet
        sh = client.open_by_url(spreadsheet_url)
        
        # Select worksheet
        worksheet = sh.worksheet(sheet_name)

        # Get all data including headers
        data = worksheet.get(data_range)
        
        if DEBUG:
            st.write("Number of columns in data:", len(data[0]) if data and len(data) > 0 else "No data")

        if not data or len(data) < 2:  # Check if we have data and at least one row besides headers
            st.error("No data found in the sheet or data is incomplete")
            return []

        # Create a DataFrame from the fetched data
        df = pd.DataFrame(data[1:], columns=data[0])

        if DEBUG:
            st.write("DataFrame shape:", df.shape)
            st.write("DataFrame columns:", df.columns)
            st.write("First row of DataFrame:", df.iloc[0] if not df.empty else "DataFrame is empty")

        # If this is the SRSNext sheet, process the timestamps
        if sheet_name == SHEET_NAME_SRS_NEXT:
            # Clean and convert Next Ask Timestamp
            df['Next Ask Timestamp'] = df['Next Ask Timestamp'].apply(lambda x: x.strip("'") if isinstance(x, str) else x)
            df['Next Ask Timestamp'] = pd.to_datetime(df['Next Ask Timestamp'], format='%Y-%m-%d')
            # Localize timestamps to EET timezone
            df['Next Ask Timestamp'] = df['Next Ask Timestamp'].dt.tz_localize('EET')
            
            # Get current time in EET
            current_time = datetime.now(TIMEZONE)
            
            if DEBUG:
                st.write("Current time:", current_time)
                st.write("First timestamp after conversion:", df['Next Ask Timestamp'].iloc[0] if not df.empty else "No timestamps")
                st.write("Timezone of current time:", current_time.tzinfo)
                st.write("Timezone of first timestamp:", df['Next Ask Timestamp'].iloc[0].tzinfo if not df.empty else "No timestamps")

            # Filter questions due for review
            active_questions = df[df['Next Ask Timestamp'] <= current_time].to_dict('records')

            if DEBUG:
                st.write("Number of active questions:", len(active_questions))
                if active_questions:
                    st.write("First active question:", active_questions[0])

            return active_questions
        else:
            # For other sheets, return the DataFrame as is
            return df

    except Exception as e:
        st.error(f"Error reading spreadsheet: {str(e)}")
        if DEBUG:
            import traceback
            st.write("Full error:", traceback.format_exc())
        # Return empty list for SRSNext sheet, empty DataFrame for other sheets
        if sheet_name == SHEET_NAME_SRS_NEXT:
            return []  # Empty list for active questions
        else:
            return pd.DataFrame()  # Empty DataFrame for other sheets

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
        # To append data, find the last non-empty row
        all_values = worksheet.get_all_values()
        # Find the last non-empty row by iterating from bottom
        last_row = 0
        for i in range(len(all_values) - 1, -1, -1):
            if any(cell.strip() for cell in all_values[i]):  # Check if row has any non-empty cells
                last_row = i + 1
                break
        
        # If sheet is completely empty, start from row 1
        first_empty_row = max(1, last_row + 1)
        start_cell = f"A{first_empty_row}"

        # Update starting at the next empty cell calculated
        worksheet.update(values=data, range_name=start_cell)
    
    if DEBUG:
        print(f"Finished writing to sheet '{spreadsheet_sheet_name}' - Append Mode: {'Yes' if flag_append else 'No'}")

try:
    # Read the sheets with caching
    df = get_cached_df(
        'df_srsnext_cache',
        'df_srsnext_cache_time',
        540,  # 9 minutes TTL
        lambda: read_sheet_to_df(googlecreds, SPREADSHEET_URL, SHEET_NAME_SRS_NEXT, DATA_RANGE)
    )
    
    df_boosters = get_cached_df(
        'df_boosters_cache',
        'df_boosters_cache_time',
        3600,  # 1 hour TTL
        lambda: read_sheet_to_df(googlecreds, SPREADSHEET_URL, SHEET_NAME_BOOSTERS, BOOSTERS_RANGE)
    )
    
    # If we're not in study mode, show set selection
    if not st.session_state.study_mode:
        st.markdown("### Choose Sets to Study")
        
        # Get available sets and their counts
        available_sets = get_available_sets(googlecreds, SPREADSHEET_URL, SHEET_NAME_SRS_NEXT, DATA_RANGE, TIMEZONE)
        
        # Initialize selected_sets if empty
        if not st.session_state.selected_sets:
            st.session_state.selected_sets = {set_name: True for set_name in available_sets.keys()}
        
        if DEBUG:
            print("Available sets:", available_sets)
            print("Current selected_sets:", st.session_state.selected_sets)
        
        # Create list of pill labels with counts
        set_names = list(available_sets.keys())
        pill_labels = [f"{set_name} - {available_sets[set_name]} words" for set_name in set_names]
        
        if DEBUG:
            print("Pill labels:", pill_labels)
        
        # Get currently selected pill indices
        default_selected = [i for i, set_name in enumerate(set_names) 
                          if st.session_state.selected_sets.get(set_name, True)]
        
        if DEBUG:
            print("Default selected indices:", default_selected)
        
        try:
            # Display pills and get selected indices
            selected = st.multiselect(
                "Select Sets",
                options=set_names,
                default=set_names,  # Start with all selected
                format_func=lambda x: f"{x} ({available_sets[x]})"
            )
            
            if DEBUG:
                print("Selected sets:", selected)
            
            # Update selected_sets based on selection
            st.session_state.selected_sets = {
                set_name: (set_name in selected)
                for set_name in set_names
            }
            
            if DEBUG:
                print("Updated selected_sets:", st.session_state.selected_sets)
                
        except Exception as e:
            if DEBUG:
                print(f"Error in set selection: {str(e)}")
                st.error(f"Error in set selection: {str(e)}")
            
        # Get Started button
        if st.button("Get Started", type="primary"):
            st.session_state.study_mode = True
            st.rerun()
            
    else:
        # Show "Change Sets" button
        if st.button("Change Sets", type="secondary"):
            toggle_study_mode()
            st.rerun()

        # Only read the sheet if we haven't loaded questions yet
        if st.session_state.active_questions is None:
            if DEBUG:
                print("\n=== Debug: Loading active questions ===")
            # Read the sheet data
            all_questions = read_sheet_to_df(googlecreds, SPREADSHEET_URL, SHEET_NAME_SRS_NEXT, DATA_RANGE)
            if DEBUG:
                print(f"Total questions from sheet: {len(all_questions)}")
            
            # Filter questions based on selected sets
            active_questions = [q for q in all_questions if st.session_state.selected_sets.get(q['Set'], False)]
            if DEBUG:
                print(f"Filtered active questions: {len(active_questions)}")
            
            # Group questions by Next Ask Timestamp, sort groups, and randomize within groups
            from itertools import chain
            from collections import defaultdict
            
            # Group questions by Next Ask Timestamp
            grouped_questions = defaultdict(list)
            for question in active_questions:
                grouped_questions[question['Next Ask Timestamp']].append(question)
            
            # Sort each group randomly and combine all groups in timestamp order
            active_questions = list(chain.from_iterable(
                random.sample(group, len(group))  # Randomize within group
                for timestamp in sorted(grouped_questions.keys())  # Sort groups by timestamp
                for group in [grouped_questions[timestamp]]  # Get the group for this timestamp
            ))
            
            st.session_state.active_questions = active_questions
            if DEBUG:
                print("Active questions stored in session state")

        active_questions = st.session_state.active_questions

        # Display number of prompts at the top
        num_prompts = len(active_questions) if active_questions else 0
        if num_prompts > 0:
            st.markdown(f"You have {num_prompts} word{'s' if num_prompts > 1 else ''} to practice today!")
        
        if len(active_questions or []) > 0 and st.session_state.current_question_index < len(active_questions):
            if DEBUG:
                print(f"\n=== Debug: Displaying question ===")
                print(f"Current index: {st.session_state.current_question_index}")
                print(f"Total questions: {len(active_questions)}")
            
            current_question = active_questions[st.session_state.current_question_index]
            prompt = current_question['Prompt']
            correct_answer = current_question['Correct Answer']
            if DEBUG:
                print(f"Current prompt: {prompt}")
            
            # Display progress
            if active_questions is not None and len(active_questions) > 0:
                progress = st.progress(st.session_state.current_question_index / len(active_questions))
                st.write(f"Question {st.session_state.current_question_index + 1} of {len(active_questions)}")

            # Display the prompt
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### Translate the following:")
            with col2:
                if st.button("🔊 Listen", help="Listen to the correct answer"):
                    audio_file = generate_tts_audio(current_question["Correct Answer"])
                    if audio_file:
                        try:
                            if DEBUG:
                                print(f"\n=== Debug: Audio Playback ===")
                                print(f"Reading audio file: {audio_file}")
                                print(f"File exists: {os.path.exists(audio_file)}")
                                
                            with open(audio_file, 'rb') as f:
                                audio_bytes = f.read()
                                if DEBUG:
                                    print(f"Audio bytes read: {len(audio_bytes)}")
                            
                            try:
                                if DEBUG:
                                    print("Attempting to create audio player...")
                                audio_player = st.audio(audio_bytes, format='audio/mpeg')
                                if DEBUG:
                                    print("Audio player created successfully")
                                    print(f"Audio player type: {type(audio_player)}")
                                
                                # Store the audio bytes in session state to prevent garbage collection
                                st.session_state.current_audio = audio_bytes
                                
                            except Exception as audio_error:
                                if DEBUG:
                                    print(f"Error creating audio player: {str(audio_error)}")
                                    print(f"Error type: {type(audio_error)}")
                                raise audio_error
                                
                            # Clean up the temporary file after successful playback
                            os.unlink(audio_file)
                            if DEBUG:
                                print("Temporary file cleaned up")
                                
                        except Exception as e:
                            if DEBUG:
                                print(f"Error in audio playback: {str(e)}")
                            st.error(f"Error playing audio: {str(e)}")
            
            st.markdown(f'<h1 style="color: {CURRENT_THEME["primaryColor"]}">{current_question["Prompt"]}</h1>', unsafe_allow_html=True)
            
            # Get theme colors from our theme dictionary
            theme_secondary_bg = CURRENT_THEME["secondaryBackgroundColor"]
            theme_primary_color = CURRENT_THEME["primaryColor"]
            
            # Create the canvas
            canvas_result = st_canvas(
                stroke_width=6,
                stroke_color=theme_primary_color,  
                background_color=theme_secondary_bg,  
                height=CANVAS_HEIGHT,
                width=CANVAS_WIDTH,
                display_toolbar=False,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state.canvas_key}",
            )

            # Create two columns for buttons
            col1, col2 = st.columns([1, 7])
            
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
                        
                        # Initialize retry_count in session state if not exists
                        if 'retry_count' not in st.session_state:
                            st.session_state.retry_count = 0
                        
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
                        response = ai_client.chat.completions.create(
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
                        elif not result['correct'] and st.session_state.retry_count == 0:
                            st.session_state.retry_count += 1
                            st.error("That's not quite right. Would you like to try again?")
                            
                            # Create a container for retry button
                            retry_container = st.container()
                            with retry_container:
                                retry_col1, retry_col2, retry_col3 = st.columns([1,2,1])
                                with retry_col2:
                                    # Create a form for the retry button
                                    with st.form(key='retry_form'):
                                        if st.form_submit_button("Retry", type="primary"):
                                            st.session_state.canvas_key += 1
                                            st.rerun()
                        else:
                            # Log the response and proceed as normal
                            current_time = datetime.now(TIMEZONE)
                            log_response_to_sheet(client, current_question, llm_response, current_time)
                            
                            print("\n=== Debug: Processing response ===")
                            print(f"Result correct: {result['correct']}")
                            print(f"Result try_again: {result['try_again']}")
                            print(f"boost_gif in session state: {'boost_gif' in st.session_state}")
                            
                            # Clear any existing boost_gif
                            if 'boost_gif' in st.session_state:
                                print("Clearing existing boost_gif from session state")
                                st.session_state.pop('boost_gif')
                            
                            # Display success/error message
                            if result['correct']:
                                print("Processing correct answer")
                                st.success(result['user_message'])
                                print("\n=== Debug: Processing correct answer boost ===")
                                if DEBUG:
                                    st.write("Debug - Getting correct boost GIF")
                                boost_url = get_random_boost_gif("Correct", df_boosters, DEBUG)
                                print(f"Got boost URL: {boost_url}")
                                if boost_url:
                                    if DEBUG:
                                        st.write("Debug - Setting boost_gif in session state:", boost_url)
                                    st.session_state.boost_gif = boost_url
                                    print(f"Session state after setting boost: {st.session_state}")
                            else:
                                print("Processing incorrect answer")
                                st.error(result['user_message'])
                                print("\n=== Debug: Processing incorrect answer boost ===")
                                if DEBUG:
                                    st.write("Debug - Getting incorrect boost GIF")
                                boost_url = get_random_boost_gif("Incorrect", df_boosters, DEBUG)
                                print(f"Got boost URL: {boost_url}")
                                if boost_url:
                                    if DEBUG:
                                        st.write("Debug - Setting boost_gif in session state:", boost_url)
                                    st.session_state.boost_gif = boost_url
                                    print(f"Session state after setting boost: {st.session_state}")
                            
                            print("\n=== Debug: Before GIF display ===")
                            print(f"boost_gif in session state: {'boost_gif' in st.session_state}")
                            if 'boost_gif' in st.session_state:
                                print(f"boost_gif value: {st.session_state.boost_gif}")
                            
                            # Calculate if we should show boost, but store it in session state
                            if 'boost_gif' in st.session_state and st.session_state.boost_gif:
                                # For correct answers, show boost 1/5 times
                                st.session_state.should_show_boost = random.randint(1, SHOW_BOOST_FREQUENCY_CORRECT if result['correct'] else SHOW_BOOST_FREQUENCY_INCORRECT) == 1
                            
                            # Reset retry count for next question
                            st.session_state.retry_count = 0
                            
                            def increment_question():
                                if DEBUG:
                                    print("\n=== Debug: increment_question called ===")
                                    print(f"Before: current_question_index = {st.session_state.current_question_index}")
                                st.session_state.current_question_index += 1
                                st.session_state.canvas_key += 1
                                if DEBUG:
                                    print(f"After: current_question_index = {st.session_state.current_question_index}")
                                if 'boost_gif' in st.session_state:
                                    st.session_state.boost_gif = None
                            
                            # Create a container for consistent button placement
                            button_container = st.container()
                            with button_container:
                                col1, col2, col3 = st.columns([1,2,1])
                                with col2:
                                    if st.button("Next Question", 
                                               type="primary", 
                                               key=f"next_{st.session_state.current_question_index}",
                                               on_click=increment_question):
                                        if DEBUG:
                                            print("Button clicked, triggering rerun")
                                        st.rerun()
                            
                        status.update(label="Analysis complete!", state="complete", expanded=True)
                    
                    except Exception as e:
                        if DEBUG:
                            st.error(f"Error: {str(e)}")
                        st.error("An error occurred while processing your response")
                        status.update(label="Error occurred!", state="error", expanded=True)
            
            # Create a container at the bottom of the page for the boost GIF
            boost_container = st.container()
            with boost_container:
                # Display boost GIF if available and should be shown
                if ('boost_gif' in st.session_state and 
                    st.session_state.boost_gif and 
                    getattr(st.session_state, 'should_show_boost', False)):
                    print("\n=== Debug: Attempting to display boost GIF ===")
                    if DEBUG:
                        st.write("Debug - Attempting to display GIF:", st.session_state.boost_gif)
                        print(f"Current boost_gif value: {st.session_state.boost_gif}")
                    try:
                        st.image(st.session_state.boost_gif)
                        print("Image display attempted")
                    except Exception as e:
                        print(f"Error displaying image: {str(e)}")
                        if DEBUG:
                            st.error(f"Error displaying GIF: {str(e)}")
            
            # Add bottom spacing
            st.empty().markdown(f'<div style="height: {BOTTOM_SPACING_PX}px"></div>', unsafe_allow_html=True)
        
        else:
            if not active_questions or len(active_questions) == 0:
                st.info("No questions are due at this time.")
            elif st.session_state.current_question_index >= len(active_questions):
                st.success("Congratulations! You've completed all your practice questions!")
    
except Exception as e:
    if DEBUG:
        st.error(f"Error reading spreadsheet: {str(e)}")
        import traceback
        st.write("Full error:", traceback.format_exc())
    else:
        st.error("An error occurred while loading the questions.")
