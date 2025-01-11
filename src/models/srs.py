import pandas as pd
from datetime import datetime
import streamlit as st
import json
import gspread
from src.config.settings import DEBUG, SHEET_NAME_SRS_LOG, SHEET_NAME_SRS_NEXT, SRS_TIME_DELAYS, SPREADSHEET_URL
from src.utils.time_utils import calculate_next_timestamp
from src.utils.sheets import write_df_to_google_sheet, update_next_ask_timestamp


def toggle_study_mode():
    st.session_state.study_mode = not st.session_state.study_mode
    if not st.session_state.study_mode:
        st.session_state.active_questions = None
        st.session_state.current_question_index = 0


def get_available_sets(googlecreds, spreadsheet_url, sheet_name, data_range, TIMEZONE):
    try:
        # Set up the credentials
        client = gspread.authorize(googlecreds)
        sh = client.open_by_url(spreadsheet_url)
        worksheet = sh.worksheet(sheet_name)
        data = worksheet.get(data_range)
        
        if not data or len(data) < 2:
            return {}
            
        # Create DataFrame
        df = pd.DataFrame(data[1:], columns=['Prompt ID', 'Set', 'Prompt', 'Correct Answer', 'Confidence Level', 'Next Ask Timestamp'])
        df['Next Ask Timestamp'] = df['Next Ask Timestamp'].apply(lambda x: x.strip("'") if isinstance(x, str) else x)
        df['Next Ask Timestamp'] = pd.to_datetime(df['Next Ask Timestamp'], format='%Y-%m-%d %H:%M:%S')
        df['Next Ask Timestamp'] = df['Next Ask Timestamp'].dt.tz_localize('EET')
        
        # Get current time
        current_time = datetime.now(TIMEZONE)
        
        # Filter for due questions
        due_questions = df[df['Next Ask Timestamp'] <= current_time]
        
        # Group by Set and count
        set_counts = due_questions.groupby('Set').size().to_dict()
        
        return set_counts
        
    except Exception as e:
        st.error(f"Error reading sets: {str(e)}")
        return {}


def update_confidence_level(current_level, is_correct):
    """Update confidence level based on answer correctness"""
    if not is_correct:
        return 0
    return min(current_level + 1, 5)


def increment_question(DEBUG=False):
    if DEBUG:
        print("\n=== Debug: increment_question called ===")
        print(f"Before: current_question_index = {st.session_state.current_question_index}")
    st.session_state.current_question_index += 1
    st.session_state.canvas_key += 1
    if DEBUG:
        print(f"After: current_question_index = {st.session_state.current_question_index}")
    if 'boost_gif' in st.session_state:
        st.session_state.boost_gif = None


def log_response_to_sheet(gspread_client, question_data, llm_response, timestamp):
    """Log the response to the SRSLog sheet"""
    try:
        # Parse LLM response as JSON
        result = json.loads(llm_response)
        
        # Get current confidence level
        current_confidence = int(question_data['Confidence Level'])
        
        # Calculate new confidence level
        new_confidence = update_confidence_level(current_confidence, result['correct'])
        
        # Calculate next ask timestamp based on new confidence
        next_ask = calculate_next_timestamp(new_confidence, timestamp, SRS_TIME_DELAYS)
        next_ask_str = next_ask.strftime('%Y-%m-%d %H:%M:%S')
            
        # Prepare the log entry
        log_data = pd.DataFrame({
            'Prompt ID': [question_data['Prompt ID']],
            'Set': [question_data['Set']],
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
        write_df_to_google_sheet(gspread_client, 
                               SPREADSHEET_URL, 
                               SHEET_NAME_SRS_LOG,
                               log_data,
                               flag_append=True)
        
        # Update the Next Ask Timestamp and Confidence Level in SRSNext
        prompt_id = question_data['Prompt ID']
        update_next_ask_timestamp(gspread_client, SPREADSHEET_URL, SHEET_NAME_SRS_NEXT, prompt_id, next_ask, new_confidence, DEBUG)
        
        if DEBUG:
            print(f"Debug - Updated confidence level from {current_confidence} to {new_confidence}")
            
    except Exception as e:
        if DEBUG:
            print(f"Debug - Error logging response: {str(e)}")
        st.error(f"Error logging response: {str(e)}")
