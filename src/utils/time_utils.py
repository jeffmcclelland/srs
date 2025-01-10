from datetime import timedelta
import gspread
import streamlit as st


def calculate_next_timestamp(confidence_level, current_time, SRS_TIME_DELAYS):
    """Calculate next ask timestamp based on confidence level"""
    delay_config = next(d for d in SRS_TIME_DELAYS if d['confidence_level'] == confidence_level)
    
    if delay_config['delay_time_unit'] == 'minutes':
        delta = timedelta(minutes=delay_config['delay_quantity'])
    elif delay_config['delay_time_unit'] == 'hours':
        delta = timedelta(hours=delay_config['delay_quantity'])
    else:  # days
        delta = timedelta(days=delay_config['delay_quantity'])
    
    return current_time + delta


def update_next_ask_timestamp(googlecreds, spreadsheet_url, sheet_name, prompt_id, next_ask_timestamp, confidence_level, DEBUG=False):
    """Update both Next Ask Timestamp and Confidence Level for a specific row"""
    try:
        if DEBUG:
            print(f"Debug - Updating row {prompt_id} with timestamp {next_ask_timestamp} and confidence {confidence_level}")
            
        client = gspread.authorize(googlecreds)
        sh = client.open_by_url(spreadsheet_url)
        worksheet = sh.worksheet(sheet_name)
        
        # Find the correct row by Prompt ID
        cell = worksheet.find(str(prompt_id), in_column=1)  # Search in first column (Prompt ID)
        if cell:
            actual_row = cell.row
            # Convert `next_ask_timestamp` to string
            next_ask_timestamp_str = next_ask_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            # Update Confidence Level (Column E) and Next Ask Timestamp (Column F)
            worksheet.batch_update([
                {'range': f'E{actual_row}', 'values': [[confidence_level]]},
                {'range': f'F{actual_row}', 'values': [[next_ask_timestamp_str]]}
            ])
            
            if DEBUG:
                print(f"Debug - Found row {actual_row} for Prompt ID {prompt_id}")
                print("Debug - Update successful")
        else:
            raise Exception(f"Could not find row with Prompt ID {prompt_id}")
            
    except Exception as e:
        if DEBUG:
            print(f"Debug - Error updating row: {str(e)}")
            st.error(f"Error updating row: {str(e)}")
