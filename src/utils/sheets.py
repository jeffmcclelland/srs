"""Google Sheets utility functions"""

import gspread
import streamlit as st
from src.config.settings import DEBUG


def write_df_to_google_sheet(gspread_client, 
                           spreadsheet_url, 
                           spreadsheet_sheet_name, 
                           df, 
                           start_cell='A1', 
                           clear_sheet=False, 
                           flag_append=False):
    """Write a DataFrame to a Google Sheet"""
    try:
        # Get the spreadsheet and worksheet
        spreadsheet = gspread_client.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet(spreadsheet_sheet_name)
        
        if DEBUG:
            print(f"\nWriting to sheet '{spreadsheet_sheet_name}'")
            print(f"Data to write:\n{df}")
        
        # Fill NaN values with empty strings and downcast types
        df = df.fillna('').infer_objects(copy=False)

        # Convert Timestamp columns to string
        for column in df.select_dtypes(include=['datetime64[ns]']).columns:
            df[column] = df[column].astype(str)
        
        # If clear_sheet is True, clear the entire sheet first
        if clear_sheet:
            worksheet.clear()
            if DEBUG:
                print("Sheet cleared")
        
        # Prepare data based on append mode
        if flag_append:
            data = df.values.tolist()  # Exclude header for appending
        else:
            data = [df.columns.values.tolist()] + df.values.tolist()  # Include header
        
        if DEBUG:
            print(f"Data prepared for writing: {len(data)} rows")
        
        if not flag_append:
            # If not appending, write data starting from start_cell
            worksheet.update(values=data, range_name=start_cell)
        else:
            # To append data, find the first empty row
            all_values = worksheet.get_all_values()
            first_empty_row = len(all_values) + 1  # Next empty row after the last filled row
            start_cell = f"A{first_empty_row}"
            
            if DEBUG:
                print(f"Appending at row: {first_empty_row}")

            # Update starting at the next empty cell calculated
            worksheet.update(values=data, range_name=start_cell)
    
        if DEBUG:
            print(f"Finished writing to sheet '{spreadsheet_sheet_name}' - Append Mode: {'Yes' if flag_append else 'No'}")
            
    except Exception as e:
        if DEBUG:
            print(f"Error writing to sheet: {str(e)}")
        raise e


def update_next_ask_timestamp(gspread_client, spreadsheet_url, sheet_name, prompt_id, next_ask_timestamp, confidence_level, DEBUG=False):
    """Update both Next Ask Timestamp and Confidence Level for a specific row"""
    try:
        if DEBUG:
            print(f"Debug - Updating row {prompt_id} with timestamp {next_ask_timestamp} and confidence {confidence_level}")
            
        sh = gspread_client.open_by_url(spreadsheet_url)
        worksheet = sh.worksheet(sheet_name)
        
        # Find the correct row by Prompt ID
        cell = worksheet.find(str(prompt_id), in_column=1)  # Search in first column (Prompt ID)
        if cell:
            actual_row = cell.row
            # Convert `next_ask_timestamp` to string
            next_ask_timestamp_str = next_ask_timestamp.strftime('%Y-%m-%d')
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
