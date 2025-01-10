import os
import yaml
import pytz

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Debug settings
DEBUG = True

# Google Sheets configuration
DATA_RANGE = "A:F"  # Prompt ID, Set, Prompt, Correct Answer, Confidence Level, Next Ask Timestamp
BOOSTERS_RANGE = "A:B"  # Range for Boost type and Boost URL columns

# Sheet names
SHEET_NAME_SRS_NEXT = "SRSNext"
SHEET_NAME_SRS_LOG = "SRSLog"
SHEET_NAME_BOOSTERS = "Boosters"

# Model settings
SELECTED_MODEL = config.get('model', None)

# Timezone settings
TIMEZONE = pytz.timezone('EET')

# Theme settings
ACTIVE_THEME = "theme2"  # Can be "theme1" or "theme2"

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

# Boost settings
SHOW_BOOST_FREQUENCY_CORRECT = 5  # Show boost 1 out of 5 times for correct answers
SHOW_BOOST_FREQUENCY_INCORRECT = 3  # Show boost 1 out of 3 times for incorrect answers

# SRS time delay configuration
SRS_TIME_DELAYS = [
    {"confidence_level": 0, "delay_quantity": 10, "delay_time_unit": 'minutes'},
    {"confidence_level": 1, "delay_quantity": 8,  "delay_time_unit": 'hours'},
    {"confidence_level": 2, "delay_quantity": 3,  "delay_time_unit": 'days'},
    {"confidence_level": 3, "delay_quantity": 5,  "delay_time_unit": 'days'},
    {"confidence_level": 4, "delay_quantity": 10, "delay_time_unit": 'days'},
    {"confidence_level": 5, "delay_quantity": 20, "delay_time_unit": 'days'},
]

# Get current theme
CURRENT_THEME = THEMES[ACTIVE_THEME]

# Spreadsheet configuration
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1NyaBvbHef_eX1lBYtTPzZJ2fBSPRG2yYxitTeEoJy-M/edit?gid=324250006#gid=324250006"
