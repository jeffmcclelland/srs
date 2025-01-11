from datetime import timedelta

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
