@tool
def get_current_date_tool():
    """Returns the current date in 'YYYY-MM-DD' format. Useful for finding flights/hotels relative to today."""
    return date.today().isoformat()