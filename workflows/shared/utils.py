def check_ok(text):
    # Split the text into words
    words = text.strip().split()

    # Check if there are any words
    if not words:
        return False

    # Check first and last words
    first_word = words[0]
    last_word = words[-1]

    # Return True if either first or last word is "Ok." or "Ok,"
    return first_word in ["Ok.", "Ok"] or last_word in ["Ok.", "Ok"]
