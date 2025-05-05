import re
import pandas as pd

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_column_names(columns):
    # Initialize list to hold unique column names and a dictionary to track occurrences
    unique_columns = []
    seen = {}

    # Loop through each column name
    for col in columns:
        # Check if the column name has been seen before
        if col in seen:
            # If so, increment the count and append the count to the column name
            seen[col] += 1
            new_col = f"{col}_{seen[col]}"
        else:
            # If not, start tracking this column name
            seen[col] = 0
            new_col = col

        # Append the new unique column name to the list
        unique_columns.append(new_col)

    return unique_columns