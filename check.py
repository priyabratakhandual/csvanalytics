import logging
import re
import pandas as pd
import numpy as np


def identify_and_convert_datetime_columns(data):
    # Define common datetime patterns to check against
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # dd/mm/yyyy or mm/dd/yyyy
        r'\d{4}-\d{1,2}-\d{1,2}',  # yyyy-mm-dd
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # dd-mm-yyyy or mm-dd-yyyy
        r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}',  # dd/mm/yyyy hh:mm:ss
        r'\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{2}:\d{2}',  # yyyy-mm-dd hh:mm:ss
        r'\d{1,2}-\d{1,2}-\d{4} \d{1,2}:\d{2}:\d{2}',  # dd-mm-yyyy hh:mm:ss
        r'\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2}',  # dd/mm/yyyy hh:mm
        r'\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{2}',  # yyyy-mm-dd hh:mm
        r'\d{1,2}-\d{1,2}-\d{4} \d{1,2}:\d{2}',  # dd-mm-yyyy hh:mm
        r'\d{2}/\d{2}/\d{2} \d{2}:\d{2} [APM]{2}',  # mm/dd/yy hh:mm AM/PM
        r'\d{2}-\d{2}-\d{2} \d{2}:\d{2} [APM]{2}',  # yy-mm-dd hh:mm AM/PM
    ]

    # Common datetime formats to attempt for conversion
    formats = [
        '%d-%m-%Y %H:%M',
        '%d/%m/%Y %H:%M',
        '%Y-%m-%d %H:%M',
        '%d-%m-%Y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y',
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%m/%d/%Y',
        '%m-%d-%Y',
        '%Y/%m/%d',
        '%d-%b-%Y',
        '%d %b %Y',
        '%d %B %Y',
        '%Y %B %d',
    ]

    for column in data.columns:
        if data[column].dtype == 'object':  # Only apply to object columns
            column_values = data[column].astype(str)

            # Check if the column contains mostly numeric values
            if column_values.str.isnumeric().mean() > 0.5:
                logging.info(f"Column '{column}' contains mostly numeric values. Skipping datetime conversion.")
                continue  # Skip columns that are mostly numeric

            # Log the start of processing for the column
            logging.info(f"Processing column '{column}' for potential datetime conversion.")

            # Check if the column has at least 80% values matching datetime patterns
            matched_count = sum(
                column_values.apply(lambda x: any(re.match(pattern, x) for pattern in patterns))
            )
            total_count = len(column_values.dropna())
            match_percentage = (matched_count / total_count) * 100 if total_count > 0 else 0

            logging.info(f"Column '{column}' has {matched_count} matching values out of {total_count} (match percentage: {match_percentage:.2f}%).")

            if match_percentage >= 80:
                logging.info(f"Column '{column}' matches datetime patterns sufficiently (>= 80%). Attempting conversion.")

                for format in formats:
                    try:
                        temp_conversion = pd.to_datetime(data[column], format=format, errors='coerce')
                        if not temp_conversion.isna().all():
                            data[column] = temp_conversion
                            logging.info(f"Column '{column}' converted to datetime with format '{format}'.")
                            break
                    except Exception as e:
                        logging.warning(f"Error trying format '{format}' for column '{column}': {e}")

                # If conversion was successful, handle NaT values with forward fill
                if pd.api.types.is_datetime64_any_dtype(data[column]):
                    data[column] = data[column].fillna(method='ffill')
                    logging.info(f"Column '{column}' has NaT values filled using forward fill.")
                else:
                    logging.warning(f"Could not convert column '{column}' to datetime.")

    return data

def convert_bool_columns(data):
    for column in data.columns:
        if data[column].dtype == 'object':  # Check if column is of object type
            try:
                # Convert to boolean only if all values are 'true' or 'false'
                if data[column].str.lower().isin(['true', 'false']).all():
                    data[column] = data[column].map({'True': True, 'False': False, 'true': True, 'false': False})
                    logging.info(f"Column '{column}' converted to boolean.")
            except Exception as e:
                logging.warning(f"Could not convert column '{column}' to boolean: {e}")

        # This check must be outside the try block
        if np.issubdtype(data[column].dtype, np.bool_):
            try:
                data[column] = data[column].astype(bool)  # Ensure standard Python boolean
                logging.info(f"Column '{column}' converted to standard boolean type.")
            except Exception as e:
                logging.warning(f"Could not convert column '{column}' to standard boolean type: {e}")

    return data
