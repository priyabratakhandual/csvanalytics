from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pandas as pd
import logging, re
import os
from datetime import datetime
from utils import allowed_file, clean_column_names
from db import get_collection
from werkzeug.utils import secure_filename
from datetime import timedelta
from bson import ObjectId
from check import identify_and_convert_datetime_columns, convert_bool_columns
import uuid
import chardet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pytz
import json
from prophet import Prophet
from sklearn.metrics import r2_score
#import redis
# from flask_compress import Compress

#REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
#REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
#REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
#redis_client = redis.StrictRedis(
    #host=REDIS_HOST,
    #port=REDIS_PORT,
    #password=REDIS_PASSWORD,
    #decode_responses=True  # Get string responses instead of bytes
#)
#print(redis_client)
# print(f'************{redis_client.ping()}')


app = Flask(__name__)
# compress = Compress(app)
app.secret_key = 'GyanIB088SInghyyyya'
app.config['SESSION_COOKIE_NAME'] = 'session_id'
app.config['SESSION_COOKIE_SAMESITE'] = 'None'  # Allow cross-site cookie usage
app.config['SESSION_COOKIE_SECURE'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
cors = CORS(app, resources={r"/*": {"origins": "*"}},supports_credentials=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

upload_in_progress = False

@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Generate a unique user ID using UUID
    logging.debug(f"User ID set to: {session['user_id']}")

@app.route('/csv-analytics/debug_session', methods=['GET'])
def debug_session():
    return jsonify({
        'user_id': session.get('user_id', 'No user_id set'),
    })

@app.route('/csv-analytics/ping')
def ping():
    logging.info("Ping request received")
    return "PONG"

def detect_encoding(file):
    try:
        # Read a portion of the file to detect encoding
        raw_data = file.read(10000)  # Read a chunk of the file
        result = chardet.detect(raw_data)
        file.seek(0)  # Reset file pointer to the beginning

        # Ensure detected encoding is not None and not 'ascii' (commonly leads to issues)
        encoding = result.get('encoding') if result.get('encoding') and result.get('encoding').lower() != 'ascii' else 'utf-8'

        # Try decoding with the detected encoding
        try:
            file.read().decode(encoding)
            file.seek(0)  # Reset file pointer to the beginning
            return encoding
        except (UnicodeDecodeError, LookupError):
            logging.warning(f"Detected encoding '{encoding}' failed. Trying 'utf-8'...")
            return handle_fallback_encoding(file)
    except Exception as e:
        logging.error(f"Error detecting file encoding: {e}")
        file.seek(0)
        return 'utf-8'  # Default to 'utf-8' if an error occurs

def handle_fallback_encoding(file):
    try:
        # Try decoding with 'utf-8'
        file.seek(0)
        file.read().decode('utf-8')
        file.seek(0)
        return 'utf-8'
    except (UnicodeDecodeError, LookupError):
        logging.warning(f"'utf-8' failed. Trying 'latin-1'...")
        try:
            # Try decoding with 'latin-1'
            file.seek(0)
            file.read().decode('latin-1')
            file.seek(0)
            return 'latin-1'
        except (UnicodeDecodeError, LookupError):
            logging.error("Failed to decode with 'utf-8' and 'latin-1'. Defaulting to 'windows-1252'.")
            return 'windows-1252'  # Default to 'windows-1252' if both fail

def check_and_convert_large_integers(record):
    for key, value in record.items():
        if isinstance(value, int):
            if value > (2**63 - 1) or value < -(2**63):  # MongoDB int64 range
                logging.warning(f"Integer value {value} is too large for MongoDB. Converting to string.")
                record[key] = str(value)  # Convert large integers to strings
    return record


@app.route('/csv-analytics/upload', methods=['POST'])
def upload_csv():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User not authenticated'}), 401

    file_tracking_collection = get_collection('file_uploads')

    # Limit the number of file uploads per user
    user_file_count = file_tracking_collection.count_documents({'user_id': user_id})
    if user_file_count >= 5:
        return jsonify({'error': 'Maximum file upload limit reached for this user.'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check the file size
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)

    max_size_mb = 100
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_length > max_size_bytes:
        return jsonify({'error': f'File size exceeds the {max_size_mb}MB limit.'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)

            # Check for duplicate filenames
            if file_tracking_collection.count_documents({'user_id': user_id, 'filename': filename}) > 0:
                return jsonify({'error': 'File with this name has already been uploaded.'}), 400

            # Detect the encoding of the file
            encoding = detect_encoding(file)

            # Reset the file pointer before reading
            file.seek(0)

            # Convert Excel files to CSV format
            if filename.endswith('.xlsx'):
                data = pd.read_excel(file)
                filename = filename.rsplit('.', 1)[0] + '.csv'  # Change filename extension to .csv
            elif filename.endswith('.csv'):
                data = pd.read_csv(file, encoding=encoding)
            else:
                return jsonify({'error': 'Unsupported file type'}), 400

            # Check if the DataFrame is empty

            if data.empty:
                return jsonify({'error': 'The uploaded file is blank. Please upload a file with data.'}), 400

            # Clean up the data
            data.columns = data.columns.str.strip()
            data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            data = data.applymap(lambda x: x.title() if isinstance(x, str) else x)
            data = data.where(pd.notnull(data), "Blank")

            data = identify_and_convert_datetime_columns(data)
            data = convert_bool_columns(data)
            # data = check_and_convert_time_columns(data)

            # Convert large integers to strings if necessary
            records = data.to_dict(orient='records')
            for record in records:
                record['user_id'] = user_id
                record['filename'] = filename
                record = check_and_convert_large_integers(record)

            # Insert records into MongoDB
            data_collection = get_collection('data')
            data_collection.insert_many(records)
            redis_client.delete(f"data:{user_id}:*")

            # Track the file upload
            file_tracking_collection.insert_one({
                'user_id': user_id,
                'filename': filename,
                'upload_time': datetime.utcnow()
            })

            return jsonify({'message': 'File uploaded and processed successfully.'}), 200

        except Exception as e:
            logging.error(f"An error occurred during file upload: {e}")
            return jsonify({'error': 'An error occurred during file upload. Please check the file and try again.'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/csv-analytics/files', methods=['GET'])
def get_uploaded_files():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User not identified'}), 400

        # # Check if files are cached in Redis
        # cached_files = redis_client.get(f"files_{user_id}")
        # print(cached_files)
        # if cached_files:
        #     return jsonify({"files": json.loads(cached_files)}), 200  # Return cached data if available

        file_collection = get_collection('file_uploads')
        data_collection = get_collection('data')

        # Get metadata for uploaded files
        files_metadata = list(file_collection.find({'user_id': user_id}, {'_id': 0}))
        ist = pytz.timezone('Asia/Kolkata')

        result = []
        for file_metadata in files_metadata:
            filename = file_metadata.get('filename')
            upload_time_utc = file_metadata.get('upload_time')

            # Convert the upload_time to a string if it exists and is a datetime object
            if isinstance(upload_time_utc, datetime):
                upload_time_ist = upload_time_utc.astimezone(ist)
                upload_time_str = upload_time_ist.strftime('%Y-%m-%d %H:%M:%S')
            else:
                upload_time_str = None

            # Retrieve top 6 rows of the file's data
            top_6_data = pd.DataFrame(
                list(data_collection.find({'user_id': user_id, 'filename': filename}, {'_id': 0})))

            if not top_6_data.empty:
                # Ensure that the data is serializable (convert datetime columns to string if needed)
                top_6_data = top_6_data.head(6).reset_index(drop=True)
                for col in top_6_data.select_dtypes(include=['datetime']).columns:
                    top_6_data[col] = top_6_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')

                top_6_data = top_6_data.to_dict(orient='records')

            # Append the result for this file
            result.append({
                'filename': filename,
                'upload_time': upload_time_str,
                'top_6_data': top_6_data
            })

        # Sort results by upload_time in reverse order (most recent first)
        result.sort(key=lambda x: x['upload_time'], reverse=True)

        # Cache result in Redis for 1 hour (3600 seconds)
        # redis_client.setex(f"files_{user_id}", 3600, json.dumps(result))

        return jsonify({"files": result}), 200
    except Exception as e:
        logging.error(f"Error occurred while retrieving files: {e}")
        return jsonify({"error": "An error occurred while retrieving files"}), 500


@app.route('/csv-analytics/analyze', methods=['GET'])
def analyze_data():
    logging.info("Analysis request received")

    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({"error": "Filename not provided"}), 400

        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User not identified'}), 400

        cache_key = f"analyze_{user_id}_{filename}"
        cached_analysis = redis_client.get(cache_key)
        print(cached_analysis)
        if cached_analysis:
            return jsonify(json.loads(cached_analysis)), 200

            # Fetch the specific file's data from MongoDB
        collection = get_collection('data')
        data = pd.DataFrame(list(collection.find({'user_id': user_id, 'filename': filename}, {'_id': 0})))

        if data.empty:
            return jsonify({"error": "No data available for the specified file."}), 400

        # Clean column names and store the cleaned names
        cleaned_columns = clean_column_names(data.columns)
        data.columns = cleaned_columns

        # Replace "Blank" with None for consistency in analysis
        data_cleaned = data.replace("Blank", None)

        # Calculate total null values for each column
        total_null_values = data_cleaned.isnull().sum().to_dict()

        # Initialize dictionaries for data types and statistics
        data_types = {}
        text_values = {}

        # Define the convert_to_serializable function
        def convert_to_serializable(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.ndarray, list)):
                return obj.tolist() if isinstance(obj, np.ndarray) else obj
            elif isinstance(obj, (np.bool_, bool)):  # Handle boolean values
                return bool(obj)  # Convert numpy.bool_ to Python bool
            elif pd.isna(obj):
                return None
            else:
                return obj

        for column in data_cleaned.columns:
            # Check if the column contains all numeric values (excluding any NaN values)
            if data_cleaned[column].dropna().apply(
                    lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)).all():
                data_types[column] = "Numeric"
            else:
                # Now check if the column contains valid datetime-like values
                converted_dates = pd.to_datetime(data_cleaned[column], errors ='coerce')
                # Only consider it a Date if more than 60% of non-NaN values are valid dates
                if converted_dates.notna().sum() > (0.6 * len(data_cleaned[column].dropna())):
                    data_types[column] = "Date"
                else:
                    # If not numeric or date, treat it as text
                    data_types[column] = "Text"
                    text_values[column] = data_cleaned[column].dropna().unique().tolist()

        # Calculate unique values count for each column
        unique_values_count = data_cleaned.nunique().to_dict()

        # Calculate total number of rows
        total_rows = len(data)

        # Generate insights
        insights = {
            "columns": list(data_cleaned.columns),  # Maintain the cleaned order
            "data_types": data_types,
            "unique_values_count": {key: convert_to_serializable(value) for key, value in unique_values_count.items()},
            "total_rows": convert_to_serializable(total_rows),
            "null_values": {key: convert_to_serializable(value) for key, value in total_null_values.items()},
            "text_values": text_values
        }
        redis_client.setex(cache_key, 86400, json.dumps(insights))

        return jsonify(insights), 200

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}")
        return jsonify({"error": f"An error occurred during analysis: {e}"}), 500


@app.route('/csv-analytics/top10', methods=['GET'])
def top_10_values():
    logging.info("Request received for top 10 values calculation")

    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({"error": "Filename not provided"}), 400

        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User not identified'}), 400

        cache_key = f"top10:{user_id}:{filename}"
        cached_data = redis_client.get(cache_key)

        # Log cached data for debugging
        if cached_data:
            logging.info(f"Cached data before JSON decode: {cached_data}")
            try:
                return jsonify(json.loads(cached_data)), 200  # Safely parse JSON
            except json.JSONDecodeError as json_error:
                logging.error(f"JSON decode error: {json_error}")
                # Optionally clear the cache for this key
                redis_client.delete(cache_key)
                return jsonify({"error": "Invalid cached data format"}), 500

        # Fetch the specific file's data from MongoDB
        collection = get_collection('data')
        data = pd.DataFrame(list(collection.find({'user_id': user_id, 'filename': filename}, {'_id': 0})))

        if data.empty:
            return jsonify({"error": "No data available for the specified file."}), 400

        # Initialize result list
        top_10_values = []

        for column in data.columns:
            if column.startswith('_'):
                continue

            # Ensure the column exists and is 1-dimensional
            if column not in data.columns or not pd.api.types.is_scalar(data[column].iloc[0]):
                logging.warning(f"Column '{column}' is not 1-dimensional or does not exist.")
                continue

            # Handle non-blank values
            non_blank_values = data[column][data[column] != "Blank"]

            # Ensure we handle empty columns gracefully
            if non_blank_values.empty:
                top_10_values.append({
                    "column": column,
                    "top_10_values": []
                })
                continue

            # Calculate top 10 values and percentages
            value_counts = non_blank_values.value_counts().head(10)
            total_count = len(non_blank_values)

            # Convert boolean values to string if necessary
            if all(isinstance(value, bool) for value in value_counts.index):
                value_counts.index = value_counts.index.map(lambda x: str(x))

            top_10_values.append({
                "column": column,
                "top_10_values": [
                    {
                        "value": str(value),
                        "count": int(count),
                        "percentage": (count / total_count) * 100
                    }
                    for value, count in value_counts.items()
                ]
            })

        # Cache the JSON string properly
        redis_client.setex(cache_key, 3600, json.dumps(top_10_values))

        return jsonify(top_10_values), 200

    except Exception as e:
        logging.error(f"An error occurred while calculating top 10 values: {e}")
        return jsonify({"error": f"An error occurred while calculating top 10 values: {e}"}), 500


@app.route('/csv-analytics/get_data/<filename>/<int:page>/<int:per_page>', methods=['GET'])
def get_data(filename, page, per_page):
    logging.info(f"Request received for data retrieval of file: {filename}, page: {page}, per_page: {per_page}")

    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User not identified'}), 400

        collection = get_collection('data')
        data = pd.DataFrame(list(collection.find({'user_id': user_id, 'filename': filename}, {'_id': 0})))

        cache_key = f"data:{user_id}:{filename}:page:{page}:per_page:{per_page}"
        cached_data = redis_client.get(cache_key)

        if cached_data:
            logging.info(f"Returning cached data for {cache_key}")
            try:
                return jsonify(json.loads(cached_data)), 200
            except json.JSONDecodeError as json_error:
                logging.error(f"JSON decode error: {json_error}")
                return jsonify({"error": "Invalid cached data format"}), 500

        if data.empty:
            return jsonify({"error": "No data available for the specified file."}), 400

        cleaned_columns = session.get('cleaned_columns')
        if cleaned_columns:
            data.columns = cleaned_columns
            data = data[cleaned_columns]

        bool_cols = data.select_dtypes(include=['bool']).columns
        data[bool_cols] = data[bool_cols].astype(str)

        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        paginated_data = data[start:end]

        for col in paginated_data.select_dtypes(include=['datetime64[ns]']).columns:
            paginated_data[col] = paginated_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        paginated_data_json = paginated_data.to_dict(orient='records')

        # Log before caching to debug the format
        logging.info(f"Caching data for {cache_key}: {paginated_data_json}")
        redis_client.set(cache_key, json.dumps(paginated_data_json))  # Cache as JSON without expiration
        logging.info(f"Data cached successfully for {cache_key}")

        return jsonify(paginated_data_json), 200

    except Exception as e:
        logging.error(f"An error occurred during data retrieval: {e}")
        return jsonify({"error": f"An error occurred during data retrieval: {e}"}), 500


@app.route('/csv-analytics/delete', methods=['DELETE'])
def delete_file():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User not identified'}), 400

    # Get filename from query parameters
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    try:
        # Delete file metadata from file_tracking_collection
        file_tracking_collection = get_collection('file_uploads')
        file_tracking_collection.delete_many({'user_id': user_id, 'filename': filename})

        # Delete file data from data collection
        data_collection = get_collection('data')
        data_collection.delete_many({'user_id': user_id, 'filename': filename})

        return jsonify({'message': 'File deleted successfully.'}), 200
    except Exception as e:
        logging.error(f"An error occurred while deleting file: {e}")
        return jsonify({'error': 'An error occurred while deleting the file'}), 500

@app.route('/csv-analytics/forecast', methods=['GET'])
def forecast_data():
    logging.info("Forecasting request received")

    try:
        filename = request.args.get('filename')
        date_column = request.args.get('date_column')
        target_column = request.args.get('target_column')
        issue_column = request.args.get('issue_column')
        issue_type = request.args.get('issue_type')

        # Validate input parameters
        if not filename or not date_column or not target_column or not issue_column or not issue_type:
            return jsonify({"error": "Filename, Date column, Target column, Issue column, and Issue type must be provided"}), 400

        cache_key = f"forecast:{filename}:{date_column}:{target_column}:{issue_column}:{issue_type}"
        cached_forecast = redis_client.get(cache_key)

        if cached_forecast:
            logging.info(f"Returning cached forecast result for {cache_key}")
            try:
                return jsonify(json.loads(cached_forecast)), 200
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error for cached data: {e}")
                redis_client.delete(cache_key)  # Clear corrupted cache

        collection = get_collection('data')
        data = pd.DataFrame(list(collection.find({'filename': filename}, {'_id': 0})))

        if data.empty:
            return jsonify({"error": "No data available for the specified file."}), 400

        new_columns = clean_column_names(data.columns)
        data.columns = new_columns

        # Validate presence of required columns
        for col in [date_column, target_column, issue_column]:
            if col not in new_columns:
                return jsonify({"error": f"Column '{col}' not found"}), 400

        # Prepare the data
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data = data.dropna(subset=[date_column])
        data.set_index(date_column, inplace=True)

        issue_data = data[data[issue_column] == issue_type]

        if issue_data.empty:
            return jsonify({"error": f"No data available for the selected issue type: {issue_type}"}), 400

        issue_data[target_column] = pd.to_numeric(issue_data[target_column], errors='coerce')
        issue_data = issue_data.dropna(subset=[target_column])
        daily_data = issue_data[target_column].resample('D').count().reset_index()
        daily_data.rename(columns={date_column: 'ds', target_column: 'y'}, inplace=True)

        if len(daily_data) < 3:
            return jsonify({"error": "Insufficient data for forecasting"}), 400

        # Create the Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.04
        )
        model.add_seasonality(name='custom_seasonality', period=30.5, fourier_order=4)
        model.fit(daily_data)

        future = model.make_future_dataframe(periods=180, freq='D')
        forecast = model.predict(future)

        # Calculate R² value
        y_true = daily_data['y'].values
        y_pred = model.predict(daily_data)['yhat'].values
        r_squared = r2_score(y_true, y_pred)

        last_date_in_data = daily_data['ds'].max()
        forecast_next_month = forecast[forecast['ds'] > last_date_in_data][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(180)

        forecast_next_month['date'] = forecast_next_month['ds'].dt.strftime('%Y-%m-%d')
        forecast_next_month = forecast_next_month.drop(columns=['ds'])

        forecast_dict = {
            row['date']: {
                'average': round(row['yhat'], 2),
                'lower': round(row['yhat_lower'], 2),
                'upper': round(row['yhat_upper'], 2)
            }
            for _, row in forecast_next_month.iterrows()
        }

        response = {
            "6_month": forecast_dict,
            "original_trend": {
                "dates": daily_data['ds'].dt.strftime("%Y-%m-%d").tolist(),
                "values": daily_data['y'].tolist()
            },
            "r_squared": r_squared * 100
        }

        # Log the response before caching
        logging.info(f"Caching forecast result for {cache_key}: {response}")
        redis_client.set(cache_key, json.dumps(response))  # Cache as JSON without expiration

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"An error occurred during forecasting: {e}")
        return jsonify({"error": f"An error occurred during forecasting: {e}"}), 500

if __name__ == "__main__":
    app.run()