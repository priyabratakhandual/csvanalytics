# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install Supervisor
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 5001

# Copy Supervisor configuration
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf

# Command to run Supervisor (which will start Flask via WSGI)
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
