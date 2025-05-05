from app import app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from flask import Flask

# Mount your app under /csv-analytics
application = DispatcherMiddleware(Flask('dummy'), {
    '/csv-analytics': app
})

if __name__ == '__main__':
    run_simple('0.0.0.0', 5001, application)
