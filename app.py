import logging, os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from config import Config
from lib.predict_util import CAM2

if not os.path.exists('./logs'):
    os.makedirs('logs')
logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename='./logs/log')
logger.addHandler(file_handler)

model = CAM2(Config())

def model_api(sentence):
    result = model.get_visualized_scores(sentence)
    return {'output': result}

app = Flask(__name__)
CORS(app)

# default route
@app.route('/')
def index():
    return render_template('index.html')

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

# API route
@app.route('/api', methods=['POST'])
def api():
    input_data = request.json
    output_data = model_api(input_data)
    logger.info(msg=request.remote_addr + "\u241E" + "\u241E".join([key + ":" + output_data[key] for key in output_data.keys()]))
    response = jsonify(output_data)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')
