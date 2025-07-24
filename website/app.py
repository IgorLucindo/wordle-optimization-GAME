import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from application.utils.instance_utils import *
from application.utils.solve_utils import *
from application.utils.wordle_tools_utils import *
from flask import Flask, request, jsonify, send_from_directory, render_template


# Create app
app = Flask(__name__)
instance = None
word_guess = None


# Set absolute paths
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/dataset/<filename>')
def get_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/create-instance', methods=['POST'])
def run_create_instance():
    global instance

    # Get instance
    instance = get_instance()

    return {}

@app.route('/solve', methods=['POST'])
def run_solve():
    global instance, word_guess
    data = request.get_json()

    # Filer instance
    instance = fiter_instance(instance, data['guessResults'])

    # Solve
    word_guess = solve_random(instance)

    return {}

@app.route('/get-guess', methods=['POST'])
def run_get_guess():
    global word_guess

    return jsonify({"wordGuess": word_guess})


if __name__ == "__main__":
    app.run(debug=True)