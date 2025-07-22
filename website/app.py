import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from application.utils.instance_utils import *
from application.utils.solve_utils import *
from application.utils.wordle_tools_utils import *
from flask import Flask, request, jsonify, session, send_from_directory, render_template


# Create app
app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Set absolute paths
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/dataset/<filename>')
def get_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/create-model', methods=['POST'])
def run_create_model():
    # Get instance
    instance = get_instance()

    # Store in session
    session['instance'] = instance

    return

@app.route('/solve-model', methods=['POST'])
def run_solve_model():
    data = request.get_json()
    instance = session.get('instance')

    # Filer instance
    instance = fiter_instance(instance, data['gameResults'])

    # Solve
    word_guess = solve(instance)

    return jsonify({"wordGuess": word_guess})

if __name__ == "__main__":
    app.run(debug=True)