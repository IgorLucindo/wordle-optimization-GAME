import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from application.main import main
from flask import Flask, request, jsonify, send_from_directory, render_template


# Create app
app = Flask(__name__)

# Set absolute paths
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/dataset/<filename>')
def get_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/solve-wordle', methods=['POST'])
def run_python_script():
    try:
        data = request.get_json()
        flags = {
            "test": False,
            'flask': True
        }
        word_guess = main(flags, data['gameResults'])

        return jsonify({"wordGuess": word_guess})
    except Exception as e:
        import traceback
        print("❌ Exception occurred in /solve-wordle:")
        traceback.print_exc()  # ← this shows full stack trace in terminal
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)