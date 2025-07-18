from flask import Flask, send_from_directory, render_template
import os


app = Flask(__name__)

# Set absolute paths
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/dataset/<filename>')
def get_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)