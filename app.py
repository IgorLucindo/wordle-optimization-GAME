import streamlit as st
from frontend import wordle_game_component
import os
import subprocess

st.set_page_config(layout="centered", page_title="Streamlit Wordle")
st.title("Streamlit Wordle Game")

def run_main_py():
    script_path = os.path.join("application", "main.py")
    if os.path.exists(script_path):
        st.info(f"Running {script_path}...")
        try:
            # Execute the Python script and capture output
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                check=True # Raise an exception for non-zero exit codes
            )
            if result.stdout:
                st.code(result.stdout, language='python')
            if result.stderr:
                st.error(result.stderr)
            st.success("main.py executed successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"Error executing script (Exit Code {e.returncode}):")
            st.error(e.stderr)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.error(f"Error: {script_path} not found.")

# Get the value from the custom component
component_return_value = wordle_game_component(key="wordle_game_instance")

st.write("---")
st.write("This is your Streamlit app content below the game.")