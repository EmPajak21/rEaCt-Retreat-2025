import streamlit as st
import os
from io import StringIO
import importlib
import sys
import unittest
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

def save_valid_submission(code, team_name, bucket_name="ddo_hackathon"):
    credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
    client = storage.Client(project="intense-pixel-446617-e2", credentials=credentials)  # pass project if needed
    bucket = client.bucket(bucket_name)

    # Create the file name for the submission
    file_name = f"{team_name}_submission.py"
    blob = bucket.blob(file_name)

    # Upload the code to the bucket
    blob.upload_from_string(code, content_type="text/x-python")
    return file_name

def run_unit_tests(code, team_name):
    #st.write("Starting unit tests...")
    buffer = StringIO()
    sys.stdout = buffer

    temp_filepath = f"{team_name}_temp_submission.py"

    try:
        with open(temp_filepath, "w") as f:
            f.write(code)
        #st.write(f"Temporary file path: {temp_filepath}")

        spec = importlib.util.spec_from_file_location("student_submission", temp_filepath)
        student_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(student_module)

        # Register the loaded module in sys.modules
        sys.modules["student_submission"] = student_module

        try:
            spec = importlib.util.spec_from_file_location("admin/test_framework", "admin/test_framework.py")
            test_framework = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_framework)

            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_framework)
            runner = unittest.TextTestRunner(stream=buffer, verbosity=2)
            result = runner.run(suite)

            errors = result.errors
            failures = result.failures

            if not errors and not failures:
                #st.write("All tests passed successfully!")
                return "All tests passed successfully! 🎉"

            #output = ["### Errors ###" if errors else "", "\n".join(f"{test}: {error.splitlines()[-1]}" for test, error in errors)]
            output = ["\n### Failures ###" if failures else "", "\n".join(f"{test}: {failure.splitlines()[-1]}" for test, failure in failures)]
            return "\n".join(filter(None, output))
        except Exception as e:
            return f"Error loading test framework: {e}"

        except Exception as e:
            return f"An error occurred while creating temp file or writing to it:{e}"
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            #st.write("Temporary file cleaned up.")
        sys.stdout = sys.__stdout__

# Tabs for Submission and Leaderboard
tabs = st.tabs(["Submission Portal", "Leaderboard"])

# Submission Tab
with tabs[0]:
    st.title("Student Submission Portal")

    status_container = st.container()

    team_name = st.text_input("Enter your team name:", help="Enter your team's name before submitting code.")

    st.markdown("### Paste Your Python Code Below:")
    code_input = st.text_area("Python Code", "", height=600, key="code_input")

    submit_button = st.button("Submit")

    if submit_button:
        if not team_name or not code_input.strip():
            with status_container:
                st.error("Please provide both team name and Python code.")
        else:
            with status_container:
                st.info("Processing your submission...")
            st.write("### Your Submitted Code:")
            st.code(code_input, language="python")

            # running unit tests
            test_results = run_unit_tests(code_input, team_name)

            with status_container:
                if "All tests passed successfully!" in test_results:
                    #st.write("Saving valid submission...")
                    valid_save_path = save_valid_submission(code_input, team_name)
                    st.success(f"{test_results}\n\nSubmission saved to {valid_save_path}.")
                else:
                    st.error(test_results)

# Leaderboard Tab
with tabs[1]:
    st.title("Leaderboard")
    st.markdown("This tab will display the leaderboard in the future.")
    st.info("Leaderboard functionality is under development. Check back later!")
