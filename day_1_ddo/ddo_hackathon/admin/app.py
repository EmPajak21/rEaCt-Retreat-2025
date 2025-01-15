import os
import sys
import importlib
import unittest
from io import StringIO

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account


def save_valid_submission(code, team_name, bucket_name="ddo_hackathon"):
    """
    Save a valid submission to Google Cloud Storage.

    Loads the service account credentials from Streamlit secrets,
    initializes the storage client, and uploads the submitted code
    as a Python file to the specified bucket.

    Args:
        code (str): The Python code to upload.
        team_name (str): The team name to create a unique filename.
        bucket_name (str, optional): The name of the GCS bucket.
            Defaults to "ddo_hackathon".

    Returns:
        str: The name of the uploaded file.
    """
    # Load credentials from Streamlit secrets (expects a JSON-compatible dict)
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    # Initialize the Storage client with specified project and credentials
    client = storage.Client(
        project="intense-pixel-446617-e2", credentials=credentials
    )
    bucket = client.bucket(bucket_name)

    # Create the file name based on the team name
    file_name = f"{team_name}_submission.py"
    blob = bucket.blob(file_name)

    # Upload the code to the bucket with the appropriate content type
    blob.upload_from_string(code, content_type="text/x-python")
    return file_name


def run_unit_tests(code, team_name):
    """
    Run unit tests on the submitted code from test_framework.py.

    Writes the code to a temporary file, dynamically loads it as a module,
    and runs tests using a test framework located in the specified path.
    
    Args:
        code (str): The submitted Python code.
        team_name (str): The team name used for naming the temporary file.

    Returns:
        str: A message indicating test success or a summary of failures/errors.
    """
    buffer = StringIO()
    sys.stdout = buffer

    temp_filepath = f"{team_name}_temp_submission.py"

    try:
        # Write the submitted code to a temporary file.
        with open(temp_filepath, "w") as f:
            f.write(code)
        # Dynamically load the student's code as a module.
        spec = importlib.util.spec_from_file_location("student_submission", temp_filepath)
        student_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(student_module)
        # Register the module so the test framework can import it.
        sys.modules["student_submission"] = student_module

        # Define the relative path to the test framework file.
        relative_path = os.path.join("day_1_ddo", "ddo_hackathon", "admin", "test_framework.py")
        spec = importlib.util.spec_from_file_location("admin.test_framework", relative_path)
        test_framework = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_framework)

        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_framework)
        runner = unittest.TextTestRunner(stream=buffer, verbosity=2)
        result = runner.run(suite)

        errors = result.errors
        failures = result.failures

        if not errors and not failures:
            return "All tests passed successfully! 🎉"

        output_lines = []
        if failures:
            output_lines.append("\n### Failures ###")
            output_lines.extend(
                f"{test}: {failure.splitlines()[-1]}"
                for test, failure in failures
            )

        return "\n".join(filter(None, output_lines))
    except Exception as e:
        return f"An error occurred while creating or writing the temporary file: {e}"
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        sys.stdout = sys.__stdout__


def get_leaderboard_html(bucket_name="ddo_hackathon", file_name="leaderboard.html"):
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    client = storage.Client(project="intense-pixel-446617-e2", credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.download_as_text()


def main():
    """Main function to run the Streamlit app with tabs."""
    tabs = st.tabs(["Submission Portal", "Leaderboard"])

    # Submission Tab
    with tabs[0]:
        st.title("Student Submission Portal")
        status_container = st.container()

        team_name = st.text_input(
            "Enter your team name:",
            help="Enter your team's name before submitting code."
        )
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

                # Run unit tests on the submitted code.
                test_results = run_unit_tests(code_input, team_name)

                with status_container:
                    if "All tests passed successfully!" in test_results:
                        valid_save_path = save_valid_submission(code_input, team_name)
                        st.success(
                            f"{test_results}\n\nSubmission saved to {valid_save_path}."
                        )
                    else:
                        st.error(test_results)

    # Leaderboard Tab
    with tabs[1]:
        st.title("Leaderboard")
        try:
            leaderboard_html = get_leaderboard_html()
            st.components.v1.html(leaderboard_html, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Could not load the leaderboard: {e}")
    
if __name__ == "__main__":
    main()