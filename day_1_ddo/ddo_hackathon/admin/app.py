import os
import sys
import importlib
import unittest
from io import StringIO

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account


def save_valid_submission(code, team_name, track, bucket_name="ddo_hackathon"):
    """
    Save a valid submission to Google Cloud Storage.

    Loads the service account credentials from Streamlit secrets,
    initializes the storage client, and uploads the submitted code
    as a Python file to the specified bucket under a folder corresponding
    to the selected track.

    Args:
        code (str): The Python code to upload.
        team_name (str): The team name to create a unique filename.
        track (str): The selected track ("Track 1" or "Track 2").
        bucket_name (str, optional): The name of the GCS bucket.
            Defaults to "ddo_hackathon".

    Returns:
        str: The name (including folder path) of the uploaded file.
    """
    # Load credentials from Streamlit secrets (expects a JSON-compatible dict)
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    # Initialize the Storage client with specified project and credentials
    client = storage.Client(project="intense-pixel-446617-e2", credentials=credentials)
    bucket = client.bucket(bucket_name)

    # Determine folder path based on track selection.
    if track == "Track 1":
        folder_path = "day1/t1"
    else:
        folder_path = "day1/t2"

    # Create the file name based on the team name and track folder.
    file_name = f"{folder_path}/{team_name}_submission.py"
    blob = bucket.blob(file_name)

    # Upload the code to the bucket with the appropriate content type.
    blob.upload_from_string(code, content_type="text/x-python")
    return file_name


def run_unit_tests(code, team_name, track):
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
        spec = importlib.util.spec_from_file_location(
            "student_submission", temp_filepath
        )
        student_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(student_module)
        # Register the module so the test framework can import it.
        sys.modules["student_submission"] = student_module

        # Define the relative path to the test framework file.
        relative_path = os.path.join(
            "day_1_ddo", "ddo_hackathon", "admin", "test_framework.py"
        )
        spec = importlib.util.spec_from_file_location(
            "admin.test_framework", relative_path
        )
        test_framework = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_framework)
        
        # Add the following line:
        test_framework.TRACK = track

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
                f"{test}: {failure.splitlines()[-1]}" for test, failure in failures
            )

        return "\n".join(filter(None, output_lines))
    except Exception as e:
        return f"An error occurred while creating or writing the temporary file: {e}"
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        sys.stdout = sys.__stdout__


def get_leaderboard_html(bucket_name="ddo_hackathon", file_name="leaderboard.html"):
    """
    Downloads the leaderboard HTML file from a Google Cloud Storage bucket.

    Args:
        bucket_name: str, optional. Google Cloud Storage bucket.
        file_name: str, optional. Name of the file to download from the bucket.

    Returns:
        str: The content of the HTML file as a string.
    """
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    client = storage.Client(project="intense-pixel-446617-e2", credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.download_as_text()


def main():
    """Main function to run the Streamlit app with tabs."""
    # Create three tabs: one for submission and two for leaderboards.
    tabs = st.tabs(["Submission Portal", "Leaderboard Track 1", "Leaderboard Track 2"])

    # Submission Tab
    with tabs[0]:
        st.title("Student Submission Portal")
        status_container = st.container()

        team_name = st.text_input(
            "Enter your team name:",
            help="Enter your team's name before submitting code.",
        )
        # Add a radio button for track selection.
        track = st.radio(
            "Select Your Track:",
            options=["Track 1", "Track 2"],
            help="Choose the track you want to participate in.",
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
                test_results = run_unit_tests(code_input, team_name, track)

                with status_container:
                    if "All tests passed successfully!" in test_results:
                        valid_save_path = save_valid_submission(
                            code_input, team_name, track
                        )
                        st.success(
                            f"{test_results}\n\nSubmission saved to {valid_save_path}."
                        )
                    else:
                        st.error(test_results)

    # Leaderboard Tab for Track 1
    with tabs[1]:
        st.title("Leaderboard - Track 1")
        if st.button("Refresh Leaderboard T1"):
            pass
        try:
            # Assuming your Track 1 leaderboard HTML file is named "leaderboard_t1.html"
            leaderboard_html = get_leaderboard_html(
                file_name="day1/leaderboard_t1.html"
            )
            st.components.v1.html(leaderboard_html, height=600, scrolling=True)
        except Exception as e:
            st.error("No leaderboard generated yet!")

    # Leaderboard Tab for Track 2
    with tabs[2]:
        st.title("Leaderboard - Track 2")
        if st.button("Refresh Leaderboard T2"):
            pass
        try:
            # Assuming your Track 2 leaderboard HTML file is named "leaderboard_t2.html"
            leaderboard_html = get_leaderboard_html(
                file_name="day1/leaderboard_t2.html"
            )
            st.components.v1.html(leaderboard_html, height=600, scrolling=True)
        except Exception as e:
            st.error("No leaderboard generated yet!")


if __name__ == "__main__":
    main()
