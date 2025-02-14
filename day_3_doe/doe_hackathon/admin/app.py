import os
import sys
import json

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account


def save_valid_submission(
    mask_values, param_values, team_name, track, bucket_name="ddo_hackathon"
):
    """
    Save a valid submission to Google Cloud Storage in JSON format.

    Loads the service account credentials from Streamlit secrets,
    initializes the storage client, and uploads the submitted data
    as a JSON file to the specified bucket under a folder corresponding
    to the selected track.

    Args:
        mask_values (list[int]): List of 9 binary integers.
        param_values (list[float]): List of 5 float values.
        team_name (str): The team name to create a unique filename.
        track (str): The selected track ("Track 1" or "Track 2").
        bucket_name (str, optional): The name of the GCS bucket.
            Defaults to "ddo_hackathon".

    Returns:
        str: The name (including folder path) of the uploaded file.
    """
    # Prepare the submission dictionary
    submission_dict = {"mask": mask_values, "params": param_values}
    submission_json = json.dumps(submission_dict, indent=2)

    # Load credentials from Streamlit secrets (expects a JSON-compatible dict)
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    # Initialize the Storage client
    client = storage.Client(project="intense-pixel-446617-e2", credentials=credentials)
    bucket = client.bucket(bucket_name)

    # Determine folder path based on track selection
    if track == "Track 1":
        folder_path = "day3/t1"
    else:
        folder_path = "day3/t2"

    # Create the file name based on the team name and track folder
    file_name = f"{folder_path}/{team_name}_submission.json"
    blob = bucket.blob(file_name)

    # Upload the JSON submission
    blob.upload_from_string(submission_json, content_type="application/json")
    return file_name


def get_leaderboard_html(bucket_name="ddo_hackathon", file_name="leaderboard.html"):
    """
    Downloads the leaderboard HTML file from a Google Cloud Storage bucket.

    Args:
        bucket_name (str, optional): Name of the GCS bucket.
        file_name (str, optional): Name of the file to download from the bucket.

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
    # Create three tabs: one for submission and two for leaderboards
    tabs = st.tabs(["Submission Portal", "Leaderboard Track 1", "Leaderboard Track 2"])

    # Submission Tab
    with tabs[0]:
        st.title("Day 3: Design of Experiments Submission Portal")
        status_container = st.container()

        # Team name and track selection
        team_name = st.text_input(
            "Enter your team name:", help="Enter your team's name before submitting."
        )
        track = st.radio(
            "Select Your Track:",
            options=["Track 1", "Track 2"],
            help="Choose the track you want to participate in.",
        )

        st.markdown("### Provide Your 'mask' and 'params' Below")

        # Collect mask (should be exactly 9 binary values)
        mask_str = st.text_input(
            "Enter 9 binary values for 'mask' (comma-separated)",
            value="0,0,1,0,0,0,0,0,0",
            help="Example: 0,0,1,0,0,0,0,0,0",
        )

        # Collect params (should be exactly 5 floats)
        params_str = st.text_input(
            "Enter 5 float values for 'params' (comma-separated)",
            value="0.1,0.2,0.3,0.4,0.5",
            help="Example: 0.1,0.2,0.3,0.4,0.5",
        )

        submit_button = st.button("Submit")

        if submit_button:
            # Validate input
            if not team_name.strip():
                st.error("Please provide a valid team name.")
                st.stop()

            # Parse mask
            try:
                mask_values = [int(x.strip()) for x in mask_str.split(",")]
            except ValueError:
                st.error("Mask must contain only integers (0 or 1).")
                st.stop()

            # Check length and binary
            if len(mask_values) != 9:
                st.error("Mask must contain exactly 9 values.")
                st.stop()
            if any(m not in [0, 1] for m in mask_values):
                st.error("Mask values must be 0 or 1 only.")
                st.stop()

            # Parse params
            try:
                param_values = [float(x.strip()) for x in params_str.split(",")]
            except ValueError:
                st.error("Params must be valid float numbers.")
                st.stop()

            # Check length
            if len(param_values) != 5:
                st.error("Params must contain exactly 5 values.")
                st.stop()

            # If all validations pass, upload as JSON
            st.info("Submitting your valid entry...")

            file_path = save_valid_submission(
                mask_values, param_values, team_name, track
            )
            st.success(f"Submission saved to {file_path}.")

    # Leaderboard Tab for Track 1
    with tabs[1]:
        st.title("Leaderboard - Track 1")
        try:
            # Assuming your Track 1 leaderboard HTML file is named "leaderboard_t1.html"
            leaderboard_html = get_leaderboard_html(
                file_name="day3/t1/leaderboard_t1.html"
            )
            st.components.v1.html(leaderboard_html, height=600, scrolling=True)
        except Exception as e:
            st.error(f"No leaderboard generated yet!")

    # Leaderboard Tab for Track 2
    with tabs[2]:
        st.title("Leaderboard - Track 2")
        try:
            # Assuming your Track 2 leaderboard HTML file is named "leaderboard_t2.html"
            leaderboard_html = get_leaderboard_html(
                file_name="day3/t2/leaderboard_t2.html"
            )
            st.components.v1.html(leaderboard_html, height=600, scrolling=True)
        except Exception as e:
            st.error(f"No leaderboard generated yet!")


if __name__ == "__main__":
    main()
