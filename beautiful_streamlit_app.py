import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.row import row
from app_utilities import *

weights_path = "./best (1).pt"
target_fps = 5

# Set page configuration and layout
st.set_page_config(layout="wide", page_title="First Response - Disaster Management")

# Set up the layout with three columns
col1, col2, col3 = st.columns(3)

# Divide column 1 into two rows
with col1:
    st.header("Video")
    video_options = st.radio("Video Upload or Camera Capture", ["Upload Video", "Use Camera"])

with col1:
    if video_options == "Upload Video":
        # Add a file uploader for video files
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        TARGET_FPS = 5
        # Check if a video file is uploaded
        if video_file is not None:
            # Get the file path
            video_path = str("./" + video_file.name)
            st.write("Video uploaded successfully!")
            st.video(video_file, start_time=0)
        else:
            st.write("No video file uploaded yet.")
    
    elif video_options == "Use Camera":
        st.warning("This feature is still in development")
        st.write("Live video feed from camera will be displayed here.")

# Column 2
with col2:
    row1 = row(1, vertical_align="center")
    row1.metric(label="Total People", value=len(unique_track_ids))

    row2 = row(2, vertical_align="center")
    row2.metric(label="Total Normal", value=len(normal_boxes), delta=5)
    row2.metric(label="Total Wounded", value=len(wounded_boxes), delta=-5)
    style_metric_cards(background_color="0d1117", border_color="252730", border_left_color="red")

    if video_file is not None:
        st.subheader("Frames with Bounding Box")
        with st.container(height=480):
            process_video_frames(video_path, weights_path, target_fps)
    else:
        st.write("No video file uploaded yet.")

# Column 3
with col3:
    st.header("Analysis")
    if video_file is not None:
        llm_analysis()
    else:
        st.write("No video file uploaded yet.")
