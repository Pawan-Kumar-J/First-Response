import streamlit as st

st.set_page_config(layout="wide")

# Set up the layout with three columns
col1, col2, col3 = st.columns(3)

# Divide column 1 into two rows
with col1:
    # Add a file uploader for video files
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    # Check if a video file is uploaded
    if video_file is not None:
        # Get the file path
        video_path = video_file.name
        st.write("Video uploaded successfully!")
        st.write("Video Path:", video_path)
    else:
        st.write("No video file uploaded yet.")

with col1:
    st.header("Column 1 - Row 2 (Bottom)")
    st.write("This is the second row of column 1.")
    # Add any content you want for the bottom row

# Column 2
with col2:
    st.header("Column 2")
    st.write("This is the second column.")

# Column 3
with col3:
    st.header("Column 3")
    st.write("This is the third column.")
