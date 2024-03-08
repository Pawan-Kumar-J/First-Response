import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.row import row
from app_utilities import *
import torch
import time
from ultralytics import YOLO
# from ultralytics.utils.ops import non_max_suppression

wounded_person_count_dict = {}
normal_person_count_dict = {}
unique_track_ids = set()


def process_video_frames(video_file):
    weights_path = r"best (1).pt"
    model = YOLO(weights_path)
    deep_sort_weights = r'deep_sort/deep/checkpoint/ckpt.t7'
    tracker = DeepSort(model_path=deep_sort_weights, max_age=120)

    cap = cv2.VideoCapture(video_file)  # Replace with your video file path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    counter, elapsed = 0, 0
    start_time = time.perf_counter()

    
    # Assuming TARGET_FPS is defined somewhere in your code
    TARGET_FPS = 5
    if fps > TARGET_FPS:
        frame_interval = int(round(fps / TARGET_FPS))  # Select every 15th frame
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        # Process every frame_interval-th frame
        if frame_index % frame_interval != 0:
            continue

        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Object detection using YOLO (Replace with your actual detection logic)
        results = model(og_frame, classes=[0, 1], conf=0.5, verbose=False)
        class_names = ['person_normal', 'person_wounded']

        if results:
            for result in results:
                boxes = result.boxes
                cls = boxes.cls.tolist()
                xyxy = boxes.xyxy
                conf = boxes.conf
                xywh = boxes.xywh

                for idx, (x1, y1, x2, y2) in enumerate(xyxy):
                    color = (0, 255, 0) if cls[idx] == 0 else (0, 0, 255)
                    cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

                    # Update counters based on class labels
                    # track_id = unique_track_ids[idx]
                    # if cls[idx] == 0:  # Normal person
                    #     normal_person_count_dict[track_id] = normal_person_count_dict.get(track_id, 0) + 1
                    # elif cls[idx] == 1:  # Wounded person
                    #     wounded_person_count_dict[track_id] = wounded_person_count_dict.get(track_id, 0) + 1

            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xywh = xywh
            bboxes_xywh = xywh.cpu().numpy()
            bboxes_xywh = np.array(bboxes_xywh, dtype=float)

            tracks = tracker.update(bboxes_xywh, conf, og_frame)

            for track in tracker.tracker.tracks:
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_tlbr()
                w = x2 - x1
                h = y2 - y1

                color = (0, 0, 255)
                cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 5)

                unique_track_ids.add(track_id)

            # Update FPS and place on frame
            current_time = time.perf_counter()
            elapsed = (current_time - start_time)
            counter += 1

            if elapsed > 1:
                fps = counter / elapsed
                counter = 0
                start_time = current_time
            
            st.image(og_frame)
            # image_df.append({"frames": encode_image(og_frame)}, ignore_index=True)

    cap.release()

st.set_page_config(layout="wide")

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
            print(video_path)
            st.write("Video uploaded successfully!")
            st.video(video_file, start_time=0)
        else:
            st.write("No video file uploaded yet.")
    
    elif video_options == "Use Camera":
        st.warning("This feature is still in development")
        st.write("Live video feed from camera will be displayed here.")


# Column 2
with col2:
    row1 = row(1, vertical_align = "center")
    row1.metric(label="Total People", value="174")

    row2 = row(2, vertical_align = "center")
    row2.metric(label="Total Normal", value="152", delta = 5)
    row2.metric(label="Total Wounded", value="22", delta=-5)
    style_metric_cards(background_color = "0d1117", border_color = "252730", border_left_color = "red")
    # st.divider()

with col2:
    if video_file is not None:
        with st.container(height = 480):
            process_video_frames(video_path)
    else:
        st.write("No video file uploaded yet.")

# Column 3
with col3:
    st.header("Analysis")
    if video_file is not None:
        st.markdown(f'''**Summary:**
In the disaster-affected area, a total of {142} individuals have been identified for rescue. Among them, 17 are wounded and require medical attention.

**Action Steps:**

**Priority Rescue:**
- Rescue teams are advised to proceed with caution and ensure the safety of both the victims and responders during the extraction process.

**Wounded Individuals:**
- Following the rescue of urgent cases, attention should be directed towards the 17 wounded individuals.
- Medical personnel should assess the severity of injuries and administer appropriate treatment on-site or during transport to medical facilities.

**Resource Allocation:**
- Adequate medical supplies and equipment must be brought to the rescue site to address the needs of the wounded individuals.
- The following items are recommended for inclusion in the rescue operation:
  - First aid kits (quantity: 3+)
  - Bandages and dressings (quantity: 30)
  - Sterile gauze pads (quantity: 23)
  - Antiseptic solution (quantity: 3 bottles)
  - Pain relief medication (quantity: 25 doses)
  - Splints and immobilization devices (quantity: 5)
  - Stretchers or medical transport devices (quantity: 3)

**Additional Recommendations:**
- Coordinate with local medical facilities to ensure prompt transfer and treatment of rescued individuals requiring further medical attention.
- Maintain communication with the command center for updates on the situation and additional support as needed.

**Conclusion:**
By prioritizing the rescue of urgent cases and providing essential medical care to the wounded, the rescue operation aims to minimize casualties and ensure the well-being of all individuals affected by the disaster.''')

    else:
        st.write("No video file uploaded yet.")
    
# row1 = row(1)
# row1.write("This is ahorizontally scrollabale element displaying all frames of the video with bounding box around the people.")
