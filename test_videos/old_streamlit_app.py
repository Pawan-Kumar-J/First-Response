import streamlit as st
from app_utilities import *
import torch
import time
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression
video_to_stream = r"disaster2.mp4"
# weights_path = r"yolov8x.pt"
weights_path = r"best (1).pt"
# model = torch.hub.load("D:\\PycharmProjects\\AU_Hackathon'24\\yolov7", 'custom', weights_path, source ="local") # load a pretrained model (recommended for training)
model = YOLO(weights_path)
deep_sort_weights = r'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=120)

st.title("Disaster Footage Monitoring Dashboard")

# Create the main container
container = st.container()

# Create columns with separator divs in between
col1, div1, col2, div2, col3 = st.columns([2, 0.2, 1, 0.2, 1])
TARGET_FPS = 5
# Fill columns with content
with col1:
    st.video(video_to_stream, start_time=0)
with col2:
    cap = cv2.VideoCapture(video_to_stream)  # Replace with your video file path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    counter, elapsed = 0, 0
    start_time = time.perf_counter()
    unique_track_ids = set()
    if fps > TARGET_FPS:
        frame_interval = int(round(fps/TARGET_FPS))  # Select every 15th frame
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1
        # print(frame_interval)
        # print(frame_index % frame_interval)

        # Process every 4th frame
        if frame_index % frame_interval != 0:
            # print(frame_index % frame_interval)
            continue

        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Object detection using YOLO
        results = model(og_frame, classes=[0,1], conf=0.5, verbose = False)
        # results.non_max_suppression
        class_names = ['person_normal', "person_wounded"]
        
        if results:
            # results = non_max_suppression(results)
            # results.
            # print(frame_index)
            # Process detection results and update the tracker
            # ...
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                probs = result.probs  # Class probabilities for classification outputs
                cls = boxes.cls.tolist()  # Convert tensor to list
                xyxy = boxes.xyxy
                conf = boxes.conf
                xywh = boxes.xywh  # box with xywh format, (N, 4)
                for class_index in cls:
                    class_name = class_names[int(class_index)]
                red_color = (0, 0, 255)  # (B, G, R)
                blue_color = (255, 0, 0)  # (B, G, R)
                green_color = (0, 255, 0)  # (B, G, R)
                
                for idx,(x1, y1, x2, y2) in enumerate(xyxy):
                    color = green_color if cls[idx] == 0 else blue_color
                    cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
                
                    #print("Class:", class_name)

            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            # xyxy = non_max_suppression(xyxy)
            bboxes_xywh = xywh
            bboxes_xywh = xywh.cpu().numpy()
            bboxes_xywh = np.array(bboxes_xywh, dtype=float)
            
            tracks = tracker.update(bboxes_xywh, conf, og_frame)
            cnt = 0

            for track in tracker.tracker.tracks:
                track_id = track.track_id
                hits = track.hits
                x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
                w = x2 - x1  # Calculate width
                h = y2 - y1  # Calculate height

                # Set color values for red, blue, and green
                
                # Determine color based on track_id
                # color = green_color if pred_cls[cnt] == 0 else red_color
                # color_id = track_id % 3
                # if color_id == 0:
                #     color = red_color
                # elif color_id == 1:
                #     color = blue_color
                # else:
                #     color = green_color

                color = red_color

                # cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
                # cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 5)


                text_color = (0, 0, 0)  # Black color for text
                # cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                # Add the track_id to the set of unique track IDs
                unique_track_ids.add(track_id)
                cnt += 1

            # Update the person count based on the number of unique track IDs
            person_count = len(unique_track_ids)

            # Update FPS and place on frame
            current_time = time.perf_counter()
            elapsed = (current_time - start_time)
            counter += 1

            if elapsed > 1:
                fps = counter / elapsed
                counter = 0
                start_time = current_time

            # Draw person count on frame
            cv2.putText(og_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            st.image(og_frame)

st.write('<div class="separator"></div>', unsafe_allow_html=True)

st.markdown('''**Summary:**
In the disaster-affected area, a total of 142 individuals have been identified for rescue. Among them, 17 are wounded and require medical attention.

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

# Add empty divs with separator styles

# st.write('<div class="separator"></div>', unsafe_allow_zhtml=True)
