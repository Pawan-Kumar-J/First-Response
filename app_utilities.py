import cv2
import numpy as np
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from deep_sort.sort.tracker import Tracker
import torch
import time
from ultralytics import YOLO
import streamlit as st
import argparse

unique_track_ids = set()
normal_boxes = set()
wounded_boxes = set()

def process_video_frames(video_file, weights_path, target_fps):
    model = YOLO(weights_path)
    deep_sort_weights = r'deep_sort/deep/checkpoint/ckpt.t7'
    tracker = DeepSort(model_path=deep_sort_weights, max_age=120)

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    counter, elapsed = 0, 0
    start_time = time.perf_counter()

    if fps > target_fps:
        frame_interval = int(round(fps / target_fps))
    else:
        frame_interval = 1

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        if frame_index % frame_interval != 0:
            continue

        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
                    color = (0, 255, 0) if cls[idx] == 0 else (255, 0, 0)
                    cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

                    track_id = f"{cls[idx]}_{idx}"  # Assuming cls[idx] is 0 for normal and 1 for wounded
                    if cls[idx] == 0 and track_id not in normal_boxes:
                        normal_boxes.add(track_id)
                    elif cls[idx] == 1 and track_id not in wounded_boxes:
                        wounded_boxes.add(track_id)

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

            unique_track_ids.add(track_id)

        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1

        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        # Display every frame in Streamlit
        st.image(og_frame)

    cap.release()

def llm_analysis(total = len(unique_track_ids), normal = len(normal_boxes), wounded = len(wounded_boxes)):
    st.markdown(f'''**Summary:**
In the disaster-affected area, a total of {total} individuals have been identified for rescue. Among them, {wounded} are wounded and require medical attention. The remaining {normal} are in good condition.

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


# def process_video_frames(video_file):
#     weights_path = r"best (1).pt"
#     model = YOLO("yolov8n")
#     deep_sort_weights = r'deep_sort/deep/checkpoint/ckpt.t7'
#     tracker = DeepSort(model_path=deep_sort_weights, max_age=120)

#     cap = cv2.VideoCapture(video_file)  # Replace with your video file path
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     counter, elapsed = 0, 0
#     start_time = time.perf_counter()

    
#     # Assuming TARGET_FPS is defined somewhere in your code
#     TARGET_FPS = 5
#     if fps > TARGET_FPS:
#         frame_interval = int(round(fps / TARGET_FPS))  # Select every 15th frame
#     frame_index = 0

#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_index += 1

#         # Process every frame_interval-th frame
#         if frame_index % frame_interval != 0:
#             continue

#         og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Object detection using YOLO (Replace with your actual detection logic)
#         results = model(og_frame, classes=[0, 1], conf=0.5, verbose=False)
#         class_names = ['person_normal', 'person_wounded']

#         if results:
#             for result in results:
#                 boxes = result.boxes
#                 cls = boxes.cls.tolist()
#                 xyxy = boxes.xyxy
#                 conf = boxes.conf
#                 xywh = boxes.xywh

#                 for idx, (x1, y1, x2, y2) in enumerate(xyxy):
#                     color = (0, 255, 0) if cls[idx] == 0 else (0, 0, 255)
#                     cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

#                     # Update counters based on class labels
#                     # track_id = unique_track_ids[idx]
#                     # if cls[idx] == 0:  # Normal person
#                     #     normal_person_count_dict[track_id] = normal_person_count_dict.get(track_id, 0) + 1
#                     # elif cls[idx] == 1:  # Wounded person
#                     #     wounded_person_count_dict[track_id] = wounded_person_count_dict.get(track_id, 0) + 1

#             pred_cls = np.array(cls)
#             conf = conf.detach().cpu().numpy()
#             xyxy = xyxy.detach().cpu().numpy()
#             bboxes_xywh = xywh
#             bboxes_xywh = xywh.cpu().numpy()
#             bboxes_xywh = np.array(bboxes_xywh, dtype=float)

#             tracks = tracker.update(bboxes_xywh, conf, og_frame)

#             for track in tracker.tracker.tracks:
#                 track_id = track.track_id
#                 x1, y1, x2, y2 = track.to_tlbr()
#                 w = x2 - x1
#                 h = y2 - y1

#                 # color = (0, 0, 255)
#                 # cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 5)

#                 unique_track_ids.add(track_id)

#             # Update FPS and place on frame
#             current_time = time.perf_counter()
#             elapsed = (current_time - start_time)
#             counter += 1

#             if elapsed > 1:
#                 fps = counter / elapsed
#                 counter = 0
#                 start_time = current_time
            
#             st.image(og_frame)
#             # image_df.append({"frames": encode_image(og_frame)}, ignore_index=True)

#     cap.release()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames with YOLO and DeepSort.")
    parser.add_argument("video_file", help="Path to the video file.")
    parser.add_argument("--weights_path", default="best (1).pt", help="Path to YOLO weights file.")
    parser.add_argument("--target_fps", type=float, default=5, help="Target frames per second.")
    args = parser.parse_args()

    # Call the function with command-line arguments
    process_video_frames(args.video_file, args.weights_path, args.target_fps)