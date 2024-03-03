import cv2
import numpy as np
#from utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from deep_sort.sort.tracker import Tracker
# from sort.tracker import Tracker


def get_every_10th_frame(video_path):
    video = cv2.VideoCapture(video_path)

    # Find OpenCV version
    # (major_ver, minor_ver, subminor_ver) = (cv2._version_).split('.')

    # if int(major_ver) < 3:
    #     fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    #     # print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    # else:
    fps = video.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    evey_nth_frame = fps/10
    c=0
    frame_list = []
    while video.isOpened():  # Check if video is still open
        ret, frame = video.read()
        if not ret:
            break  # Exit loop if frame cannot be read

        if c % evey_nth_frame == 0:
            frame_list.append(frame)
        c += 1

    # Release the video capture object (important!)
    video.release()

    return frame_list


def get_inferences(model, frame,frame_idx):
    og_frame = frame.copy()
    results = model(frame, classes = 0, conf = 0.7)
    results = model(og_frame, classes=0, conf=0.7)

    class_names = ['person']

    if results:
        print(frame_index)
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
                #print("Class:", class_name)

        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)

        tracks = tracker.update(bboxes_xywh, conf, og_frame)

        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # Set color values for red, blue, and green
            red_color = (0, 0, 255)  # (B, G, R)
            blue_color = (255, 0, 0)  # (B, G, R)
            green_color = (0, 255, 0)  # (B, G, R)

            # # Determine color based on track_id
            # color_id = track_id % 3
            # if color_id == 0:
            #     color = red_color
            # elif color_id == 1:
            #     color = blue_color
            # else:
            #     color = green_color

            color = red_color

            # cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255,255,255), 5)


            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            # Add the track_id to the set of unique track IDs
            unique_track_ids.add(track_id)

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

        # Write the frame to the output video file
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))