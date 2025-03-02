import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects


# extract frames
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    #check video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")



# detect objects 
def detect_objects_in_frames(input_folder, output_folder):
    """Run YOLOv8 object detection on extracted frames."""
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            frame = cv2.imread(img_path)
            results = model(frame)  # Run object detection

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    conf = box.conf[0].item()

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, frame)

    print(f"Processed frames saved in {output_folder}")


def track_objects_in_video(video_path, output_path):
    """Track objects using YOLOv8 and Norfair."""
    model = YOLO("yolov8n.pt")  # Load YOLO model
    tracker = Tracker(distance_function="euclidean", distance_threshold=30)  # Initialize Norfair Tracker

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))  # Save output video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)  # Detect objects
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                class_id = int(box.cls[0])

                # Convert bounding box to Norfair detection format
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                detections.append(Detection(points=np.array([center_x, center_y]), scores=np.array([conf])))

        # Update tracker
        tracked_objects = tracker.update(detections)

        # Draw tracking results
        draw_tracked_objects(frame, tracked_objects)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Tracking complete. Output saved to {output_path}")



#main method
def main_processor():
    #welcome note
    print("Welcome to the video processor program\n")
    #video path
    video_path=input("Enter video path: \n")
    
    #frames_output_path=input("Frames Output path: \n")
    #objects_output_path=input("Objects outpu path: \n")


    #check if videos frames output path is valid
    if len(video_path)>20:
        print("")
        # Example usage
        #extract_frames(video_path, frames_output_path)
    
        #track objects
        tracking_output = "tracked_video.mp4"    
        track_objects_in_video(video_path, tracking_output)    
    
    else:
        print("Invalid video path")   

    #out put detected objects from frames
    #detect_objects_in_frames(frames_output_path, objects_output_path)  # Step 2: Detect Objects



if __name__=="__main__":
    main_processor()




