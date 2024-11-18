import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import sys
import cv2
import csv
from ultralytics import YOLOWorld
import os
import tkinter as tk
from tkinter import filedialog

print("Environment Ready")

# Function to open a file dialog and select a .bag file
def select_bag_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    bag_file_path = filedialog.askopenfilename(title="Select a .bag file",
                                                filetypes=[("Bag files", "*.bag")])
    return bag_file_path

# Check if a bag file path is provided as an argument, otherwise open a file dialog
if len(sys.argv) < 2:
    bag_file_path = select_bag_file()
else:
    bag_file_path = sys.argv[1]

if not bag_file_path:
    print("No .bag file selected. Exiting.")
    sys.exit(1)

output_video_path = f"{os.path.splitext(bag_file_path)[0]}.mp4"

# Create a pipeline for RealSense
pipeline = rs.pipeline()
config = rs.config()
try:
    rs.config.enable_device_from_file(config, bag_file_path)
    config.enable_all_streams()

    # Start the pipeline
    pipeline.start(config)
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    print(f"Depth Scale: {depth_scale}")
    print(intrinsics)

    # Load YOLO model
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(["child", "children", "robot", "robotNAO", "NAO"])

    # List to store detected objects and their distances
    detected_objects = []
    frame_count = 0

    # Create VideoWriter to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # You can set the desired FPS
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))  # Adjust resolution as needed

    # Process the frames from the .bag file
    while True:
        frame_count += 1
        # Wait for a pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align depth and color frames
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not color_frame or not depth_frame:
            print("No depth or color frame detected.")
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        results = model.predict(color_image, conf=0.10, verbose=False)

        # Process detection results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                box_coords = box.xyxy[0]
                class_id = int(box.cls)  # Get the class ID
                class_name = model.names[class_id]  # Get the class name

                # Draw bounding box
                cv2.rectangle(color_image, (int(box_coords[0]), int(box_coords[1])),
                              (int(box_coords[2]), int(box_coords[3])), (0, 255, 0), 2)

                # Calculate the center of the bounding box
                u = int((box_coords[0] + box_coords[2]) / 2)
                v = int((box_coords[1] + box_coords[3]) / 2)

                # Get depth value at the center of the bounding box
                depth_value = depth_frame.get_distance(u, v)
                if depth_value > 0:
                    # Convert the 2D point (u, v) to 3D space
                    X = (u - intrinsics.ppx) * depth_value / intrinsics.fx
                    Y = (v - intrinsics.ppy) * depth_value / intrinsics.fy
                    Z = depth_value

                    # Store the object and its distance
                    detected_objects.append({
                        'Frame_id': frame_count,
                        'Object': class_name,
                        'Point_2d': (u, v),
                        'Point_3d': (X, Y, Z)
                    })

                    print(f"Point in 3D space for bbox at ({u}, {v}): ({X}, {Y}, {Z})")

                    # Draw label below the bounding box
                    label = f"{class_name}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    label_position = (int(box_coords[0]), int(box_coords[3]) + label_size[1] + 5)  # Adjust label position
                    cv2.rectangle(color_image, (int(box_coords[0]), int(box_coords[3]) + 5),
                                  (int(box_coords[0]) + label_size[0], int(box_coords[3]) + label_size[1] + 10),
                                  (0, 255, 0), -1)  # Filled rectangle for label background
                    cv2.putText(color_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        objects_of_the_frame = [obj['Object'] for obj in detected_objects if obj['Frame_id'] == frame_count]
        point_of_the_frame = [obj['Point_3d'] for obj in detected_objects if obj['Frame_id'] == frame_count]
        point_2d_of_the_frame = [obj['Point_2d'] for obj in detected_objects if obj['Frame_id'] == frame_count]
        for i, point in enumerate(point_of_the_frame):
            for j in range(i + 1, len(point_of_the_frame)):
                distance = np.linalg.norm(np.array(point) - np.array(point_of_the_frame[j]))
                print(f"Distance between {objects_of_the_frame[i]} and {objects_of_the_frame[j]}: {distance:.2f} meters")
                cv2.line(color_image, point_2d_of_the_frame[i], point_2d_of_the_frame[j], (0, 0, 255), 2)
                cv2.putText(color_image, f"{distance:.2f}m", (int((point_2d_of_the_frame[i][0] + point_2d_of_the_frame[j][0]) / 2),
                                                              int((point_2d_of_the_frame[i][1] + point_2d_of_the_frame[j][1]) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the current frame to the video file
        video_writer.write(color_image)

        # Show images
        cv2.imshow("color_image", color_image)
        cv2.imshow("depth_image", depth_image)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

        Save detected objects to a CSV file using csv library
        if detected_objects:
            with open('detected_objects_distances.csv', mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['Frame_id','Object', 'Point_3d'])
                writer.writeheader()
                writer.writerows(detected_objects)
            print("Detected objects and distances saved to 'detected_objects_distances.csv'.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    pipeline.stop()
