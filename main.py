import cv2
import mediapipe as mp
import numpy as np
import time
from scoring import TARGET_POSES, get_coords, evaluate_all_poses

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculates the 2D angle at point 'b' given points 'a', 'b', and 'c'.
    Each point is a tuple or list: [x, y]
    """
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def main():
    # Attempt to open the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    
    # Check if we successfully opened the camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # To track FPS
    pTime = 0
    
    print("Starting webcam feed... ")
    print("Application is in AUTO-DETECT mode.")
    print("Press 'q' to exit.")

    # Initialize the Pose model
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        # State variables for smoothing
        smoothed_score = 0.0
        smoothing_factor = 0.25 # 0.25 smooths over ~4 frames
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Process the image and find pose landmarks
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image.flags.writeable = True
            
            # Dictionary to hold the real-time calculated angles
            current_angles = {}
            score = 0.0
            display_score = 0.0
            feedback = []
            detected_pose = "No pose detected"

            # Draw landmarks and process scores
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                try:
                    # Map MediaPipe landmarks to our generic joint names
                    # Shoulders
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    
                    # Elbows
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    
                    # Wrists
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    # Hips
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    
                    # Knees
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    
                    # Ankles
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    # Calculate angles
                    current_angles["left_elbow"] = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    current_angles["right_elbow"] = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    current_angles["left_shoulder"] = calculate_angle(l_hip, l_shoulder, l_elbow)
                    current_angles["right_shoulder"] = calculate_angle(r_hip, r_shoulder, r_elbow)
                    current_angles["left_hip"] = calculate_angle(l_shoulder, l_hip, l_knee)
                    current_angles["right_hip"] = calculate_angle(r_shoulder, r_hip, r_knee)
                    current_angles["left_knee"] = calculate_angle(l_hip, l_knee, l_ankle)
                    current_angles["right_knee"] = calculate_angle(r_hip, r_knee, r_ankle)

                    # Evaluate all target poses and get the best match
                    detected_pose, raw_score, feedback = evaluate_all_poses(current_angles)

                    # Apply Temporal Smoothing
                    if detected_pose != "No pose detected":
                        if smoothed_score == 0.0:
                            smoothed_score = raw_score # Jump to score initially
                        else:
                            smoothed_score = (smoothed_score * (1 - smoothing_factor)) + (raw_score * smoothing_factor)
                    else:
                        smoothed_score = 0.0
                        
                    display_score = round(smoothed_score, 1)

                except Exception as e:
                    pass

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime

            # --- UI Rendering ---
            
            # Setup status box
            cv2.rectangle(image, (0, 0), (640, 100), (245, 117, 16), -1)
            
            # 1. Pose Text
            cv2.putText(image, 'DETECTED POSE', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, detected_pose, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 2. Score Text (Only display if a pose is detected)
            if detected_pose != "No pose detected":
                cv2.putText(image, 'MATCH SCORE (/10)', (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                # Change color based on score (Green > 8.0, Yellow > 6.0, Red else)
                score_color = (0, 255, 0) if display_score > 8.0 else ((0, 255, 255) if display_score > 6.0 else (0, 0, 255))
                cv2.putText(image, str(display_score), (350, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3, cv2.LINE_AA)
                
                # 3. Feedback overlay
                if feedback and len(feedback) > 0:
                    y_offset = 130
                    cv2.putText(image, "Tips:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    for f in feedback:
                        y_offset += 30
                        cv2.putText(image, f"- {f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # 4. FPS overlay (Bottom left)
            h, w, _ = image.shape
            cv2.putText(image, f"FPS: {int(fps)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Display the resulting frame
            cv2.imshow('Yoga Pose Analyzer', image)
            
            # Key mappings
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
