import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyttsx3
from scoring import TARGET_POSES, get_coords, evaluate_all_poses

# ---- Audio Announcement Setup ----
def announce_score(score):
    """
    Runs the TTS engine in a separate thread.
    This prevents the audio processing from freezing the webcam feed.
    """
    try:
        engine = pyttsx3.init()
        # Optional: adjust speech rate
        engine.setProperty('rate', 150) 
        text_to_say = f"Score locked. {score} out of 10."
        engine.say(text_to_say)
        engine.runAndWait()
    except Exception as e:
        print(f"Audio Error: {e}")

# ---- Hardware Setup (Raspberry Pi Only) ----
try:
    from rpi_lcd import LCD
    lcd = LCD()
    lcd_available = True
except ImportError:
    lcd = None
    lcd_available = False
    print("Warning: rpi_lcd not installed or not running on Raspberry Pi. LCD output disabled.")

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# ---- Competition Mode Constants ----
HOLD_DURATION = 10.0       # Seconds the competitor must hold the pose
DETECTION_THRESHOLD = 6.0  # Minimum score to start the hold timer

# ---- Application States ----
STATE_WAITING = "WAITING"   # No valid pose detected, waiting for competitor
STATE_HOLDING = "HOLDING"   # Pose detected, countdown is active
STATE_LOCKED  = "LOCKED"    # Score is frozen and displayed

def calculate_angle(a, b, c):
    """
    Calculates the 2D angle at point 'b' given points 'a', 'b', and 'c'.
    Each point is a tuple or list: [x, y]
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def apply_grading_band(score):
    """
    Converts a raw decimal score into a strict integer grading band.
    This makes the final score predictable and easy for judges to read.
    """
    if score >= 9.5:
        return 10
    elif score >= 8.5:
        return 9
    elif score >= 7.5:
        return 8
    elif score >= 6.5:
        return 7
    elif score >= 5.5:
        return 6
    elif score >= 4.5:
        return 5
    elif score >= 3.5:
        return 4
    elif score >= 2.5:
        return 3
    elif score >= 1.5:
        return 2
    elif score >= 0.5:
        return 1
    else:
        return 0

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    pTime = 0
    
    print("=" * 50)
    print("  YOGA POSE COMPETITION JUDGE")
    print("=" * 50)
    print(f"  Hold Duration : {int(HOLD_DURATION)} seconds")
    print(f"  Min Threshold : {DETECTION_THRESHOLD}/10")
    print("  Press 'R' to reset for next competitor")
    print("  Press 'Q' to quit")
    print("=" * 50)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        # ---- State Machine Variables ----
        state = STATE_WAITING
        hold_start_time = 0.0
        score_buffer = []          # Stores all raw scores during the hold
        locked_score = 0           # The final graded integer score
        locked_pose_name = ""      # The pose name when locked
        detected_pose_during_hold = ""
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # ---- Pose Detection ----
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image.flags.writeable = True
            
            current_angles = {}
            raw_score = 0.0
            feedback = []
            detected_pose = "No pose detected"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                try:
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    current_angles["left_elbow"] = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    current_angles["right_elbow"] = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    current_angles["left_shoulder"] = calculate_angle(l_hip, l_shoulder, l_elbow)
                    current_angles["right_shoulder"] = calculate_angle(r_hip, r_shoulder, r_elbow)
                    current_angles["left_hip"] = calculate_angle(l_shoulder, l_hip, l_knee)
                    current_angles["right_hip"] = calculate_angle(r_shoulder, r_hip, r_knee)
                    current_angles["left_knee"] = calculate_angle(l_hip, l_knee, l_ankle)
                    current_angles["right_knee"] = calculate_angle(r_hip, r_knee, r_ankle)

                    detected_pose, raw_score, feedback = evaluate_all_poses(current_angles)

                except Exception:
                    pass

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # ---- State Machine Logic ----
            now = time.time()

            if state == STATE_WAITING:
                if detected_pose != "No pose detected" and raw_score >= DETECTION_THRESHOLD:
                    # A valid pose detected! Start the hold timer.
                    state = STATE_HOLDING
                    hold_start_time = now
                    score_buffer = [raw_score]
                    detected_pose_during_hold = detected_pose

            elif state == STATE_HOLDING:
                elapsed = now - hold_start_time
                
                if detected_pose == "No pose detected" or raw_score < DETECTION_THRESHOLD:
                    # Competitor broke the pose! Reset timer.
                    state = STATE_WAITING
                    score_buffer = []
                    hold_start_time = 0.0
                else:
                    # Still holding, accumulate score
                    score_buffer.append(raw_score)
                    detected_pose_during_hold = detected_pose
                    
                    if elapsed >= HOLD_DURATION:
                        # Time's up! Calculate the final aggregated score.
                        avg_raw = sum(score_buffer) / len(score_buffer)
                        locked_score = apply_grading_band(avg_raw)
                        locked_pose_name = detected_pose_during_hold
                        state = STATE_LOCKED
                        print(f"\n>>> SCORE LOCKED: {locked_pose_name} = {locked_score}/10 (from {len(score_buffer)} frames) <<<\n")
                        
                        # --- Trigger Audio Announcement ---
                        threading.Thread(target=announce_score, args=(locked_score,), daemon=True).start()
                        
                        # --- Push to Hardware LCD ---
                        if lcd_available:
                            try:
                                lcd.text(f"Pose: {locked_pose_name}", 1) 
                                lcd.text(f"Score: {locked_score}/10", 2)
                            except Exception as e:
                                print(f"LCD Error: {e}")

            # STATE_LOCKED: Do nothing, score stays frozen until 'R' is pressed.

            # ---- FPS ----
            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime

            # ---- UI Rendering ----
            h, w, _ = image.shape

            if state == STATE_WAITING:
                # Blue header bar
                cv2.rectangle(image, (0, 0), (w, 100), (180, 120, 30), -1)
                cv2.putText(image, 'WAITING FOR POSE...', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Hold any pose for {int(HOLD_DURATION)}s to lock score', (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            elif state == STATE_HOLDING:
                elapsed = now - hold_start_time
                remaining = max(0, HOLD_DURATION - elapsed)
                progress = elapsed / HOLD_DURATION
                
                # Yellow/Orange header bar
                cv2.rectangle(image, (0, 0), (w, 100), (0, 180, 255), -1)
                
                # Pose name
                cv2.putText(image, f'HOLDING: {detected_pose_during_hold}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Countdown timer
                cv2.putText(image, f'Time left: {remaining:.1f}s', (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Live score preview (small)
                cv2.putText(image, f'Live: {raw_score}', (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Progress bar
                bar_x = 15
                bar_y = 85
                bar_w = w - 30
                bar_h = 10
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 255, 0), -1)
                
                # Feedback tips below the bar
                if feedback and len(feedback) > 0:
                    y_offset = 130
                    cv2.putText(image, "Tips:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    for f in feedback:
                        y_offset += 30
                        cv2.putText(image, f"- {f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            elif state == STATE_LOCKED:
                # Green header bar
                cv2.rectangle(image, (0, 0), (w, 120), (0, 160, 0), -1)
                
                cv2.putText(image, 'SCORE LOCKED', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, locked_pose_name, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Big frozen score
                score_text = f"{locked_score}/10"
                cv2.putText(image, score_text, (w - 220, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4, cv2.LINE_AA)
                
                # Reset instruction
                cv2.putText(image, "Press 'R' to reset for next competitor", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

            # FPS (Bottom left, always visible)
            cv2.putText(image, f"FPS: {int(fps)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Display
            cv2.imshow('Yoga Pose Competition Judge', image)
            
            # Key mappings
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset for next competitor
                state = STATE_WAITING
                score_buffer = []
                hold_start_time = 0.0
                locked_score = 0
                locked_pose_name = ""
                print(">>> RESET: Ready for next competitor <<<")
                
                # --- Clear Hardware LCD ---
                if lcd_available:
                    try:
                        lcd.clear()
                    except Exception as e:
                        pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

