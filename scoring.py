import numpy as np

# Dictionary to hold the target angles for key joints for different poses.
TARGET_POSES = {
    "Warrior II": {
        "description": "Arms extended parallel to the floor, front leg bent at 90 degrees.",
        "angles": {
            "left_elbow": 180,
            "right_elbow": 180,
            "left_shoulder": 90,
            "right_shoulder": 90,
            "right_knee": 90,
            "left_knee": 180
        }
    },
    "Tree Pose": {
        "description": "One leg straight, the other bent with foot on the inner thigh of the standing leg.",
        "angles": {
            "left_knee": 180, 
            "right_knee": 60, 
            "left_hip": 180   
        }
    },
    "Downward Dog": {
        "description": "Body in an inverted V shape.",
        "angles": {
            "left_hip": 90,   
            "right_hip": 90,
            "left_shoulder": 180, 
            "right_shoulder": 180,
            "left_knee": 180, 
            "right_knee": 180
        }
    }
}

def get_coords(landmarks, landmark_id):
    """Safely extracts X,Y coordinates of a specific MediaPipe landmark."""
    landmark = landmarks.landmark[landmark_id]
    return np.array([landmark.x, landmark.y])

def calculate_pose_score(target_angles, current_angles):
    """
    Internal function to calculate the score for a specific set of target angles.
    It includes a grace margin so humans don't lose points for minor jitter.
    """
    total_error = 0
    joints_checked = 0
    feedback = []

    # Maximum degrees a joint can be off and still be considered "perfect" (10/10)
    ANGLE_TOLERANCE = 15.0

    for joint, target_angle in target_angles.items():
        if joint in current_angles:
            actual_angle = current_angles[joint]
            error = abs(target_angle - actual_angle)
            
            # Apply concession/tolerance margin
            if error <= ANGLE_TOLERANCE:
                error = 0.0
            else:
                # Subtract tolerance so score degrades smoothly once outside the perfect zone
                error -= ANGLE_TOLERANCE

            total_error += error
            joints_checked += 1
            
            if error > 20.0:
                feedback.append(f"Fix {joint.replace('_', ' ')}")
            
    if joints_checked == 0:
        return 0, []

    avg_error = total_error / joints_checked
    
    # We lowered the threshold because the first 15 degrees of error are already forgiven!
    MAX_ERROR_THRESHOLD = 35.0
    
    score = 10.0 * (1 - (avg_error / MAX_ERROR_THRESHOLD))
    score = max(0, min(10, score))
    
    return round(score, 1), feedback

def evaluate_all_poses(current_angles):
    """
    Scans through all defined poses and returns the best matching pose,
    its score, and corresponding feedback. Returns 'No pose detected' if
    no pose scores high enough.
    """
    best_pose_name = "No pose detected"
    best_score = 0.0
    best_feedback = []
    
    DETECTION_THRESHOLD = 6.0 # Minimum score (/10) required to recognize a pose
    
    for pose_name, pose_data in TARGET_POSES.items():
        score, feedback = calculate_pose_score(pose_data["angles"], current_angles)
        
        # If this pose scores higher than any we've seen so far, it becomes the new best candidate
        if score > best_score:
            best_score = score
            best_feedback = feedback
            best_pose_name = pose_name
            
    # If the best matching pose is still too far off, default to neutral state
    if best_score < DETECTION_THRESHOLD:
        return "No pose detected", 0.0, []
        
    return best_pose_name, best_score, best_feedback
