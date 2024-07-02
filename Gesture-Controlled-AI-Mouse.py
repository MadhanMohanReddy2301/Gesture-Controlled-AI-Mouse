import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cap.read()
    f_height, f_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand_landmarks in hands:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Get landmarks of interest (e.g., index finger, mid finger, thumb)
            for id, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * f_width)
                y = int(landmark.y * f_height)
                
                if id == 8:  # Index finger
                    index_x = screen_w / f_width * x
                    index_y = screen_h / f_height * y
                    pyautogui.moveTo(index_x, index_y)
                    
                if id == 11:  # Middle finger
                    mid_x = screen_w / f_width * x
                    mid_y = screen_h / f_height * y
                    
                if id == 4:  # Thumb
                    thumb_x = screen_w / f_width * x
                    thumb_y = screen_h / f_height * y
                    
            # Perform action based on thumb and middle finger distance
            if abs(mid_y - thumb_y) < 20:
                pyautogui.click()
                pyautogui.sleep(0.5)
    
    cv2.imshow('AI Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
