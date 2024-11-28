import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller
from time import sleep

# Initialize the keyboard
keyboard = Controller()

# Define the keyboard layout
keyboard_keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                 ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
                 ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

final_text = ""

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Define a class for buttons on the virtual keyboard
class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

# Define a function to draw buttons on the screen
def draw_buttons(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 144, 30), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
    return img

# Create a list of buttons for the keyboard
buttonList = []
for k in range(len(keyboard_keys)):
    for x, key in enumerate(keyboard_keys[k]):
        buttonList.append(Button([100 * x + 25, 100 * k + 50], key))

# Start capturing video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image for better experience

    # Convert the image to RGB for MediaPipe processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw keyboard buttons
    img = draw_buttons(img, buttonList)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # Check for fingertip (index finger tip is id 8)
            if lmList:
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size

                    if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 255), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

                        # Measure the distance between index finger and middle finger (id 12)
                        finger_distance = np.linalg.norm(np.array([lmList[8][1], lmList[8][2]]) - 
                                                         np.array([lmList[12][1], lmList[12][2]]))

                        # If the distance is small, consider it a click
                        if finger_distance < 30:
                            keyboard.press(button.text)
                            cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                            final_text += button.text
                            sleep(0.3)  # Small delay to avoid multiple key presses

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Display the typed text
    cv2.rectangle(img, (25, 350), (700, 450), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, final_text, (60, 425), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

    # Show the image
    cv2.imshow("Virtual Keyboard", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
