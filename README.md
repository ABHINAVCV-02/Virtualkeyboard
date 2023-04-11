# Virtualkeyboard
How it works
The program uses OpenCV to capture the video feed from the webcam and applies color segmentation
to isolate the green rectangular area that represents the virtual keyboard. The program then uses 
contour detection to identify the individual keys on the keyboard. When the user hovers their finger 
over a key, the program detects the coordinates of the finger and maps it to the corresponding key. 
The program then simulates a key press event by sending the appropriate keyboard input to the operating system.

Limitations
The program relies on color segmentation to isolate the virtual keyboard, 
so lighting conditions can affect the program's accuracy. Additionally, 
the program currently only supports the English alphabet and a limited set of 
punctuation marks. However, the program can be extended to support other languages and characters.
