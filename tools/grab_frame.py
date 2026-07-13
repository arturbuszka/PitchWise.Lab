"""Grab a frame from a video for visual inspection."""
import sys
import cv2

cap = cv2.VideoCapture(sys.argv[1])
cap.set(cv2.CAP_PROP_POS_FRAMES, int(sys.argv[3]) if len(sys.argv) > 3 else 100)
ok, frame = cap.read()
assert ok, "could not read frame"
cv2.imwrite(sys.argv[2], frame)
print("saved", sys.argv[2])
