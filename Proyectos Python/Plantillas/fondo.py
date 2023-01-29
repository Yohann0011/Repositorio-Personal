import os
import cv2
import mediapipe as mp
import os
di = os.path.dirname(os.path.abspath(__file__)) + '/Videos' + '/nate.mp4'
mp_selfie_segmentation = mp.solutions.selfie_segmentation
cap = cv2.VideoCapture(di)
with mp_selfie_segmentation.SelfieSegmentation(
     model_selection=1) as selfie_segmentation:
     while True:
          ret, frame = cap.read()
          if ret == False:
               break
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = selfie_segmentation.process(frame_rgb)
          cv2.imshow("Frame", frame)
          cv2.imshow("Mask", results.segmentation_mask)
          if cv2.waitKey(1) & 0xFF == 27:
               break
cap.release()
cv2.destroyAllWindows()