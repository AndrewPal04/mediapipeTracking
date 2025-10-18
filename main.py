import cv2
import mediapipe as mp
import pygame

pygame.mixer.pre_init(44100, -16, 2, 256)
pygame.init()
sound=pygame.mixer.Sound("noise.mp3")
channel=pygame.mixer.Channel(0)

thinker=cv2.imread("theThinker.png",cv2.IMREAD_UNCHANGED)
thinker=cv2.resize(thinker, (1280, 720))
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

was_inside = False

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_result = face_detection.process(rgb)
        hand_result = hands.process(rgb)

        if face_result.detections:
            for det in face_result.detections:
                rb = det.location_data.relative_bounding_box
                x = int(rb.xmin * w)
                y = int(rb.ymin * h)
                bw = int(rb.width * w)
                bh = int(rb.height * h)
                side = max(bw, bh)
                # cv2.rectangle(frame, (x, y), (x + side, y + side), (150, 50, 0), 2)  #Draws rectangle around face for testing
        
        now_inside = False

        if hand_result.multi_hand_landmarks:
            for hlms in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

                finger=hlms.landmark[8]
                fx,fy=int(finger.x*w),int(finger.y*h)
                if x is not None and y is not None:
                    if (x<=fx<=x+side) and (y<=fy<=y+side):
                        now_inside = True
        if now_inside and not was_inside:
            channel.play(sound)
        was_inside = now_inside
        if now_inside:
            cv2.imshow("Webcam",thinker)
        else:
            cv2.imshow("Webcam",frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
