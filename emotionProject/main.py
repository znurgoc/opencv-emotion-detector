import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)




def detect_emotion(landmarks, image_width, image_height):
    def get_point(index):
        lm = landmarks[index]
        return np.array([int(lm.x * image_width), int(lm.y * image_height)])


    brow_point = get_point(65)
    eye_point = get_point(159)

    brow_lift = np.linalg.norm(brow_point - eye_point)


    mouth_left = get_point(61)
    mouth_right = get_point(291)


    mouth_width = np.linalg.norm(mouth_left - mouth_right)


    if brow_lift >18:
        return "Saskin"
    elif mouth_width > 50:
        return "Mutlu"
    else:
        return "Notr"


while True:
    success, frame = cap.read()
    if not success:
        print("Kamera açılamadı veya video akışı kesildi.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape


    emotion = "Yuz Yok"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            emotion = detect_emotion(face_landmarks.landmark, w, h)

            cv2.putText(frame, f"Duygu: {emotion}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,

                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                             circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1)
            )


    cv2.imshow("Canli  duygu takibi", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()