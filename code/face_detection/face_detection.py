import cv2
import mediapipe as mp

# 카메라 설정
cap = cv2.VideoCapture(0)

# Mediapipe 얼굴 감지 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽지 못했습니다.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    # 얼굴 수 초기화
    face_count = 0

    if results.detections:
        face_count = len(results.detections)
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # 감지된 얼굴 수 좌측 상단에 출력
    cv2.putText(frame, f'Faces: {face_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow('Mediapipe Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
