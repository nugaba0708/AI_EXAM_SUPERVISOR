import cv2
import mediapipe as mp
import numpy as np
import time

# 눈 영역 포인트 (FaceMesh 기준)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 387, 385, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh

def get_gaze_ratio(eye_indices, landmarks, frame, gray):
    h, w = frame.shape[:2]
    eye_region = np.array([(int(landmarks[i][0] * w), int(landmarks[i][1] * h)) for i in eye_indices], np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    th_h, th_w = threshold_eye.shape
    left_white = cv2.countNonZero(threshold_eye[:, 0:int(th_w / 2)])
    right_white = cv2.countNonZero(threshold_eye[:, int(th_w / 2):])

    if left_white == 0 or right_white == 0:
        return 1  # 눈 감은 것으로 처리
    else:
        return left_white / right_white

def main():
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    gaze_baseline = None
    gaze_margin = 0.5
    frame_count = 0
    baseline_frames = 30
    baseline_sum = 0
    last_output_time = time.time()

    print("눈동자 추적을 시작합니다. 기준값 설정 중...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0]
            landmarks = [(lm.x, lm.y) for lm in mesh.landmark]

            gaze_left = get_gaze_ratio(LEFT_EYE, landmarks, frame, gray)
            gaze_right = get_gaze_ratio(RIGHT_EYE, landmarks, frame, gray)
            gaze_ratio = (gaze_left + gaze_right) / 2

            if frame_count < baseline_frames:
                baseline_sum += gaze_ratio
                frame_count += 1
                if frame_count == baseline_frames:
                    gaze_baseline = baseline_sum / baseline_frames
                    print(f"기준 시선 값 설정 완료: {gaze_baseline:.2f}")
                continue

            if gaze_baseline is not None:
                if time.time() - last_output_time >= 3:
                    if gaze_ratio < gaze_baseline - gaze_margin:
                        print("눈이 왼쪽으로 움직였습니다!")
                        last_output_time = time.time()
                    elif gaze_ratio > gaze_baseline + gaze_margin:
                        print("눈이 오른쪽으로 움직였습니다!")
                        last_output_time = time.time()

        cv2.imshow("MediaPipe Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
