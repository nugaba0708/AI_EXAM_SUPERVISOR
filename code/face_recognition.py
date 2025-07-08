import cv2
import mediapipe as mp
import numpy as np
import face_recognition
import time
import os
import json
from datetime import datetime

# EAR 계산 함수
def calculate_ear(landmarks, eye_indices):
    left = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    right = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    top = (np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) +
           np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])) / 2
    bottom = (np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]) +
              np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])) / 2
    horizontal = np.linalg.norm(left - right)
    vertical = np.linalg.norm(top - bottom)
    return vertical / horizontal

# 눈 랜드마크 인덱스
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# 얼굴 인증 함수 (수정됨)
def authenticate_face(img_path, dataset_path="./dataset"):
    known_encodings = []
    known_names = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.png')):
            image = face_recognition.load_image_file(os.path.join(dataset_path, filename))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])

    unknown_image = face_recognition.load_image_file(img_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    result_data = {}

    if unknown_encodings:
        results = face_recognition.compare_faces(known_encodings, unknown_encodings[0], tolerance=0.4)
        distances = face_recognition.face_distance(known_encodings, unknown_encodings[0])

        if True in results:
            idx = results.index(True)
            print(f"✅ 신원 확인 성공: {known_names[idx]} (거리: {distances[idx]:.4f})")
            result_data = {
                "authenticated": True,
                "name": known_names[idx],
                "distance": float(f"{distances[idx]:.4f}")
            }
        else:
            print("❌ 신원 확인 실패: 등록된 인물이 아닙니다.")
            result_data = {
                "authenticated": False,
                "reason": "등록된 인물이 아님"
            }
    else:
        print("❗ 얼굴 인식 실패: 이미지에 얼굴이 감지되지 않았습니다.")
        result_data = {
            "authenticated": False,
            "reason": "얼굴 인식 실패"
        }

    return result_data

# 눈 깜빡임 감지 및 신원 확인
def detect_blink_and_authenticate(dataset_path="./dataset"):
    cap = cv2.VideoCapture(2)  # 카메라 인덱스 환경에 맞게 변경
    BLINK_THRESHOLD = 0.21
    BLINK_REQUIRED = 2
    DURATION = 6  # 감지 시간(초)
    MAX_HISTORY = 100

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while True:
            print("⌛ 대기 중: 'c' 키를 누르면 감지를 시작합니다. ('q' 키로 종료)")

            # 대기 루프
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                cv2.putText(frame, "Press 'c' to start blink detection", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('Waiting', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    print("▶ 눈 깜빡임 감지 시작")
                    break
                elif key == ord('q'):
                    print("👋 종료합니다.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            blink_count = 0
            blink_flag = False
            ear_history = []
            ear_log_data = []  # EAR JSON 로그용 리스트
            frame_to_save = None
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                ear = None

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        left_ear = calculate_ear(landmarks, LEFT_EYE)
                        right_ear = calculate_ear(landmarks, RIGHT_EYE)
                        ear = (left_ear + right_ear) / 2.0

                        # EAR 텍스트 출력
                        cv2.putText(frame, f"EAR: {ear:.3f}", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # 눈 깜빡임 감지
                        if ear < BLINK_THRESHOLD and not blink_flag:
                            blink_count += 1
                            blink_flag = True
                            print(f"👁 깜빡임 감지됨! 총 {blink_count}회")
                        elif ear >= BLINK_THRESHOLD:
                            blink_flag = False

                        frame_to_save = frame.copy()

                # EAR 기록
                if ear is not None:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ear_log_data.append({"timestamp": timestamp, "ear": round(ear, 4)})

                    ear_history.append(ear)
                    if len(ear_history) > MAX_HISTORY:
                        ear_history.pop(0)

                    # EAR 그래프 (제거하길 원하시면 이 부분 삭제하세요)
                    for i in range(1, len(ear_history)):
                        pt1 = (frame.shape[1] - MAX_HISTORY + i - 1, 140 - int(ear_history[i - 1] * 100))
                        pt2 = (frame.shape[1] - MAX_HISTORY + i, 140 - int(ear_history[i] * 100))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

                    threshold_y = 140 - int(BLINK_THRESHOLD * 100)
                    cv2.line(frame, (frame.shape[1] - MAX_HISTORY, threshold_y),
                             (frame.shape[1], threshold_y), (0, 0, 255), 1)
                    cv2.putText(frame, "EAR Graph", (frame.shape[1] - MAX_HISTORY, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Blink Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if time.time() - start_time > DURATION:
                    break

            cv2.destroyAllWindows()

            # 깜빡임 결과 평가 및 신원 확인
            if blink_count >= BLINK_REQUIRED and frame_to_save is not None:
                print("✅ 실제 사람으로 판별됨. 신원 확인 시작...")
                img_path = "capture.jpg"
                cv2.imwrite(img_path, frame_to_save)

                auth_result = authenticate_face(img_path, dataset_path)

                # 인증 결과를 로그에 추가
                final_log = {
                    "ear_log": ear_log_data,
                    "blink_count": blink_count,
                    "authentication": auth_result,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # 로그 저장 폴더와 파일명 설정
                log_dir = "./log"
                os.makedirs(log_dir, exist_ok=True)

                # 기존 파일 덮어쓰기 방지 위해 파일명에 시간 추가
                filename = datetime.now().strftime("ear_log_%Y%m%d_%H%M%S.json")
                filepath = os.path.join(log_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(final_log, f, indent=2, ensure_ascii=False)
                    print(f"📝 EAR 및 인증 로그가 '{filepath}' 파일에 저장되었습니다.")

                break
            else:
                print("❌ 사진으로 판별됨 (눈 깜빡임 감지되지 않음). 다시 시도하세요.\n")
                time.sleep(2)


if __name__ == "__main__":
    detect_blink_and_authenticate(dataset_path="./dataset")
