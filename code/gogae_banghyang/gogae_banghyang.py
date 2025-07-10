import cv2
import mediapipe as mp
import time

# MediaPipe 얼굴 메쉬 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# 키 포인트 인덱스 정의
LEFT_EYE = 33     # 왼쪽 눈 바깥
RIGHT_EYE = 263   # 오른쪽 눈 바깥
NOSE_TIP = 1      # 코 끝

# 고정 임계값들
X_THRESHOLD = 0.2   # 좌우 회전 임계값
Y_THRESHOLD = 0.45   # 아래쪽 임계값
SUSTAINED_TIME = 2.0  # 지속 시간 (초)

# 상태 추적 변수들
is_abnormal_state = False  # 현재 비정상 상태인지 (Forward가 아님)
abnormal_start_time = time.time()  # 비정상 상태가 시작된 시간
is_violation_detected = False

def get_head_direction(landmarks, image_w, image_h):
    """방향 판단"""
    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]
    nose_tip = landmarks[NOSE_TIP]
    
    left_eye_x = left_eye.x * image_w
    right_eye_x = right_eye.x * image_w
    nose_x = nose_tip.x * image_w
    nose_y = nose_tip.y * image_h
    eye_center_y = ((left_eye.y + right_eye.y) / 2) * image_h
    
    # 가로 기준
    face_center_x = (left_eye_x + right_eye_x) / 2
    offset_x = nose_x - face_center_x
    face_width = abs(right_eye_x - left_eye_x)
    
    # 세로 기준
    offset_y = nose_y - eye_center_y
    face_height = face_width
    
    if face_width == 0:
        return "Forward"
    
    x_ratio = offset_x / face_width
    y_ratio = offset_y / face_height
    
    # 방향 판단
    if y_ratio > Y_THRESHOLD:
        return "Down"
    elif x_ratio > X_THRESHOLD:
        return "Left"
    elif x_ratio < -X_THRESHOLD:
        return "Right"
    else:
        return "Forward"

# 카메라 찾기
def find_camera():
    for camera_idx in [2, 0, 1, 3]:
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ 카메라 {camera_idx}번 연결됨")
                return cap
            cap.release()
    return None

# 웹캠 열기
cap = find_camera()
if cap is None:
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("고개 감지 시작됨. ESC 키를 누르면 종료됩니다.")
print(f"설정: 정면 외 방향이 {SUSTAINED_TIME}초 이상 지속 시 부정행위 감지")
print("-" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    
    results = face_mesh.process(frame_rgb)
    direction = "No Face"
    current_time = time.time()
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            direction = get_head_direction(face_landmarks.landmark, img_w, img_h)
            
            # 얼굴 랜드마크 시각화
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )
    
    # 비정상 상태 판단 (정면이 아닌 모든 방향)
    current_abnormal = direction in ["Left", "Right", "Down"]
    
    # 정상/비정상 상태가 바뀌었을 때만 타이머 리셋
    if current_abnormal != is_abnormal_state:
        is_abnormal_state = current_abnormal
        abnormal_start_time = current_time
        is_violation_detected = False
    
    # 지속 시간 계산 (비정상 상태일 때만)
    if is_abnormal_state:
        duration = current_time - abnormal_start_time
        # 비정상 상태가 지속시간 이상 계속되면 위반
        if duration >= SUSTAINED_TIME:
            is_violation_detected = True
    else:
        duration = 0
        is_violation_detected = False
    
    # 화면 표시용 색상 및 텍스트 결정
    if direction == "No Face":
        color = (0, 0, 255)  # 빨간색
        status_text = "얼굴 없음"
    elif is_violation_detected:
        color = (0, 0, 255)  # 빨간색 (위반 감지)
        status_text = f"부정행위 감지! ({direction})"
    elif direction == "Forward":
        color = (0, 255, 0)  # 초록색 (정상)
        status_text = "정상"
    else:
        color = (0, 165, 255)  # 주황색 (비정상이지만 아직 위반 아님)
        status_text = f"비정상 상태 {duration:.1f}초 ({direction})"
    
    # 정보 표시
    cv2.putText(frame, f"Direction: {direction}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.putText(frame, f"Status: {status_text}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 지속 시간 표시 (비정상 상태일 때만)
    if is_abnormal_state:
        progress = min(duration / SUSTAINED_TIME, 1.0) * 100
        cv2.putText(frame, f"비정상 지속시간: {duration:.1f}s ({progress:.0f}%)", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow("Head Direction Detection", frame)
    
    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
