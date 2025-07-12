import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import json
import sys
import os
import face_recognition
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
from pathlib import Path
import queue
import time
import subprocess

tts_queue = queue.Queue()

# TTS import
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class SimpleExamSupervisor:
    """단순화된 AI 시험 감독관 - 원본 기능 완전 보존"""
    
    def __init__(self):
        self.setup_config()
        self.setup_mediapipe()
        self.setup_variables()
        self.setup_gui()
        
        # 카메라 상태
        self.cap = None
        self.is_running = False
        self.current_frame = None
        
        print("🔒 AI 시험 감독관 시스템 v2.6 (단순화) 초기화 완료")
        
    def setup_config(self):
        default_config = {
            "camera": {"index": 0, "width": 640, "height": 480, "fps": 20, "mirror": True},
            "detection": {"x_threshold": 0.15, "y_threshold": 0.5, "sustained_time": 2.0, 
                         "gaze_margin": 0.6, "face_lost_threshold": 1.0},
            "identity": {"dataset_path": "./dataset", "tolerance": 0.4, "max_attempts": 5,
                        "blink_threshold": 0.21, "blink_required": 2, "blink_detection_duration": 6},
            "system": {"baseline_frames": 30, "save_video": True, "log_path": "./logs", "max_warnings": 5}
        }
        
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                    else:
                        for subkey in default_config[key]:
                            if subkey not in config[key]:
                                config[key][subkey] = default_config[key][subkey]
            except:
                config = default_config
        else:
            config = default_config
            
        self.config = config
        # 설정값 적용
        self.CAMERA_INDEX = config["camera"]["index"]
        self.CAMERA_WIDTH = config["camera"]["width"]
        self.CAMERA_HEIGHT = config["camera"]["height"]
        self.CAMERA_FPS = config["camera"]["fps"]
        self.MIRROR_CAMERA = config["camera"]["mirror"]
        
        self.X_THRESHOLD = config["detection"]["x_threshold"]
        self.Y_THRESHOLD = config["detection"]["y_threshold"]
        self.SUSTAINED_TIME = config["detection"]["sustained_time"]
        self.GAZE_MARGIN = config["detection"]["gaze_margin"]
        self.FACE_LOST_THRESHOLD = config["detection"]["face_lost_threshold"]
        
        self.DATASET_PATH = config["identity"]["dataset_path"]
        self.FACE_TOLERANCE = config["identity"]["tolerance"]
        self.MAX_IDENTITY_ATTEMPTS = config["identity"]["max_attempts"]
        self.BLINK_THRESHOLD = config["identity"]["blink_threshold"]
        self.BLINK_REQUIRED = config["identity"]["blink_required"]
        self.BLINK_DETECTION_DURATION = config["identity"]["blink_detection_duration"]
        
        self.BASELINE_FRAMES = config["system"]["baseline_frames"]
        self.SAVE_VIDEO = config["system"]["save_video"]
        self.LOG_PATH = config["system"]["log_path"]
        self.MAX_WARNINGS = config["system"]["max_warnings"]
        
    def setup_mediapipe(self):
        """MediaPipe 초기화"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True, static_image_mode=False, max_num_faces=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 랜드마크 포인트들
        self.NOSE_TIP = 1
        self.LEFT_EYE_LEFT = 33
        self.RIGHT_EYE_RIGHT = 263
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291
        self.CHIN = 18
        
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 387, 385, 263, 373, 380]
        self.LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
        self.MAX_EAR_HISTORY = 100
        
    def setup_variables(self):
        """상태 변수 초기화"""
        # 시스템 상태
        self.system_phase = "IDLE"
        self.authenticated_user = None
        self.exam_start_time = None
        self.exam_terminated = False
        self.termination_reason = ""
        
        # 경고 시스템
        self.total_warnings = 0
        self.head_warnings = 0
        self.gaze_warnings = 0
        
        # 아이트래킹 변수들
        self.gaze_baseline = None
        self.frame_count = 0
        self.baseline_sum = 0
        self.last_gaze_violation_time = 0
        
        # 얼굴 추적 변수들
        self.last_face_landmarks = None
        self.face_lost_time = 0
        
        # 로깅 시스템
        self.violation_log = []
        self.total_violations = 0
        self.identity_attempts = 0
        
        # 위반 상태들
        self.reset_violation_states()
        
        # 폴더 생성
        Path(self.DATASET_PATH).mkdir(exist_ok=True)
        Path(self.LOG_PATH).mkdir(exist_ok=True)
        
    def create_gui(self):
        """간단한 GUI 생성"""
        # 메인 컨테이너
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 왼쪽: 카메라
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # 카메라 표시
        camera_frame = ttk.LabelFrame(left_frame, text="📹 실시간 카메라")
        camera_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.camera_label = tk.Label(camera_frame, text="카메라 대기 중...", 
                                   bg="black", fg="white", font=("Arial", 14))
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 상태 표시
        status_frame = ttk.LabelFrame(left_frame, text="📊 시스템 상태")
        status_frame.pack(fill="x")
        
        # 상태 라벨들 생성
        self.create_status_labels(status_frame)
        
        # 오른쪽: 제어 및 로그
        right_frame = tk.Frame(main_frame, width=350)
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)
        
        # 제어 버튼들
        self.create_control_buttons(right_frame)
        
        # 로그
        self.create_log_panel(right_frame)
        
    def setup_gui(self):
        """GUI 초기화"""
        self.root = tk.Tk()
        self.root.title("🤖 AI 시험 감독관 v2.6 (단순화)")
        self.root.geometry("1200x800")
        
        self.create_gui()
        
    def create_status_labels(self, parent):
        """상태 라벨 생성"""
        # 2열 구성
        info_frame = tk.Frame(parent)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        left_col = tk.Frame(info_frame)
        left_col.pack(side="left", fill="both", expand=True)
        right_col = tk.Frame(info_frame)
        right_col.pack(side="right", fill="both", expand=True)
        
        # 상태 아이템들 (GUI 라벨에 직접 접근)
        self.user_label = self.create_status_item(left_col, "👤 사용자:", "미인증")
        self.phase_label = self.create_status_item(left_col, "🔄 단계:", "대기 중")
        self.time_label = self.create_status_item(left_col, "⏰ 시험 시간:", "00:00:00")
        self.face_label = self.create_status_item(right_col, "👥 얼굴:", "0명")
        self.head_label = self.create_status_item(right_col, "🎯 고개:", "정면")
        self.gaze_label = self.create_status_item(right_col, "👁 시선:", "대기 중")
        self.warning_label = self.create_status_item(right_col, "⚠️ 경고:", "0/5")
        
    def create_status_item(self, parent, label_text, default_value):
        """상태 아이템 생성"""
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        tk.Label(frame, text=label_text, width=12, anchor="w").pack(side="left")
        value_label = tk.Label(frame, text=default_value, anchor="w", 
                              font=("Arial", 9, "bold"))
        value_label.pack(side="left")
        
        return value_label
        
    def create_control_buttons(self, parent):
        """제어 버튼 생성"""
        control_frame = ttk.LabelFrame(parent, text="🎛️ 시스템 제어")
        control_frame.pack(fill="x", pady=(0, 10), padx=5)
        
        # 버튼들
        tk.Button(control_frame, text="▶️ 시스템 시작", command=self.start_system,
                 bg="#28a745", fg="white", font=("Arial", 11, "bold")).pack(fill="x", pady=5, padx=10)
        
        tk.Button(control_frame, text="👤 신원 확인", command=self.start_identity,
                 bg="#17a2b8", fg="white", font=("Arial", 11)).pack(fill="x", pady=5, padx=10)
        
        tk.Button(control_frame, text="🔍 시험 감독", command=self.start_monitoring,
                 bg="#ffc107", fg="black", font=("Arial", 11)).pack(fill="x", pady=5, padx=10)
        
        tk.Button(control_frame, text="⏹️ 시스템 중지", command=self.stop_system,
                 bg="#dc3545", fg="white", font=("Arial", 11)).pack(fill="x", pady=5, padx=10)
        
    def create_log_panel(self, parent):
        """로그 패널"""
        log_frame = ttk.LabelFrame(parent, text="📝 실시간 로그")
        log_frame.pack(fill="both", expand=True, padx=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, wrap=tk.WORD, 
                                                font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        
    # =====================================
    # 메인 로직
    # =====================================
    
    def start_system(self):
        """시스템 시작"""
        if self.is_running:
            return
            
        # 카메라 찾기
        self.cap = self.find_camera()
        if self.cap is None:
            messagebox.showerror("오류", "카메라를 찾을 수 없습니다!")
            return
            
        self.is_running = True
        self.log_message("시스템이 시작되었습니다.", "SUCCESS")
        
        # 메인 루프 시작 (단일 타이머)
        self.update_loop()
        
    def find_camera(self):
        """카메라 찾기"""
        camera_indices = [self.CAMERA_INDEX] + [i for i in range(5) if i != self.CAMERA_INDEX]
        
        for camera_idx in camera_indices:
            try:
                cap = cv2.VideoCapture(camera_idx)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, self.CAMERA_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.log_message(f"✅ 카메라 {camera_idx}번 연결 성공 ({self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT}@{self.CAMERA_FPS}fps)", "SUCCESS")
                        return cap
                    cap.release()
            except:
                continue
        return None
        
    def update_loop(self):
        """메인 업데이트 루프"""
        if not self.is_running or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(50, self.update_loop)
            return
            
        if self.MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)
            
        self.current_frame = frame.copy()
        
        # 단계별 처리
        if self.system_phase == "IDENTITY_CHECK":
            self.process_identity_frame(frame)
        elif self.system_phase == "EXAM_MONITORING":
            self.process_monitoring_frame(frame)
            
        # GUI 업데이트 (직접 호출)
        self.update_camera_display(frame)
        self.update_status_display()
        
        # 다음 프레임 스케줄링
        self.root.after(33, self.update_loop)  # ~30 FPS
        
    def update_camera_display(self, frame):
        """카메라 화면 업데이트"""
        try:
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 크기 조정
            display_width = 600
            display_height = 450
            frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
            
            # PIL 변환
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)
            
            # 라벨 업데이트
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo  # 참조 유지
            
        except Exception as e:
            pass
            
    def update_status_display(self):
        """상태 정보 직접 업데이트"""
        # 사용자 상태
        if self.authenticated_user:
            self.user_label.configure(text=self.authenticated_user, fg="green")
        else:
            self.user_label.configure(text="미인증", fg="red")
            
        # 단계 상태
        self.phase_label.configure(text=self.system_phase)
        
        # 시험 시간
        if self.exam_start_time:
            elapsed = int(time.time() - self.exam_start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.time_label.configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
        # 경고 상태
        color = "red" if self.total_warnings >= self.MAX_WARNINGS else "orange" if self.total_warnings > 0 else "green"
        self.warning_label.configure(text=f"{self.total_warnings}/{self.MAX_WARNINGS}", fg=color)
        
    # =====================================
    # 신원 확인
    # =====================================
    
    def start_identity(self):
        """신원 확인 시작"""
        if not self.is_running:
            messagebox.showwarning("경고", "시스템을 먼저 시작하세요!")
            return
            
        # 데이터셋 확인
        dataset_files = list(Path(self.DATASET_PATH).glob("*.jpg")) + \
                       list(Path(self.DATASET_PATH).glob("*.jpeg")) + \
                       list(Path(self.DATASET_PATH).glob("*.png"))
        
        if not dataset_files:
            messagebox.showwarning("경고", "데이터셋 폴더에 얼굴 이미지가 없습니다.")
            return
            
        self.system_phase = "IDENTITY_CHECK"
        self.identity_attempts = 0
        self.log_message("🔍 신원 확인 시작")
        
        # 깜빡임 감지 시작
        self.start_blink_detection()
        
    def start_blink_detection(self):
        """깜빡임 감지 시작"""
        self.identity_attempts += 1
        self.blink_count = 0
        self.blink_flag = False
        self.blink_start_time = time.time()
        self.blink_detection_active = True
        
        self.log_message(f"👁 {self.identity_attempts}번째 시도: {self.BLINK_DETECTION_DURATION}초 동안 {self.BLINK_REQUIRED}회 깜빡이세요")
        speak_tts("신원 조회를 시작합니다. 카메라를 바라봐 주세요.")
        
    def process_identity_frame(self, frame):
        """신원 확인 프레임 처리"""
        if not hasattr(self, 'blink_detection_active') or not self.blink_detection_active:
            return
            
        # MediaPipe 처리
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb)
            ear = None
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # EAR 계산
                    left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_EAR)
                    right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_EAR)
                    ear = (left_ear + right_ear) / 2.0
                    
                    # 깜빡임 감지
                    if ear < self.BLINK_THRESHOLD and not self.blink_flag:
                        self.blink_count += 1
                        self.blink_flag = True
                        self.log_message(f"👁 깜빡임 감지! 총 {self.blink_count}회")
                        
                    elif ear >= self.BLINK_THRESHOLD:
                        self.blink_flag = False
                        
            # 화면에 상태 표시
            elapsed_time = time.time() - self.blink_start_time
            remaining_time = max(0, self.BLINK_DETECTION_DURATION - elapsed_time)
            
            cv2.putText(frame, f"Blinks: {self.blink_count}/{self.BLINK_REQUIRED}", 
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {remaining_time:.1f}s", 
                       (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if ear is not None:
                color = (0, 255, 0) if ear >= self.BLINK_THRESHOLD else (0, 0, 255)
                cv2.putText(frame, f"EAR: {ear:.3f}", 
                           (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
        # 시간 확인
        if time.time() - self.blink_start_time > self.BLINK_DETECTION_DURATION:
            self.complete_blink_detection()
            
    def complete_blink_detection(self):
        """깜빡임 감지 완료"""
        self.blink_detection_active = False
        
        if self.blink_count >= self.BLINK_REQUIRED and self.current_frame is not None:
            self.log_message(f"✅ 실제 사람 판별됨! ({self.blink_count}회)", "SUCCESS")
            self.log_message("🔍 얼굴 인식 처리 중...")
            
            # 얼굴 비교
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"identity_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            
            # 원본 얼굴 비교 함수 사용
            matched_name = self.compare_with_dataset(filename)
            
            if matched_name:
                self.authenticated_user = matched_name
                self.log_message(f"🎉 인증 성공! {matched_name}님, 환영합니다.", "SUCCESS")
                self.system_phase = "IDLE"
                speak_tts(f"{matched_name}님, 신원 확인이 완료되었습니다.")
            else:
                self.log_message(f"❌ 얼굴 인증 실패 ({self.identity_attempts}/{self.MAX_IDENTITY_ATTEMPTS})", "ERROR")
                if self.identity_attempts < self.MAX_IDENTITY_ATTEMPTS:
                    self.log_message(f"다시 시도하세요. 남은 횟수: {self.MAX_IDENTITY_ATTEMPTS - self.identity_attempts}회")
                    self.start_blink_detection()  # 재시도
                else:
                    self.log_message("❌ 신원 확인 실패: 최대 시도 횟수 초과", "ERROR")
                    self.system_phase = "IDLE"
                    
            try:
                os.remove(filename)
            except:
                pass
        else:
            self.log_message(f"❌ 사진으로 판별됨 (눈 깜빡임 {self.blink_count}회 < {self.BLINK_REQUIRED}회)", "ERROR")
            self.log_message("실제 사람인지 확인하기 위해 다시 시도해주세요.")
            if self.identity_attempts < self.MAX_IDENTITY_ATTEMPTS:
                self.start_blink_detection()  # 재시도
            else:
                self.log_message("❌ 신원 확인 실패: 최대 시도 횟수 초과", "ERROR")
                self.system_phase = "IDLE"
                
    # =====================================
    #               시험 감독
    # =====================================
    
    def start_monitoring(self):
        """시험 감독 시작"""
        if not self.authenticated_user:
            messagebox.showwarning("경고", "신원 확인을 먼저 완료하세요!")
            return
            
        self.system_phase = "EXAM_MONITORING"
        self.exam_start_time = time.time()
        self.exam_terminated = False
        
        # 상태 초기화
        self.reset_violation_states()
        self.total_warnings = 0
        self.head_warnings = 0
        self.gaze_warnings = 0
        self.violation_log = []
        self.total_violations = 0
        
        # 시선 캘리브레이션 초기화
        self.gaze_baseline = None
        self.frame_count = 0
        self.baseline_sum = 0
        
        self.log_message("📝 시험 감독 시작", "SUCCESS")
        self.log_message(f"응시자: {self.authenticated_user}", "SUCCESS")
        self.log_message("실시간 부정행위 탐지 시작...")
        speak_tts("시험 감독이 시작되었습니다.")
        
    def process_monitoring_frame(self, frame):
        """시험 감독 프레임 처리"""
        if self.exam_terminated:
            return
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w = frame.shape[:2]
        
        results = self.face_mesh.process(frame_rgb)
        
        face_count = 0
        head_direction = "No Face"
        x_ratio, y_ratio = 0, 0
        gaze_ratio = 1
        
        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)
            
            # 첫 번째 얼굴 분석
            best_face = results.multi_face_landmarks[0]
            current_landmarks = self.get_landmarks_coords(best_face, img_w, img_h)
            
            # 머리 방향
            head_direction, x_ratio, y_ratio = self.get_head_direction(current_landmarks, img_w, img_h)
            
            # 시선 분석
            landmarks_normalized = [(lm.x, lm.y) for lm in best_face.landmark]
            gaze_left = self.get_gaze_ratio(self.LEFT_EYE, landmarks_normalized, frame, gray)
            gaze_right = self.get_gaze_ratio(self.RIGHT_EYE, landmarks_normalized, frame, gray)
            current_gaze = (gaze_left + gaze_right) / 2
            
            # 캘리브레이션 처리
            if self.frame_count < self.BASELINE_FRAMES:
                self.baseline_sum += current_gaze
                self.frame_count += 1
                
                if self.frame_count == self.BASELINE_FRAMES:
                    self.gaze_baseline = self.baseline_sum / self.BASELINE_FRAMES
                    self.log_message(f"✅ 시선 캘리브레이션 완료! (기준값: {self.gaze_baseline:.2f})", "SUCCESS")
                    speak_tts("시험을 시작합니다.")
                    
                gaze_ratio = current_gaze
            else:
                gaze_ratio = current_gaze
                
        # GUI 상태 직접 업데이트
        self.face_label.configure(text=f"{face_count}명", 
                                fg="green" if face_count == 1 else "red")
        
        if head_direction == "Forward":
            self.head_label.configure(text="정면", fg="green")
        elif head_direction == "No Face":
            self.head_label.configure(text="감지 안됨", fg="red")
        else:
            self.head_label.configure(text=head_direction, fg="orange")
            
        # 시선 상태 업데이트
        if self.gaze_baseline is None:
            progress = (self.frame_count / self.BASELINE_FRAMES) * 100
            self.gaze_label.configure(text=f"캘리브레이션 중 {progress:.0f}%", fg="blue")
        else:
            if gaze_ratio != 1:
                current_gaze_abnormal = (gaze_ratio < self.gaze_baseline - self.GAZE_MARGIN or 
                                       gaze_ratio > self.gaze_baseline + self.GAZE_MARGIN)
                if current_gaze_abnormal:
                    direction = "왼쪽" if gaze_ratio < self.gaze_baseline else "오른쪽"
                    self.gaze_label.configure(text=f"{direction} 이탈", fg="orange")
                else:
                    self.gaze_label.configure(text="정면", fg="green")
            else:
                self.gaze_label.configure(text="정면", fg="green")
                    
        # 위반 상태 업데이트
        self.update_violation_states(face_count, head_direction, gaze_ratio)
        
        # 화면에 정보 표시
        self.draw_status_info(frame, face_count, head_direction, gaze_ratio, x_ratio, y_ratio)
        
    def stop_system(self):
        """시스템 중지"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.log_message("시스템이 중지되었습니다.", "WARNING")
        
    # =====================================
    #         핵심 부정행위 탐지!
    # =====================================
    
    def calculate_ear(self, landmarks, eye_indices):
        """EAR 계산"""
        try:
            left = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
            right = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
            top = (np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) +
                   np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])) / 2
            bottom = (np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]) +
                      np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])) / 2
            
            horizontal = np.linalg.norm(left - right)
            vertical = np.linalg.norm(top - bottom)
            
            return vertical / horizontal if horizontal != 0 else 0.0
        except:
            return 0.0
            
    def compare_with_dataset(self, captured_image_path):
        """데이터셋 비교"""
        try:
            known_encodings = []
            known_names = []
            
            dataset_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                dataset_files.extend(Path(self.DATASET_PATH).glob(ext))
            
            if not dataset_files:
                self.log_message("❌ 데이터셋에 이미지가 없습니다.", "ERROR")
                return None

            self.log_message(f"📂 데이터셋에서 {len(dataset_files)}개 파일 로드 중...")
            
            for file_path in dataset_files:
                try:
                    known_image = face_recognition.load_image_file(str(file_path))
                    encodings = face_recognition.face_encodings(known_image)
                    
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(file_path.stem)
                        self.log_message(f"   ✅ {file_path.name} 로드 완료")
                    else:
                        self.log_message(f"   ❗ {file_path.name} 얼굴 인식 실패", "WARNING")
                except Exception as e:
                    self.log_message(f"   ❗ {file_path.name} 파일 처리 오류: {e}", "ERROR")
                    continue

            if not known_encodings:
                self.log_message("❌ 유효한 얼굴 데이터가 없습니다.", "ERROR")
                return None

            unknown_image = face_recognition.load_image_file(captured_image_path)
            unknown_encodings = face_recognition.face_encodings(unknown_image)

            if not unknown_encodings:
                self.log_message("❗ 캡처된 이미지에서 얼굴을 찾지 못했습니다.", "ERROR")
                return None

            unknown_encoding = unknown_encodings[0]
            
            results = face_recognition.compare_faces(known_encodings, unknown_encoding, 
                                                   tolerance=self.FACE_TOLERANCE)
            distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            
            self.log_message("📏 거리 값 (각 등록된 얼굴과의 거리):")
            for name, dist in zip(known_names, distances):
                self.log_message(f"   📊 {name}: {dist:.4f}")
            
            if True in results:
                idx = results.index(True)
                matched_name = known_names[idx]
                distance = distances[idx]
                self.log_message(f"✅ [매칭 성공] {matched_name} (거리: {distance:.4f})", "SUCCESS")
                return matched_name
            else:
                min_distance = min(distances)
                self.log_message(f"❌ [매칭 실패] 최소 거리: {min_distance:.4f} (임계값: {self.FACE_TOLERANCE})", "ERROR")
                self.log_message("등록된 인물이 아닙니다.", "ERROR")
                return None
                
        except Exception as e:
            self.log_message(f"❌ 얼굴 인식 처리 오류: {e}", "ERROR")
            return None
            
    def get_landmarks_coords(self, face_landmarks, image_w, image_h):
        """랜드마크 좌표 변환"""
        coords = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image_w)
            y = int(landmark.y * image_h)
            coords.append([x, y])
        return np.array(coords)
        
    def get_head_direction(self, landmarks, image_w, image_h):
        """머리 방향 판단"""
        nose_tip = landmarks[self.NOSE_TIP]
        left_eye_left = landmarks[self.LEFT_EYE_LEFT]
        right_eye_right = landmarks[self.RIGHT_EYE_RIGHT]
        
        # 얼굴 중심선 계산
        face_width = abs(right_eye_right[0] - left_eye_left[0])
        face_center_x = (left_eye_left[0] + right_eye_right[0]) / 2
        
        # 좌우 방향 판단
        offset_x = nose_tip[0] - face_center_x
        
        # 상하 방향 판단
        eye_center_y = (left_eye_left[1] + right_eye_right[1]) / 2
        offset_y = nose_tip[1] - eye_center_y
        
        if face_width == 0:
            return "Forward", 0, 0
        
        # 정규화
        x_ratio = offset_x / face_width
        y_ratio = offset_y / face_width
        
        # 방향 판단
        if y_ratio > self.Y_THRESHOLD:
            return "Down", x_ratio, y_ratio
        elif x_ratio > self.X_THRESHOLD:
            return "Right", x_ratio, y_ratio
        elif x_ratio < -self.X_THRESHOLD:
            return "Left", x_ratio, y_ratio
        else:
            return "Forward", x_ratio, y_ratio
            
    def get_gaze_ratio(self, eye_indices, landmarks, frame, gray):
        """시선 방향 계산"""
        h, w = frame.shape[:2]
        
        try:
            # 눈 영역 좌표 계산
            eye_region = np.array([(int(landmarks[i][0] * w), int(landmarks[i][1] * h)) 
                                  for i in eye_indices], np.int32)
            
            # 마스크 생성
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [eye_region], 255)
            
            # 눈 영역 추출
            eye = cv2.bitwise_and(gray, gray, mask=mask)
            
            # 경계 좌표 계산
            min_x = np.min(eye_region[:, 0])
            max_x = np.max(eye_region[:, 0])
            min_y = np.min(eye_region[:, 1])
            max_y = np.max(eye_region[:, 1])
            
            # 경계 검사
            if min_x >= max_x or min_y >= max_y or min_x < 0 or min_y < 0 or max_x >= w or max_y >= h:
                return 1.0
            
            # 눈 영역 크롭
            gray_eye = eye[min_y:max_y, min_x:max_x]
            
            if gray_eye.size == 0:
                return 1.0
            
            # 임계값 적용
            _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
            
            th_h, th_w = threshold_eye.shape
            
            if th_w < 2:
                return 1.0
            
            # 좌우 영역의 흰색 픽셀 수 계산
            left_white = cv2.countNonZero(threshold_eye[:, 0:int(th_w / 2)])
            right_white = cv2.countNonZero(threshold_eye[:, int(th_w / 2):])
            
            if left_white == 0 or right_white == 0:
                return 1.0
            else:
                return left_white / right_white
                
        except Exception as e:
            return 1.0
            
    def reset_violation_states(self):
        """위반 상태 초기화"""
        self.is_head_abnormal = False
        self.head_abnormal_start_time = time.time()
        self.is_head_violation = False
        
        self.is_multiple_faces = False
        self.multiple_faces_start_time = time.time()
        self.is_multiple_faces_violation = False
        
        self.is_no_face = False
        self.no_face_start_time = time.time()
        self.is_no_face_violation = False
        
        self.is_gaze_abnormal = False
        self.gaze_abnormal_start_time = time.time()
        self.is_gaze_violation = False
        
    def update_violation_states(self, face_count, head_direction, gaze_ratio):
        """위반 상태 업데이트"""
        current_time = time.time()
        
        # 시험이 이미 중단된 경우 처리 중지
        if self.exam_terminated:
            return
        
        # 1. 다중 인물 감지 - 즉시 중단
        current_multiple_faces = face_count > 1
        if current_multiple_faces != self.is_multiple_faces:
            self.is_multiple_faces = current_multiple_faces
            self.multiple_faces_start_time = current_time
            self.is_multiple_faces_violation = False
        
        if self.is_multiple_faces:
            duration = current_time - self.multiple_faces_start_time
            if duration >= self.SUSTAINED_TIME and not self.is_multiple_faces_violation:
                self.is_multiple_faces_violation = True
                # 부정행위 알림 출력
                self.print_violation_alert("다중 인물 감지", f"감지된 얼굴 수: {face_count}명", 
                                         is_start=True, duration=duration)
                # 시험 즉시 중단
                self.terminate_exam(f"다중 인물 감지 ({face_count}명)")
                return
            elif duration < self.SUSTAINED_TIME and duration > 0.5:
                self.print_warning("다중 인물 감지", f"{face_count}명 감지됨", duration)
        
        # 2. 화면 이탈 감지 - 즉시 중단
        current_no_face = face_count == 0
        if current_no_face != self.is_no_face:
            self.is_no_face = current_no_face
            self.no_face_start_time = current_time
            self.is_no_face_violation = False
        
        if self.is_no_face:
            duration = current_time - self.no_face_start_time
            if duration >= self.SUSTAINED_TIME and not self.is_no_face_violation:
                self.is_no_face_violation = True
                # 부정행위 알림 출력
                self.print_violation_alert("화면 이탈", "얼굴 감지 불가 - 화면에서 완전히 이탈", 
                                         is_start=True, duration=duration)
                # 시험 즉시 중단
                self.terminate_exam("화면 이탈 (얼굴 감지 불가)")
                return
            elif duration < self.SUSTAINED_TIME and duration > 0.5:
                self.print_warning("화면 이탈", "얼굴이 감지되지 않음", duration)
        
        # 3. 고개 방향 감지 - 경고 후 중단
        current_head_abnormal = head_direction in ["Left", "Right", "Down"]
        if current_head_abnormal != self.is_head_abnormal:
            if not current_head_abnormal and self.is_head_violation:
                # 위반 상태 종료
                total_duration = current_time - self.head_abnormal_start_time
                self.log_message(f"고개 방향 정상화 (지속시간: {total_duration:.1f}초)", "SUCCESS")
            
            self.is_head_abnormal = current_head_abnormal
            self.head_abnormal_start_time = current_time
            self.is_head_violation = False
        
        if self.is_head_abnormal:
            duration = current_time - self.head_abnormal_start_time
            if duration >= self.SUSTAINED_TIME and not self.is_head_violation:
                self.is_head_violation = True
                # 경고 발급
                is_terminated = self.issue_warning("고개 방향", f"방향: {head_direction}")
                if is_terminated:
                    return
            elif duration < self.SUSTAINED_TIME and duration > 0.5:
                self.print_warning("고개 방향", f"{head_direction} 방향으로 움직임", duration)
        
        # 4. 시선 이탈 감지 - 경고 후 중단
        if self.gaze_baseline is not None and gaze_ratio != 1:
            current_gaze_abnormal = (gaze_ratio < self.gaze_baseline - self.GAZE_MARGIN or 
                                   gaze_ratio > self.gaze_baseline + self.GAZE_MARGIN)
            
            if current_gaze_abnormal != self.is_gaze_abnormal:
                if not current_gaze_abnormal and self.is_gaze_violation:
                    # 위반 상태 종료
                    total_duration = current_time - self.gaze_abnormal_start_time
                    self.log_message(f"시선 정상화 (지속시간: {total_duration:.1f}초)", "SUCCESS")
                
                self.is_gaze_abnormal = current_gaze_abnormal
                self.gaze_abnormal_start_time = current_time
                self.is_gaze_violation = False
            
            if self.is_gaze_abnormal:
                duration = current_time - self.gaze_abnormal_start_time
                if duration >= self.SUSTAINED_TIME and not self.is_gaze_violation:
                    self.is_gaze_violation = True
                    direction = "왼쪽" if gaze_ratio < self.gaze_baseline else "오른쪽"
                    # 경고 발급
                    is_terminated = self.issue_warning("시선 이탈", f"{direction} 방향으로 시선 이탈")
                    if is_terminated:
                        return
                elif duration < self.SUSTAINED_TIME and duration > 0.5:
                    direction = "왼쪽" if gaze_ratio < self.gaze_baseline else "오른쪽"
                    deviation = abs(gaze_ratio - self.gaze_baseline)
                    self.print_warning("시선 이탈", f"{direction} 시선 (편차: {deviation:.2f})", duration)
                    
    def terminate_exam(self, reason):
        """시험 중단"""
        self.exam_terminated = True
        self.termination_reason = reason
        
        self.log_message("🚨 부정행위 탐지! 시험 즉시 중단! 🚨", "ERROR")
        self.log_message(f"사유: {reason}", "ERROR")
        self.log_message("심각한 부정행위가 탐지되어 시험이 즉시 중단되었습니다.", "ERROR")
        self.log_message("이 결과는 시험 관리자에게 보고됩니다.", "ERROR")
        
        speak_tts("심각한 부정행위가 탐지되어 시험을 중단합니다.")
        
        # 로그 기록
        self.log_violation("부정행위-시험중단", reason)
        
    def issue_warning(self, warning_type, details):
        """경고 발급 (통합 5회 시스템)"""
        # 개별 경고 횟수 증가 (표시용)
        if warning_type == "고개 방향":
            self.head_warnings += 1
        elif warning_type == "시선 이탈":
            self.gaze_warnings += 1
        
        # 통합 경고 횟수 증가
        self.total_warnings += 1
        remaining = self.MAX_WARNINGS - self.total_warnings
        
        speak_tts(f"{warning_type}부정행위가 탐지되었습니다.")
        
        self.log_message(f"⚠️  경고 {self.total_warnings}/{self.MAX_WARNINGS} - {warning_type}", "WARNING")
        self.log_message(f"상세: {details}", "WARNING")
        
        if self.total_warnings >= self.MAX_WARNINGS:
            self.log_message(f"🚨 총 {self.MAX_WARNINGS}회 경고 누적! 부정행위로 판정됩니다.", "ERROR")
            self.terminate_exam(f"경고 {self.MAX_WARNINGS}회 누적 (고개: {self.head_warnings}회, 시선: {self.gaze_warnings}회)")
            return True
        else:
            self.log_message(f"남은 경고: {remaining}회 (고개: {self.head_warnings}회, 시선: {self.gaze_warnings}회)")
        
        # 로그 기록
        self.log_violation(f"경고-{warning_type}", details)
        return False
        
    def print_violation_alert(self, violation_type, details, is_start=True, duration=0):
        """위반 사항 터미널 알림"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if is_start:
            self.log_message("🚨 위반 감지! 🚨", "ERROR")
            self.log_message(f"시간: {timestamp} 유형: {violation_type}", "ERROR")
            self.log_message(f"상세: {details}", "ERROR")
            self.log_message(f"지속시간: {duration:.1f}초", "ERROR")
            
            self.total_violations += 1
            self.log_violation(violation_type, details)
            
        else:
            self.log_message(f"위반 종료: {violation_type} (총 지속시간: {duration:.1f}초)", "SUCCESS")
            
    def print_warning(self, warning_type, details, duration):
        """경고 사항 터미널 출력"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        progress = min(duration / self.SUSTAINED_TIME, 1.0) * 100
        bar_length = 20
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        self.log_message(f"⚠️  {warning_type}: {details} [{bar}] {progress:.0f}% ({duration:.1f}s)", "WARNING")
        
    def log_violation(self, violation_type, details):
        """위반 사항 로그 기록"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "user": self.authenticated_user,
            "type": violation_type,
            "details": details,
            "exam_duration": int(time.time() - self.exam_start_time) if self.exam_start_time else 0,
            "total_warnings": self.total_warnings,
            "head_warnings": self.head_warnings,
            "gaze_warnings": self.gaze_warnings
        }
        self.violation_log.append(log_entry)
        self.total_violations += 1
        
    def draw_status_info(self, frame, face_count, head_direction, gaze_ratio, x_ratio, y_ratio):
        """화면에 상태 정보 표시"""
        current_time = time.time()
        exam_duration = int(current_time - self.exam_start_time) if self.exam_start_time else 0
        
        # 기본 정보 표시
        cv2.putText(frame, f"User: {self.authenticated_user}", (30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Exam Time: {exam_duration}s", (30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Faces: {face_count}", (30, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Head: {head_direction}", (30, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.gaze_baseline is not None:
            cv2.putText(frame, f"Gaze: {gaze_ratio:.2f} (Base: {self.gaze_baseline:.2f})", (30, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Gaze: {gaze_ratio:.2f} (Calibrating...)", (30, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 경고 횟수 표시 (통합)
        warning_color = (0, 255, 255) if self.total_warnings < self.MAX_WARNINGS else (0, 0, 255)
        cv2.putText(frame, f"Total Warnings: {self.total_warnings}/{self.MAX_WARNINGS}", (30, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)
        
        cv2.putText(frame, f"Head: {self.head_warnings}, Gaze: {self.gaze_warnings}", (30, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 위반 상태 표시
        y_offset = 240
        
        violations = []
        if self.is_multiple_faces_violation:
            violations.append("Multiple Faces")
        if self.is_no_face_violation:
            violations.append("No Face")
        if self.is_head_violation:
            violations.append(f"Head: {head_direction}")
        if self.is_gaze_violation:
            violations.append("Gaze Direction")
        
        if violations:
            cv2.putText(frame, f"VIOLATIONS: {', '.join(violations)}", (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 시험 중단 상태 표시
        if self.exam_terminated:
            # 부정행위 탐지로 인한 중단 강조 표시
            cv2.putText(frame, "CHEATING DETECTED!", (30, y_offset + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, "EXAM TERMINATED", (30, y_offset + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame, f"Reason: {self.termination_reason}", (30, y_offset + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 화면 전체에 경고 테두리 표시
            cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 255), 5)
        
        # 캘리브레이션 상태 표시
        if self.gaze_baseline is None:
            progress = (self.frame_count / self.BASELINE_FRAMES) * 100
            cv2.putText(frame, f"Eye Calibration: {self.frame_count}/{self.BASELINE_FRAMES} ({progress:.0f}%)", 
                       (30, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Look forward and stay still", 
                       (30, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 위반 횟수 표시
        cv2.putText(frame, f"Total Violations: {self.total_violations}", 
                   (frame.shape[1] - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if self.total_violations > 0 else (0, 255, 0), 2)
        
    # =====================================
    # 유틸리티 함수
    # =====================================
        
    def log_message(self, message, level="INFO"):
        """로그 메시지 (Queue 없이 직접)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted)
        
        # 색상 적용
        if level == "ERROR":
            color = "red"
        elif level == "WARNING":
            color = "orange"
        elif level == "SUCCESS":
            color = "green"
        else:
            color = "black"
            
        # 스크롤
        self.log_text.see(tk.END)
        
        # 라인 수 제한
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 500:
            self.log_text.delete('1.0', '100.0')
            
    def run(self):
        """GUI 실행"""
        self.log_message("🤖 AI 시험 감독관 v2.6 (단순화) 시작", "SUCCESS")
        self.log_message("Queue 시스템을 제거하고 직접 GUI 업데이트로 단순화했습니다.")
        self.log_message("원본 감독 로직은 100% 보존되었습니다.")
        self.log_message("부정행위 탐지 및 TTS 기능이 모두 포함되어 있습니다.")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """종료 처리"""
        if self.is_running:
            if messagebox.askokcancel("종료", "시험이 진행 중입니다. 정말 종료하시겠습니까?"):
                self.stop_system()
                self.root.destroy()
        else:
            self.root.destroy()

# =====================================
# TTS 함수
# =====================================

def tts_worker_thread():
    """TTS 전용 재생 루프 (한 번에 하나씩 처리)"""
    last_text = ""
    last_time = 0

    while True:
        text = tts_queue.get()
        now = time.time()

        # 중복 방지 (같은 문장을 3초 이내 반복 금지)
        if text == last_text and now - last_time < 3:
            print(f"[TTS] 중복 생략: {text}")
            continue

        last_text = text
        last_time = now

        try:
            print(f"[TTS] 재생: {text}")
            tts = gTTS(text=text, lang='ko')
            tts_path = "tts_output.mp3"
            tts.save(tts_path)
            subprocess.run(["mpg123", tts_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(tts_path)
        except Exception as e:
            print(f"[TTS 오류] {e}")


def speak_tts(text):
    """TTS 요청을 큐에 추가"""
    if not TTS_AVAILABLE:
        return
    tts_queue.put(text)

# 실행
if __name__ == "__main__":
    try:
        threading.Thread(target=tts_worker_thread, daemon=True).start()
        app = SimpleExamSupervisor()
        app.run()
    except Exception as e:
        print(f"오류: {e}")
        print("필요한 라이브러리를 설치해주세요:")
        print("pip install opencv-python mediapipe face-recognition pillow gtts")
        sys.exit(1)
