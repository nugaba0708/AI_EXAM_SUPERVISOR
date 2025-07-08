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
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from PIL import Image, ImageTk
import queue

# TTS import (선택적)
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# CustomTkinter import (try-except로 선택적 로드)
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
except ImportError:
    CTK_AVAILABLE = False

class AIExamSupervisorIntegrated:
    """원본 AI 시험 감독관 클래스 - 키보드 입력 제거, GUI 전용"""
    
    def __init__(self, config_path="config.json"):
        """통합 AI 시험 감독관 초기화"""
        
        # GUI 콜백 설정
        self.gui_callback = None
        self.gui_frame_callback = None
        
        # 설정 로드
        self.load_config(config_path)
        
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False, 
            max_num_faces=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 얼굴 랜드마크 포인트 정의
        self.NOSE_TIP = 1
        self.LEFT_EYE_LEFT = 33
        self.RIGHT_EYE_RIGHT = 263
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291
        self.CHIN = 18
        
        # 눈 영역 포인트 (아이트래킹용)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 387, 385, 263, 373, 380]
        
        # 눈 깜빡임 감지용 (EAR 계산)
        self.LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
        
        # 깜빡임 감지 설정
        self.BLINK_THRESHOLD = 0.21
        self.BLINK_REQUIRED = 2
        self.BLINK_DETECTION_DURATION = 6
        self.MAX_EAR_HISTORY = 100
        
        # 시스템 상태
        self.system_phase = "IDLE"  # IDLE -> IDENTITY_CHECK -> EXAM_MONITORING
        self.authenticated_user = None
        self.exam_start_time = None
        self.exam_terminated = False
        self.termination_reason = ""
        
        # 위반 상태 초기화
        self.reset_violation_states()
        
        # 경고 시스템 초기화
        self.MAX_WARNINGS = 5
        self.total_warnings = 0
        self.head_warnings = 0
        self.gaze_warnings = 0
        
        # 얼굴 추적 변수들
        self.last_face_landmarks = None
        self.face_lost_time = 0
        
        # 아이트래킹 변수들
        self.gaze_baseline = None
        self.frame_count = 0
        self.baseline_sum = 0
        self.last_gaze_violation_time = 0
        
        # 로깅 시스템
        self.violation_log = []
        self.total_violations = 0
        self.identity_attempts = 0
        
        # 데이터셋 경로 확인
        self.ensure_dataset_exists()
        
        # 신원 확인 관련 상태
        self.identity_active = False
        self.blink_detection_active = False
        
        print("🔒 AI 시험 감독관 시스템 v2.5 초기화 완료")
    
    def set_gui_callback(self, callback, frame_callback=None):
        """GUI 콜백 설정"""
        self.gui_callback = callback
        self.gui_frame_callback = frame_callback
    
    def log_message(self, message, level="INFO"):
        """로그 메시지 출력"""
        if self.gui_callback:
            self.gui_callback('log', message, level)
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def update_gui_status(self, **kwargs):
        """GUI 상태 업데이트"""
        if self.gui_callback:
            self.gui_callback('status', **kwargs)
    
    def calculate_ear(self, landmarks, eye_indices):
        """EAR (Eye Aspect Ratio) 계산 함수"""
        try:
            left = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
            right = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
            top = (np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) +
                   np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])) / 2
            bottom = (np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]) +
                      np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])) / 2
            
            horizontal = np.linalg.norm(left - right)
            vertical = np.linalg.norm(top - bottom)
            
            if horizontal == 0:
                return 0.0
            
            return vertical / horizontal
        except Exception as e:
            return 0.0
    
    def load_config(self, config_path):
        """설정 파일 로드"""
        default_config = {
            "camera": {
                "index": 0,
                "width": 640,
                "height": 480,
                "fps": 20,
                "mirror": True
            },
            "detection": {
                "x_threshold": 0.15,
                "y_threshold": 0.5,
                "sustained_time": 2.0,
                "gaze_margin": 0.5,
                "face_lost_threshold": 1.0
            },
            "identity": {
                "dataset_path": "./dataset",
                "tolerance": 0.4,
                "max_attempts": 5,
                "blink_threshold": 0.21,
                "blink_required": 2,
                "blink_detection_duration": 6
            },
            "system": {
                "print_interval": 1.0,
                "baseline_frames": 30,
                "gaze_debug_mode": False,
                "save_video": True,
                "log_path": "./logs",
                "max_warnings": 5
            }
        }
        
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
            except Exception as e:
                config = default_config
        else:
            config = default_config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 설정값 적용
        self.config = config
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
        
        self.PRINT_INTERVAL = config["system"]["print_interval"]
        self.BASELINE_FRAMES = config["system"]["baseline_frames"]
        self.GAZE_DEBUG_MODE = config["system"]["gaze_debug_mode"]
        self.SAVE_VIDEO = config["system"]["save_video"]
        self.LOG_PATH = config["system"]["log_path"]
        self.MAX_WARNINGS = config["system"]["max_warnings"]
    
    def ensure_dataset_exists(self):
        """데이터셋 폴더 존재 확인 및 생성"""
        Path(self.DATASET_PATH).mkdir(exist_ok=True)
        Path(self.LOG_PATH).mkdir(exist_ok=True)
        
        dataset_files = list(Path(self.DATASET_PATH).glob("*.jpg")) + \
                       list(Path(self.DATASET_PATH).glob("*.jpeg")) + \
                       list(Path(self.DATASET_PATH).glob("*.png"))
        
        if not dataset_files:
            self.log_message(f"⚠️  데이터셋 폴더가 비어있습니다: {self.DATASET_PATH}", "WARNING")
    
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
    
    def terminate_exam(self, reason):
        """시험 중단"""
        self.exam_terminated = True
        self.termination_reason = reason
        
        self.log_message("🚨 부정행위 탐지! 시험 즉시 중단! 🚨", "ERROR")
        self.log_message(f"사유: {reason}", "ERROR")
        
        speak_tts("심각한 부정행위가 탐지되어 시험을 중단합니다.")
        
        self.update_gui_status(exam_terminated=True, termination_reason=reason)
        self.log_violation("부정행위-시험중단", reason)
    
    def issue_warning(self, warning_type, details):
        """경고 발급"""
        if warning_type == "고개 방향":
            self.head_warnings += 1
        elif warning_type == "시선 이탈":
            self.gaze_warnings += 1
        
        self.total_warnings += 1
        remaining = self.MAX_WARNINGS - self.total_warnings
        
        speak_tts(f"{warning_type}부정행위가 탐지되었습니다.")
        
        self.log_message(f"⚠️  경고 {self.total_warnings}/{self.MAX_WARNINGS} - {warning_type}", "WARNING")
        self.log_message(f"상세: {details}", "WARNING")
        
        self.update_gui_status(
            total_warnings=self.total_warnings,
            head_warnings=self.head_warnings,
            gaze_warnings=self.gaze_warnings,
            max_warnings=self.MAX_WARNINGS
        )
        
        if self.total_warnings >= self.MAX_WARNINGS:
            self.log_message(f"🚨 총 {self.MAX_WARNINGS}회 경고 누적! 부정행위로 판정됩니다.", "ERROR")
            self.terminate_exam(f"경고 {self.MAX_WARNINGS}회 누적 (고개: {self.head_warnings}회, 시선: {self.gaze_warnings}회)")
            return True
        
        self.log_violation(f"경고-{warning_type}", details)
        return False
    
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
            except Exception as e:
                continue
        return None
    
    def compare_with_dataset(self, captured_image_path):
        """데이터셋과 얼굴 비교"""
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
            
            results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=self.FACE_TOLERANCE)
            distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            
            self.log_message("📏 거리 값 (각 등록된 얼굴과의 거리):")
            for name, dist in zip(known_names, distances):
                self.log_message(f"   📊 {name}: {dist:.4f}")

            if True in results:
                idx = results.index(True)
                matched_name = known_names[idx]
                distance = distances[idx]
               
                speak_tts("신원 확인이 완료되었습니다.")
                self.log_message(f"✅ [매칭 성공] {matched_name} (거리: {distance:.4f})", "SUCCESS")
                
                return matched_name
            else:
                min_distance = min(distances)
                self.log_message(f"❌ [매칭 실패] 최소 거리: {min_distance:.4f} (임계값: {self.FACE_TOLERANCE})", "ERROR")
                return None
                
        except Exception as e:
            self.log_message(f"❌ 얼굴 인식 처리 오류: {e}", "ERROR")
            return None
    
    def start_identity_check(self):
        """신원 확인 시작 (GUI에서 호출)"""
        self.identity_active = True
        self.identity_attempts = 0
        self.system_phase = "IDENTITY_CHECK"
        
        self.log_message("🔍 신원 확인 단계 시작", "SUCCESS")
        self.log_message("👁 눈 깜빡임을 통한 실제 사람 확인 + 얼굴 인식")
        
        self.update_gui_status(
            phase="신원 확인 중",
            identity_attempts=0,
            max_attempts=self.MAX_IDENTITY_ATTEMPTS
        )
        
        return True
    
    def start_blink_detection(self):
        """깜빡임 감지 시작 (GUI에서 호출)"""
        if not self.identity_active:
            return False
        
        self.identity_attempts += 1
        self.blink_detection_active = True
        
        self.log_message(f"▶ {self.identity_attempts}번째 시도: 눈 깜빡임 감지 시작")
        self.log_message(f"👁 {self.BLINK_DETECTION_DURATION}초 동안 최소 {self.BLINK_REQUIRED}회 깜빡여 주세요.")
        
        self.update_gui_status(
            identity_attempts=self.identity_attempts,
            blink_detection_active=True,
            blink_count=0
        )
        
        return True
    
    def process_identity_frame(self, frame):
        """신원 확인 프레임 처리 (GUI용)"""
        if not self.identity_active:
            return frame, None
        
        if not self.blink_detection_active:
            # 대기 화면 표시
            cv2.putText(frame, f"Identity Verification ({self.identity_attempts + 1}/{self.MAX_IDENTITY_ATTEMPTS})", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Click 'Start Blink Detection' button", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return frame, None
        
        # 깜빡임 감지 진행 중
        if not hasattr(self, 'blink_start_time'):
            self.blink_start_time = time.time()
            self.blink_count = 0
            self.blink_flag = False
            self.ear_history = []
            self.ear_log_data = []
        
        # MediaPipe 처리
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            results = face_mesh.process(rgb)
            ear = None
            frame_to_save = None
            
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
                        self.log_message(f"👁 깜빡임 감지됨! 총 {self.blink_count}회")
                        
                        self.update_gui_status(blink_count=self.blink_count)
                        
                    elif ear >= self.BLINK_THRESHOLD:
                        self.blink_flag = False
                    
                    frame_to_save = frame.copy()
            
            # EAR 데이터 기록
            if ear is not None:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.ear_log_data.append({"timestamp": timestamp, "ear": round(ear, 4)})
                
                self.ear_history.append(ear)
                if len(self.ear_history) > self.MAX_EAR_HISTORY:
                    self.ear_history.pop(0)
                
                # EAR 그래프 그리기
                self.draw_ear_graph(frame, self.ear_history, ear)
            
            # 상태 정보 표시
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
            
            # 시간 종료 확인
            if elapsed_time > self.BLINK_DETECTION_DURATION:
                return self.complete_blink_detection(frame_to_save)
        
        return frame, None
    
    def complete_blink_detection(self, frame_to_save):
        """깜빡임 감지 완료 처리"""
        self.blink_detection_active = False
        
        self.update_gui_status(blink_detection_active=False)
        
        # 깜빡임 결과 평가
        if self.blink_count >= self.BLINK_REQUIRED and frame_to_save is not None:
            self.log_message(f"✅ 실제 사람으로 판별됨! ({self.blink_count}회 깜빡임 감지)", "SUCCESS")
            self.log_message("🔍 얼굴 인식 처리 중...")
            
            # 캡처 이미지 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"identity_check_{timestamp}_{self.identity_attempts}.png"
            cv2.imwrite(filename, frame_to_save)
            
            # 얼굴 비교 수행
            matched_name = self.compare_with_dataset(filename)
            
            if matched_name:
                self.authenticated_user = matched_name
                self.identity_active = False
                
                self.log_message(f"🎉 인증 성공! {matched_name}님, 환영합니다.", "SUCCESS")
                
                self.update_gui_status(
                    authenticated_user=matched_name,
                    identity_success=True,
                    phase="신원 확인 완료"
                )
                
                try:
                    os.remove(filename)
                except:
                    pass
                
                return frame_to_save, "SUCCESS"
            else:
                self.log_message(f"❌ 얼굴 인증 실패 ({self.identity_attempts}/{self.MAX_IDENTITY_ATTEMPTS})", "ERROR")
                
                try:
                    os.remove(filename)
                except:
                    pass
                
                if self.identity_attempts >= self.MAX_IDENTITY_ATTEMPTS:
                    self.identity_active = False
                    self.log_message("❌ 신원 확인 실패: 최대 시도 횟수 초과", "ERROR")
                    return frame_to_save, "FAILED"
                
                return frame_to_save, "RETRY"
        else:
            self.log_message(f"❌ 사진으로 판별됨 (눈 깜빡임 {self.blink_count}회 < {self.BLINK_REQUIRED}회)", "ERROR")
            
            if self.identity_attempts >= self.MAX_IDENTITY_ATTEMPTS:
                self.identity_active = False
                self.log_message("❌ 신원 확인 실패: 최대 시도 횟수 초과", "ERROR")
                return frame_to_save, "FAILED"
            
            return frame_to_save, "RETRY"
    
    def draw_ear_graph(self, frame, ear_history, current_ear):
        """EAR 그래프를 화면에 그리기"""
        if len(ear_history) < 2:
            return
        
        graph_width = min(self.MAX_EAR_HISTORY, len(ear_history))
        graph_x = frame.shape[1] - graph_width - 10
        graph_y_base = 200
        
        for i in range(1, len(ear_history)):
            if i >= graph_width:
                break
            
            pt1 = (graph_x + i - 1, graph_y_base - int(ear_history[i - 1] * 100))
            pt2 = (graph_x + i, graph_y_base - int(ear_history[i] * 100))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        threshold_y = graph_y_base - int(self.BLINK_THRESHOLD * 100)
        cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_width, threshold_y), (0, 0, 255), 2)
        
        cv2.putText(frame, "EAR Graph", (graph_x, graph_y_base - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {self.BLINK_THRESHOLD}", (graph_x, graph_y_base + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def start_exam_monitoring(self):
        """시험 감독 시작"""
        if not self.authenticated_user:
            self.log_message("신원 확인을 먼저 완료해주세요.", "ERROR")
            return False
        
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
        
        self.update_gui_status(
            phase="시험 감독 중",
            exam_start_time=self.exam_start_time
        )
        
        speak_tts("시험 감독이 시작되었습니다.")
        
        return True
    
    def get_landmarks_coords(self, face_landmarks, image_w, image_h):
        """MediaPipe 랜드마크를 픽셀 좌표로 변환"""
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
        
        face_width = abs(right_eye_right[0] - left_eye_left[0])
        face_center_x = (left_eye_left[0] + right_eye_right[0]) / 2
        
        offset_x = nose_tip[0] - face_center_x
        
        eye_center_y = (left_eye_left[1] + right_eye_right[1]) / 2
        offset_y = nose_tip[1] - eye_center_y
        
        if face_width == 0:
            return "Forward", 0, 0
        
        x_ratio = offset_x / face_width
        y_ratio = offset_y / face_width
        
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
            eye_region = np.array([(int(landmarks[i][0] * w), int(landmarks[i][1] * h)) 
                                  for i in eye_indices], np.int32)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [eye_region], 255)
            
            eye = cv2.bitwise_and(gray, gray, mask=mask)
            
            min_x = np.min(eye_region[:, 0])
            max_x = np.max(eye_region[:, 0])
            min_y = np.min(eye_region[:, 1])
            max_y = np.max(eye_region[:, 1])
            
            if min_x >= max_x or min_y >= max_y or min_x < 0 or min_y < 0 or max_x >= w or max_y >= h:
                return 1.0
            
            gray_eye = eye[min_y:max_y, min_x:max_x]
            
            if gray_eye.size == 0:
                return 1.0
            
            _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
            
            th_h, th_w = threshold_eye.shape
            
            if th_w < 2:
                return 1.0
            
            left_white = cv2.countNonZero(threshold_eye[:, 0:int(th_w / 2)])
            right_white = cv2.countNonZero(threshold_eye[:, int(th_w / 2):])
            
            if left_white == 0 or right_white == 0:
                return 1.0
            else:
                return left_white / right_white
                
        except Exception as e:
            return 1.0
    
    def process_monitoring_frame(self, frame):
        """시험 감독 프레임 처리"""
        if self.system_phase != "EXAM_MONITORING" or self.exam_terminated:
            return frame
        
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
            
            best_face = results.multi_face_landmarks[0]
            current_landmarks = self.get_landmarks_coords(best_face, img_w, img_h)
            
            head_direction, x_ratio, y_ratio = self.get_head_direction(current_landmarks, img_w, img_h)
            
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
                    speak_tts("시선 기준 설정이 완료되었습니다.")
                    
                    self.update_gui_status(
                        gaze_baseline=self.gaze_baseline,
                        calibration_complete=True
                    )
                    
                gaze_ratio = current_gaze
            else:
                gaze_ratio = current_gaze
        
        # GUI 상태 업데이트
        current_time = time.time()
        exam_duration = int(current_time - self.exam_start_time) if self.exam_start_time else 0
        
        self.update_gui_status(
            face_count=face_count,
            head_direction=head_direction,
            gaze_ratio=gaze_ratio,
            exam_duration=exam_duration,
            total_violations=self.total_violations,
            frame_count=self.frame_count if self.gaze_baseline is None else self.BASELINE_FRAMES
        )
        
        # 위반 상태 업데이트
        self.update_violation_states(face_count, head_direction, gaze_ratio)
        
        # 화면에 정보 표시
        self.draw_status_info(frame, face_count, head_direction, gaze_ratio, x_ratio, y_ratio)
        
        return frame
    
    def update_violation_states(self, face_count, head_direction, gaze_ratio):
        """위반 상태 업데이트"""
        current_time = time.time()
        
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
                self.terminate_exam(f"다중 인물 감지 ({face_count}명)")
                return
        
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
                self.terminate_exam("화면 이탈 (얼굴 감지 불가)")
                return
        
        # 3. 고개 방향 감지 - 경고 후 중단
        current_head_abnormal = head_direction in ["Left", "Right", "Down"]
        if current_head_abnormal != self.is_head_abnormal:
            self.is_head_abnormal = current_head_abnormal
            self.head_abnormal_start_time = current_time
            self.is_head_violation = False
        
        if self.is_head_abnormal:
            duration = current_time - self.head_abnormal_start_time
            if duration >= self.SUSTAINED_TIME and not self.is_head_violation:
                self.is_head_violation = True
                is_terminated = self.issue_warning("고개 방향", f"방향: {head_direction}")
                if is_terminated:
                    return
        
        # 4. 시선 이탈 감지 - 경고 후 중단
        if self.gaze_baseline is not None and gaze_ratio != 1:
            current_gaze_abnormal = (gaze_ratio < self.gaze_baseline - self.GAZE_MARGIN or 
                                   gaze_ratio > self.gaze_baseline + self.GAZE_MARGIN)
            
            if current_gaze_abnormal != self.is_gaze_abnormal:
                self.is_gaze_abnormal = current_gaze_abnormal
                self.gaze_abnormal_start_time = current_time
                self.is_gaze_violation = False
            
            if self.is_gaze_abnormal:
                duration = current_time - self.gaze_abnormal_start_time
                if duration >= self.SUSTAINED_TIME and not self.is_gaze_violation:
                    self.is_gaze_violation = True
                    direction = "왼쪽" if gaze_ratio < self.gaze_baseline else "오른쪽"
                    is_terminated = self.issue_warning("시선 이탈", f"{direction} 방향으로 시선 이탈")
                    if is_terminated:
                        return
    
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
    
    def draw_status_info(self, frame, face_count, head_direction, gaze_ratio, x_ratio, y_ratio):
        """상태 정보를 화면에 표시"""
        current_time = time.time()
        exam_duration = int(current_time - self.exam_start_time) if self.exam_start_time else 0
        
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
        
        warning_color = (0, 255, 255) if self.total_warnings < self.MAX_WARNINGS else (0, 0, 255)
        cv2.putText(frame, f"Total Warnings: {self.total_warnings}/{self.MAX_WARNINGS}", (30, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)
        
        if self.exam_terminated:
            cv2.putText(frame, "CHEATING DETECTED!", (30, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, "EXAM TERMINATED", (30, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 255), 5)
        
        if self.gaze_baseline is None:
            progress = (self.frame_count / self.BASELINE_FRAMES) * 100
            cv2.putText(frame, f"Eye Calibration: {self.frame_count}/{self.BASELINE_FRAMES} ({progress:.0f}%)", 
                       (30, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def get_exam_duration_string(self):
        """시험 시간을 문자열로 반환"""
        if not self.exam_start_time:
            return "00:00:00"
        
        elapsed = int(time.time() - self.exam_start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class AIExamSupervisorGUI:
    """GUI 메인 클래스 - v2.4 스타일 기반"""
    
    def __init__(self):
        self.setup_gui()
        self.supervisor = AIExamSupervisorIntegrated()
        self.supervisor.set_gui_callback(self.gui_callback)
        
        self.camera_thread = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.log_queue = queue.Queue()
        self.status_queue = queue.Queue()
        
        # 신원 확인 관련
        self.identity_window = None
        self.identity_phase = False
        
        self.update_gui()
    
    def setup_gui(self):
        """GUI 설정"""
        if CTK_AVAILABLE:
            self.root = ctk.CTk()
            self.root.title("🤖 AI 시험 감독관 시스템 v2.5")
            self.root.geometry("1400x900")
            self.root.minsize(1200, 800)
        else:
            self.root = tk.Tk()
            self.root.title("🤖 AI 시험 감독관 시스템 v2.5")
            self.root.geometry("1400x900")
            self.root.minsize(1200, 800)
        
        self.create_main_layout()
        self.create_menu()
    
    def create_menu(self):
        """메뉴바 생성"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="설정 불러오기", command=self.load_config)
        file_menu.add_command(label="설정 저장", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.on_closing)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도구", menu=tools_menu)
        tools_menu.add_command(label="로그 폴더 열기", command=self.open_log_folder)
        tools_menu.add_command(label="데이터셋 폴더 열기", command=self.open_dataset_folder)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="사용법", command=self.show_help)
        help_menu.add_command(label="정보", command=self.show_about)
    
    def create_main_layout(self):
        """메인 레이아웃 생성"""
        main_container = tk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        if CTK_AVAILABLE:
            self.left_frame = ctk.CTkFrame(main_container)
            self.left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
            self.right_frame = ctk.CTkFrame(main_container)
            self.right_frame.pack(side="right", fill="y", padx=(10, 0))
        else:
            self.left_frame = ttk.Frame(main_container)
            self.left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
            self.right_frame = ttk.Frame(main_container, width=400)
            self.right_frame.pack(side="right", fill="y", padx=(10, 0))
            self.right_frame.pack_propagate(False)
        
        self.create_camera_panel()
        self.create_status_panel()
        self.create_control_panel()
        self.create_log_panel()
    
    def create_camera_panel(self):
        """카메라 패널 생성"""
        if CTK_AVAILABLE:
            camera_frame = ctk.CTkFrame(self.left_frame)
            camera_frame.pack(fill="both", expand=True, pady=(0, 10))
            title = ctk.CTkLabel(camera_frame, text="📹 실시간 카메라", font=("Arial", 18, "bold"))
            title.pack(pady=15)
        else:
            camera_frame = ttk.LabelFrame(self.left_frame, text="📹 실시간 카메라", padding=15)
            camera_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.camera_container = tk.Frame(camera_frame, bg="black", relief="sunken", bd=2)
        self.camera_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.camera_label = tk.Label(self.camera_container, 
                                   text="카메라 연결 대기 중...\n\n'시험 시작' 버튼을 눌러주세요", 
                                   bg="black", fg="white", font=("Arial", 14))
        self.camera_label.pack(expand=True)
    
    def create_status_panel(self):
        """상태 패널 생성"""
        if CTK_AVAILABLE:
            status_frame = ctk.CTkFrame(self.left_frame)
            status_frame.pack(fill="x", pady=(0, 10))
            title = ctk.CTkLabel(status_frame, text="📊 시스템 상태", font=("Arial", 16, "bold"))
            title.pack(pady=10)
        else:
            status_frame = ttk.LabelFrame(self.left_frame, text="📊 시스템 상태", padding=10)
            status_frame.pack(fill="x", pady=(0, 10))
        
        info_container = tk.Frame(status_frame)
        info_container.pack(fill="x", padx=15, pady=10)
        
        left_col = tk.Frame(info_container)
        left_col.pack(side="left", fill="both", expand=True)
        
        right_col = tk.Frame(info_container)
        right_col.pack(side="right", fill="both", expand=True)
        
        status_items_left = [
            ("👤 사용자:", "user_label", "미인증", "red"),
            ("🔄 단계:", "phase_label", "대기 중", "blue"),
            ("⏰ 시험 시간:", "time_label", "00:00:00", "black"),
            ("👥 얼굴 감지:", "face_label", "0명", "black")
        ]
        
        status_items_right = [
            ("🎯 고개 방향:", "head_label", "정면", "green"),
            ("👁 시선 상태:", "gaze_label", "캘리브레이션 중", "blue"),
            ("⚠️ 경고 횟수:", "warning_label", "0/5", "green"),
            ("🚨 위반 횟수:", "violation_label", "0", "green")
        ]
        
        for label_text, attr_name, default_text, color in status_items_left:
            self.create_status_item(left_col, label_text, attr_name, default_text, color)
        
        for label_text, attr_name, default_text, color in status_items_right:
            self.create_status_item(right_col, label_text, attr_name, default_text, color)
    
    def create_status_item(self, parent, label_text, attr_name, default_text, color):
        """상태 아이템 생성"""
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=3)
        
        tk.Label(frame, text=label_text, width=15, anchor="w", 
                font=("Arial", 10)).pack(side="left")
        
        label = tk.Label(frame, text=default_text, fg=color, anchor="w", 
                        font=("Arial", 10, "bold"))
        label.pack(side="left")
        setattr(self, attr_name, label)
    
    def create_control_panel(self):
        """제어 패널 생성"""
        if CTK_AVAILABLE:
            control_frame = ctk.CTkFrame(self.right_frame)
            control_frame.pack(fill="x", pady=(0, 15))
            title = ctk.CTkLabel(control_frame, text="🎛️ 시스템 제어", font=("Arial", 16, "bold"))
            title.pack(pady=15)
            
            button_frame = ctk.CTkFrame(control_frame)
            button_frame.pack(fill="x", padx=15, pady=10)
            
            self.start_button = ctk.CTkButton(button_frame, text="▶️ 시험 시작", 
                                            command=self.start_exam, width=200, height=45,
                                            font=("Arial", 14, "bold"), fg_color="green")
            self.start_button.pack(pady=8)
            
            self.stop_button = ctk.CTkButton(button_frame, text="⏹️ 시험 중지", 
                                           command=self.stop_exam, width=200, height=40, 
                                           state="disabled", fg_color="red")
            self.stop_button.pack(pady=5)
            
            self.identity_button = ctk.CTkButton(button_frame, text="👤 신원 확인", 
                                               command=self.start_identity_check, width=200, height=40)
            self.identity_button.pack(pady=5)
            
            self.report_button = ctk.CTkButton(button_frame, text="📄 보고서 보기", 
                                             command=self.show_report, width=200, height=40)
            self.report_button.pack(pady=5)
        else:
            control_frame = ttk.LabelFrame(self.right_frame, text="🎛️ 시스템 제어", padding=15)
            control_frame.pack(fill="x", pady=(0, 15))
            
            self.start_button = tk.Button(control_frame, text="▶️ 시험 시작", 
                                        command=self.start_exam, width=22, height=2, 
                                        bg="#28a745", fg="white", font=("Arial", 12, "bold"))
            self.start_button.pack(pady=8)
            
            self.stop_button = tk.Button(control_frame, text="⏹️ 시험 중지", 
                                       command=self.stop_exam, width=22, height=2, 
                                       bg="#dc3545", fg="white", font=("Arial", 11),
                                       state="disabled")
            self.stop_button.pack(pady=5)
            
            self.identity_button = tk.Button(control_frame, text="👤 신원 확인", 
                                           command=self.start_identity_check, width=22, height=2,
                                           bg="#17a2b8", fg="white", font=("Arial", 11))
            self.identity_button.pack(pady=5)
            
            self.report_button = tk.Button(control_frame, text="📄 보고서 보기", 
                                         command=self.show_report, width=22, height=2,
                                         bg="#6f42c1", fg="white", font=("Arial", 11))
            self.report_button.pack(pady=5)
    
    def create_log_panel(self):
        """로그 패널 생성"""
        if CTK_AVAILABLE:
            log_frame = ctk.CTkFrame(self.right_frame)
            log_frame.pack(fill="both", expand=True)
            title = ctk.CTkLabel(log_frame, text="📝 실시간 로그", font=("Arial", 14, "bold"))
            title.pack(pady=10)
        else:
            log_frame = ttk.LabelFrame(self.right_frame, text="📝 실시간 로그", padding=10)
            log_frame.pack(fill="both", expand=True)
        
        log_container = tk.Frame(log_frame)
        log_container.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_container, width=45, height=20, 
                                                wrap=tk.WORD, font=("Consolas", 9),
                                                bg="#f8f9fa", relief="sunken", bd=2)
        self.log_text.pack(fill="both", expand=True)
    
    def gui_callback(self, callback_type, *args, **kwargs):
        """백엔드로부터의 콜백 처리"""
        if callback_type == 'log':
            message = args[0]
            level = args[1] if len(args) > 1 else "INFO"
            self.log_queue.put((message, level))
        elif callback_type == 'status':
            self.status_queue.put(kwargs)
    
    def log_message(self, message, level="INFO"):
        """로그 메시지 추가"""
        self.log_queue.put((message, level))
    
    def update_log_display(self):
        """로그 표시 업데이트"""
        try:
            while True:
                message, level = self.log_queue.get_nowait()
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {message}\n"
                
                self.log_text.insert(tk.END, formatted_message)
                
                # 색상 적용
                start_line = self.log_text.index(tk.END + "-2l linestart")
                end_line = self.log_text.index(tk.END + "-1l lineend")
                
                if level == "ERROR":
                    color = "red"
                elif level == "WARNING":
                    color = "orange"
                elif level == "SUCCESS":
                    color = "green"
                else:
                    color = "black"
                
                self.log_text.tag_add(color, start_line, end_line)
                self.log_text.tag_config(color, foreground=color)
                self.log_text.see(tk.END)
                
                lines = int(self.log_text.index('end-1c').split('.')[0])
                if lines > 1000:
                    self.log_text.delete('1.0', '100.0')
        except queue.Empty:
            pass
    
    def update_status_display(self):
        """상태 정보 업데이트"""
        try:
            while True:
                status_data = self.status_queue.get_nowait()
                
                if 'authenticated_user' in status_data:
                    user = status_data['authenticated_user']
                    if user:
                        self.user_label.configure(text=user, fg="green")
                    else:
                        self.user_label.configure(text="미인증", fg="red")
                
                if 'phase' in status_data:
                    phase = status_data['phase']
                    color = "blue" if "진행" in phase else "black"
                    self.phase_label.configure(text=phase, fg=color)
                
                if 'exam_duration' in status_data:
                    duration = status_data['exam_duration']
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    self.time_label.configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
                
                if 'face_count' in status_data:
                    count = status_data['face_count']
                    if count == 0:
                        self.face_label.configure(text="얼굴 없음", fg="red")
                    elif count == 1:
                        self.face_label.configure(text="1명", fg="green")
                    else:
                        self.face_label.configure(text=f"{count}명", fg="red")
                
                if 'head_direction' in status_data:
                    direction = status_data['head_direction']
                    if direction == "Forward":
                        self.head_label.configure(text="정면", fg="green")
                    elif direction == "No Face":
                        self.head_label.configure(text="감지 안됨", fg="red")
                    else:
                        self.head_label.configure(text=direction, fg="orange")
                
                if 'gaze_baseline' in status_data:
                    if status_data.get('calibration_complete'):
                        self.gaze_label.configure(text="캘리브레이션 완료", fg="green")
                elif 'frame_count' in status_data:
                    frame_count = status_data['frame_count']
                    baseline_frames = 30
                    if frame_count < baseline_frames:
                        progress = (frame_count / baseline_frames) * 100
                        self.gaze_label.configure(text=f"캘리브레이션 중 {progress:.0f}%", fg="blue")
                
                if 'total_warnings' in status_data:
                    warnings = status_data['total_warnings']
                    max_warnings = status_data.get('max_warnings', 5)
                    self.warning_label.configure(text=f"{warnings}/{max_warnings}")
                    if warnings >= max_warnings:
                        self.warning_label.configure(fg="red")
                    elif warnings > 0:
                        self.warning_label.configure(fg="orange")
                    else:
                        self.warning_label.configure(fg="green")
                
                if 'total_violations' in status_data:
                    violations = status_data['total_violations']
                    self.violation_label.configure(text=str(violations))
                    if violations > 0:
                        self.violation_label.configure(fg="red")
                    else:
                        self.violation_label.configure(fg="green")
                        
        except queue.Empty:
            pass
    
    def update_camera_display(self):
        """카메라 화면 업데이트"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.camera_container.update_idletasks()
                container_width = self.camera_container.winfo_width()
                container_height = self.camera_container.winfo_height()
                
                if container_width < 100 or container_height < 100:
                    return
                
                display_width = max(320, container_width - 20)
                display_height = max(240, container_height - 20)
                
                # 비율 유지하면서 리사이즈
                h, w = frame_rgb.shape[:2]
                aspect_ratio = w / h
                target_ratio = display_width / display_height
                
                if aspect_ratio > target_ratio:
                    new_width = display_width
                    new_height = int(display_width / aspect_ratio)
                else:
                    new_height = display_height
                    new_width = int(display_height * aspect_ratio)
                
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                result = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                
                y_offset = (display_height - new_height) // 2
                x_offset = (display_width - new_width) // 2
                
                result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame_resized
                
                image = Image.fromarray(result)
                photo = ImageTk.PhotoImage(image)
                
                self.camera_label.configure(image=photo, text="")
                self.camera_label.image = photo
                
        except queue.Empty:
            pass
        except Exception as e:
            pass
    
    def start_exam(self):
        """시험 시작"""
        if self.is_running:
            return
        
        # 카메라 연결
        cap = self.supervisor.find_camera()
        if cap is None:
            messagebox.showerror("오류", "카메라를 찾을 수 없습니다!")
            return
        
        self.is_running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # 카메라 스레드 시작
        self.camera_thread = threading.Thread(target=self.camera_loop, args=(cap,), daemon=True)
        self.camera_thread.start()
        
        self.log_message("시스템이 시작되었습니다. 신원 확인을 진행하세요.", "SUCCESS")
    
    def camera_loop(self, cap):
        """카메라 루프"""
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if self.supervisor.MIRROR_CAMERA:
                    frame = cv2.flip(frame, 1)
                
                # 프레임 처리
                if self.supervisor.system_phase == "IDENTITY_CHECK":
                    processed_frame, result = self.supervisor.process_identity_frame(frame)
                    if result:
                        if result == "SUCCESS":
                            self.log_message("신원 확인이 완료되었습니다! 시험 감독을 시작할 수 있습니다.", "SUCCESS")
                        elif result == "FAILED":
                            self.log_message("신원 확인에 실패했습니다.", "ERROR")
                        # RETRY의 경우 계속 진행
                elif self.supervisor.system_phase == "EXAM_MONITORING":
                    processed_frame = self.supervisor.process_monitoring_frame(frame)
                else:
                    processed_frame = frame
                
                # GUI에 프레임 전송
                try:
                    self.frame_queue.put_nowait(processed_frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(processed_frame)
                    except queue.Empty:
                        pass
                
                time.sleep(0.03)  # 30 FPS
                
        except Exception as e:
            self.log_message(f"카메라 오류: {e}", "ERROR")
        finally:
            cap.release()
    
    def stop_exam(self):
        """시험 중지"""
        self.is_running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        
        self.supervisor.exam_terminated = True
        
        self.log_message("시스템이 중지되었습니다.", "WARNING")
    
    def start_identity_check(self):
        """신원 확인 시작"""
        if not self.is_running:
            messagebox.showwarning("경고", "시스템을 먼저 시작하세요!")
            return
        
        # 데이터셋 확인
        dataset_path = Path("./dataset")
        dataset_files = list(dataset_path.glob("*.jpg")) + \
                       list(dataset_path.glob("*.jpeg")) + \
                       list(dataset_path.glob("*.png"))
        
        if not dataset_files:
            messagebox.showwarning("경고", 
                                 "데이터셋 폴더에 인증용 얼굴 이미지가 없습니다.\n"
                                 "'./dataset' 폴더에 이미지를 추가한 후 다시 시도하세요.")
            return
        
        self.show_identity_window()
    
    def show_identity_window(self):
        """신원 확인 창 표시"""
        if self.identity_window and self.identity_window.winfo_exists():
            self.identity_window.lift()
            return
        
        self.identity_window = tk.Toplevel(self.root)
        self.identity_window.title("👤 신원 확인 시스템")
        self.identity_window.geometry("600x500")
        self.identity_window.resizable(False, False)
        self.identity_window.transient(self.root)
        
        # 안전한 grab_set
        self.identity_window.update_idletasks()
        try:
            self.identity_window.grab_set()
        except tk.TclError:
            pass
        
        # 헤더
        header_frame = tk.Frame(self.identity_window, bg="#2c3e50", height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="🔍 신원 확인 시스템", 
                              font=("Arial", 20, "bold"), fg="white", bg="#2c3e50")
        header_label.pack(expand=True)
        
        # 안내 메시지
        info_frame = tk.Frame(self.identity_window, bg="#ecf0f1", height=100)
        info_frame.pack(fill="x")
        info_frame.pack_propagate(False)
        
        info_text = """👁 눈 깜빡임을 통한 실제 사람 확인
🔍 등록된 얼굴과의 매칭 검증
📸 6초 동안 최소 2회 이상 깜빡여 주세요"""
        
        tk.Label(info_frame, text=info_text, font=("Arial", 12), 
                bg="#ecf0f1", justify="left").pack(expand=True, pady=10)
        
        # 상태 표시
        status_frame = tk.Frame(self.identity_window)
        status_frame.pack(fill="x", padx=30, pady=20)
        
        # 시도 횟수
        attempt_frame = tk.Frame(status_frame)
        attempt_frame.pack(fill="x", pady=5)
        
        tk.Label(attempt_frame, text="시도 횟수:", font=("Arial", 12)).pack(side="left")
        self.attempt_label = tk.Label(attempt_frame, text="0/5", font=("Arial", 12, "bold"), fg="green")
        self.attempt_label.pack(side="left", padx=(10, 0))
        
        # 깜빡임 횟수
        blink_frame = tk.Frame(status_frame)
        blink_frame.pack(fill="x", pady=5)
        
        tk.Label(blink_frame, text="깜빡임 횟수:", font=("Arial", 12)).pack(side="left")
        self.blink_count_label = tk.Label(blink_frame, text="0/2", font=("Arial", 12, "bold"), fg="blue")
        self.blink_count_label.pack(side="left", padx=(10, 0))
        
        # 상태 메시지
        self.identity_status_label = tk.Label(self.identity_window, 
                                            text="'신원 확인 시작' 버튼을 눌러 시작하세요",
                                            font=("Arial", 12), fg="#2c3e50", 
                                            wraplength=500, justify="center")
        self.identity_status_label.pack(pady=20)
        
        # 버튼들
        button_frame = tk.Frame(self.identity_window)
        button_frame.pack(pady=30)
        
        self.identity_start_button = tk.Button(button_frame, text="🚀 신원 확인 시작", 
                                             command=self.start_identity_process,
                                             width=15, height=2, 
                                             bg="#27ae60", fg="white", 
                                             font=("Arial", 12, "bold"))
        self.identity_start_button.pack(side="left", padx=10)
        
        self.blink_start_button = tk.Button(button_frame, text="👁 깜빡임 감지 시작", 
                                          command=self.start_blink_process,
                                          width=15, height=2, 
                                          bg="#3498db", fg="white", 
                                          font=("Arial", 12, "bold"),
                                          state="disabled")
        self.blink_start_button.pack(side="left", padx=10)
        
        tk.Button(button_frame, text="❌ 취소", command=self.close_identity_window,
                 width=10, height=2, 
                 bg="#e74c3c", fg="white", 
                 font=("Arial", 12, "bold")).pack(side="left", padx=10)
        
        self.identity_window.protocol("WM_DELETE_WINDOW", self.close_identity_window)
    
    def start_identity_process(self):
        """신원 확인 프로세스 시작"""
        if self.supervisor.start_identity_check():
            self.identity_start_button.configure(state="disabled")
            self.blink_start_button.configure(state="normal")
            self.identity_status_label.configure(text="신원 확인 단계가 시작되었습니다.\n'깜빡임 감지 시작' 버튼을 눌러주세요.")
    
    def start_blink_process(self):
        """깜빡임 감지 프로세스 시작"""
        if self.supervisor.start_blink_detection():
            self.blink_start_button.configure(state="disabled", text="깜빡임 감지 중...")
            self.identity_status_label.configure(text="6초 동안 최소 2회 깜빡여 주세요!")
            
            # 시도 횟수 업데이트
            self.attempt_label.configure(text=f"{self.supervisor.identity_attempts}/5")
            self.blink_count_label.configure(text="0/2")
            
            # 6초 후 다시 활성화
            self.root.after(6000, self.reset_blink_button)
    
    def reset_blink_button(self):
        """깜빡임 버튼 리셋"""
        if self.identity_window and self.identity_window.winfo_exists():
            if self.supervisor.identity_active:
                self.blink_start_button.configure(state="normal", text="👁 깜빡임 감지 시작")
                self.identity_status_label.configure(text="다시 시도하거나 결과를 확인하세요.")
            else:
                # 신원 확인 완료됨
                self.close_identity_window()
    
    def close_identity_window(self):
        """신원 확인 창 닫기"""
        if self.identity_window:
            try:
                self.identity_window.grab_release()
            except tk.TclError:
                pass
            self.identity_window.destroy()
            self.identity_window = None
        
        self.supervisor.identity_active = False
        
        # 인증 성공 시 시험 감독 시작
        if self.supervisor.authenticated_user:
            self.supervisor.start_exam_monitoring()
    
    def show_report(self):
        """보고서 표시"""
        if not self.supervisor.authenticated_user:
            messagebox.showinfo("정보", "시험 데이터가 없습니다.")
            return
        
        report_window = tk.Toplevel(self.root)
        report_window.title("📄 시험 보고서")
        report_window.geometry("800x600")
        
        report_text = scrolledtext.ScrolledText(report_window, wrap=tk.WORD)
        report_text.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 보고서 내용 생성
        report_content = f"""
📋 AI 시험 감독관 보고서

사용자: {self.supervisor.authenticated_user}
시험 시간: {self.supervisor.get_exam_duration_string()}
총 경고: {self.supervisor.total_warnings}/{self.supervisor.MAX_WARNINGS}
총 위반: {self.supervisor.total_violations}회

시험 상태: {"❌ 부정행위로 중단됨" if self.supervisor.exam_terminated else "✅ 진행 중"}
"""
        
        if self.supervisor.exam_terminated:
            report_content += f"중단 사유: {self.supervisor.termination_reason}\n"
        
        if self.supervisor.violation_log:
            report_content += "\n위반 내역:\n"
            for violation in self.supervisor.violation_log:
                report_content += f"[{violation['timestamp']}] {violation['type']}: {violation['details']}\n"
        
        report_text.insert(tk.END, report_content)
        report_text.configure(state="disabled")
    
    def update_gui(self):
        """GUI 업데이트"""
        try:
            self.update_camera_display()
            self.update_status_display()
            self.update_log_display()
            
            # 깜빡임 횟수 업데이트
            if (self.identity_window and self.identity_window.winfo_exists() and 
                hasattr(self.supervisor, 'blink_count')):
                self.blink_count_label.configure(text=f"{self.supervisor.blink_count}/2")
        except Exception as e:
            pass
        
        self.root.after(30, self.update_gui)
    
    def load_config(self):
        """설정 불러오기"""
        self.log_message("설정 불러오기 기능은 구현 예정입니다.", "INFO")
    
    def save_config(self):
        """설정 저장"""
        self.log_message("설정 저장 기능은 구현 예정입니다.", "INFO")
    
    def open_log_folder(self):
        """로그 폴더 열기"""
        log_path = "./logs"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        try:
            if sys.platform.startswith('linux'):
                os.system(f"xdg-open {log_path}")
            elif sys.platform.startswith('win'):
                os.system(f"explorer {log_path}")
            elif sys.platform.startswith('darwin'):
                os.system(f"open {log_path}")
        except Exception as e:
            self.log_message(f"폴더 열기 실패: {e}", "ERROR")
    
    def open_dataset_folder(self):
        """데이터셋 폴더 열기"""
        dataset_path = "./dataset"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        try:
            if sys.platform.startswith('linux'):
                os.system(f"xdg-open {dataset_path}")
            elif sys.platform.startswith('win'):
                os.system(f"explorer {dataset_path}")
            elif sys.platform.startswith('darwin'):
                os.system(f"open {dataset_path}")
        except Exception as e:
            self.log_message(f"폴더 열기 실패: {e}", "ERROR")
    
    def show_help(self):
        """도움말 표시"""
        help_text = """
🤖 AI 시험 감독관 시스템 v2.5 사용법

1. 시작하기:
   - '시험 시작' 버튼을 클릭하여 카메라를 연결합니다
   - dataset/ 폴더에 인증용 얼굴 이미지를 추가하세요

2. 신원 확인:
   - '신원 확인' 버튼을 클릭합니다
   - 신원 확인 창에서 '신원 확인 시작' 버튼을 누릅니다
   - '깜빡임 감지 시작' 버튼을 눌러 눈 깜빡임을 시작합니다
   - 6초 동안 최소 2회 깜빡여주세요

3. 시험 진행:
   - 신원 확인 완료 후 자동으로 시험 감독이 시작됩니다
   - 실시간으로 부정행위가 감지됩니다

4. 부정행위 탐지:
   - 다중 인물, 화면 이탈: 즉시 중단
   - 고개/시선 이탈: 5회 경고 후 중단
        """
        messagebox.showinfo("사용법", help_text)
    
    def show_about(self):
        """정보 표시"""
        about_text = """
🤖 AI 시험 감독관 시스템 v2.5

개발: AI Assistant
버전: v2.5 (키보드 입력 제거, GUI 전용)
목적: 원격 시험 부정행위 방지

특징:
✅ 실시간 얼굴 감지 및 추적
✅ 눈 깜빡임 기반 실제 사람 확인  
✅ 정밀한 고개 방향 및 시선 추적
✅ GUI 기반 사용자 친화적 인터페이스
✅ 원본 알고리즘 완전 보존

라이선스: MIT License
        """
        messagebox.showinfo("정보", about_text)
    
    def on_closing(self):
        """프로그램 종료"""
        if self.is_running:
            if messagebox.askokcancel("종료", "시험이 진행 중입니다. 정말 종료하시겠습니까?"):
                self.stop_exam()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """GUI 실행"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.log_message("AI 시험 감독관 시스템 v2.5가 시작되었습니다.", "SUCCESS")
        self.log_message("원본 기능을 모두 유지하면서 키보드 입력을 제거했습니다.", "INFO")
        self.log_message("'시험 시작' 버튼을 눌러 카메라를 연결하세요.", "INFO")
        
        self.root.mainloop()


def speak_tts(text, lang='ko'):
    """TTS 음성을 생성하고 재생"""
    if not TTS_AVAILABLE:
        return
    
    def tts_worker():
        try:
            tts = gTTS(text=text, lang=lang)
            tts_path = "tts_output.mp3"
            tts.save(tts_path)
            
            if sys.platform.startswith('linux'):
                os.system(f"mpg123 {tts_path} > /dev/null 2>&1")
            elif sys.platform.startswith('win'):
                os.system(f"start /min {tts_path}")
            elif sys.platform.startswith('darwin'):
                os.system(f"afplay {tts_path}")
                
            try:
                os.remove(tts_path)
            except:
                pass
        except Exception as e:
            print(f"TTS 오류: {e}")
    
    threading.Thread(target=tts_worker, daemon=True).start()


def main():
    """메인 함수"""
    try:
        app = AIExamSupervisorGUI()
        app.run()
    except Exception as e:
        print(f"GUI 시작 실패: {e}")
        print("필요한 라이브러리를 설치해주세요:")
        print("pip install opencv-python mediapipe face-recognition pillow")
        print("pip install customtkinter  # 선택사항")
        sys.exit(1)


if __name__ == "__main__":
    main()
