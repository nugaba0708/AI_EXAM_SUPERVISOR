## 설치해야할 파일 및 참고사항
* 라이브러리
```
pip install opencv-python mediapipe numpy face-recognition pillow
```
* 사전 패키지 설치:
```
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
sudo apt-get install python3-tk
```
* 폴더 준비
```
./dataset 폴더:
얼굴 인증에 사용할 이미지 파일(.jpg, .png 등)이 최소 1장 이상 있어야 합니다.
./log 폴더:
로그 파일이 저장되는 폴더입니다.
코드에서 자동 생성하지만, 권한 문제를 피하려면 미리 만들어 두는 것도 좋습니다.
```
* 카메라 연결 확인
```
# num 값: 기본카메라:0, 외장카메라: 1 or 2
cap = cv2.VideoCapture(num)
```
