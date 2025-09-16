# OpenCV Video Recorder

> 고급컴퓨터비전 과제 1 — **웹캠/카메라 프리뷰 & 녹화 툴**  
> OpenCV로 구현한 비디오 레코더입니다.
> 프리뷰/수동 녹화/모션 인식 자동 녹화, 필터, 자동 파일 분할 등을 지원합니다.  
> **키 입력은 영문 배열 기준**입니다. (예: `m`, `p`, `space`, `ESC` 등)

---

## 목차

- [주요 기능](#주요-기능)
- [데모](#데모)
- [요구 사항](#요구-사항)
- [빠른 시작](#빠른-시작)
- [키 맵](#키-맵)
- [CLI 옵션](#cli-옵션)
- [모션 녹화 동작 원리](#모션-녹화-동작-원리)
- [상태 머신](#상태-머신)
- [출력물 구조](#출력물-구조)
- [트러블슈팅](#트러블슈팅)


---

## 주요 기능

- **Preview / Record 모드**: `Space`로 토글, `ESC`로 종료
- **REC 인디케이터**: 좌상단 붉은 점(녹화 중), 일시정지 시 노란 점
- **HUD 오버레이**: FPS/해상도/필터/모드/모션 레벨/분할 카운트다운
- **비디오 저장**: FourCC 코덱/FPS/해상도 지정
- **실시간 필터**: flip, grayscale, blur, contrast, brightness
- **스냅샷 저장**: PNG로 저장 (`s`)
- **일시정지/재개**: `p`
- **모션-트리거 자동 녹화**: `m`으로 ON/OFF  
  - **워밍업 지연** + **임계 초과 연속 프레임** + **유휴 연속 시간**으로 오탐 최소화
- **자동 파일 분할**: `--auto-split-min N` 분마다 새 파일로 롤오버

---

## 데모

> 아래 자리에 GIF/스크린샷을 추가하세요.
>
> ```text
> recordings/rec_YYYYMMDD_HHMMSS.mp4
> recordings/rec_YYYYMMDD_HHMMSS.png
> ```
>
> 예시: HUD/REC 표시 및 “Recording started/stopped”, “Motion auto recording: ON/OFF” 토스트 메시지 표기

---

## 요구 사항

- Python **3.8+**
- cv==1.0.0
- numpy==2.2.6
- opencv-python==4.12.0.88
- pillow==11.3.0
- wheel==0.45.1



---

## 실행 명령어 예시

python video_recorder.py \
  --source auto \
  --backend any \
  --codec mp4v \
  --fps 30 \
  --width 1280 --height 720 \
  --outdir ./recordings \
  --auto-split-min 10

