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
- [실행 명령어 예시](#실행-명령어-예시)
- [키 맵](#키-맵)
- [모션 녹화 동작 원리](#모션-녹화-동작-원리)


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

> 영상 촬영 상태인 화면 예시
<img width="1440" height="900" alt="Image" src="https://github.com/user-attachments/assets/64da4394-0780-4c44-a219-3d7f9daa512d" />

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
> ```text
> python video_recorder.py \
>  --source auto \
>  --backend any \
>  --codec mp4v \
>  --fps 30 \
>  --width 1280 --height 720 \
>  --outdir ./recordings \
>  --auto-split-min 10
> ```


---

## 키 맵
한/영 키를 이용하여 '영어' 입력 상태로 진행해야 합니다.

| Key       | Action                                  |
| --------- | --------------------------------------- |
| `ESC`     | Exit                                    |
| `Space`   | Manual record toggle (Preview ↔ Record) |
| `p`       | Pause/Resume (when recording)           |
| `s`       | Save snapshot (PNG)                     |
| `m`       | Motion-triggered recording ON/OFF       |
| `t`       | Toggle timestamp HUD                    |
| `h`       | Toggle help HUD                         |
| `f`       | Flip (mirror) ON/OFF                    |
| `g`       | Grayscale ON/OFF                        |
| `b`       | Blur ON/OFF                             |
| `[` / `]` | Blur kernel size − / + *(odd only)*     |
| `-` / `=` | Contrast − / +                          |
| `;` / `'` | Brightness − / +                        |
| `n`       | Split to a new file (roll over now)     |



---

## 모션 녹화 동작 원리

- **지연(Arming Delay)**  
  모션 모드 **ON 직후** 일정 시간(`arm_delay`) 동안은 감지를 **무시**합니다. 초기 배경 모델 적응 중 발생하는 스파이크를 막기 위함입니다.

- **시작(Start) 판정**  
  `motion_level ≥ motion_thresh` 상태가 **연속 `min_on_frames` 프레임** 발생하면 **활성(녹화 시작)** 됩니다.

- **정지(Stop) 판정**  
  활성 상태에서 `motion_level < motion_thresh`가 **연속 `motion_idle_sec`초** 이상 유지되면 **비활성(녹화 종료)** 됩니다.

> **참고(HUD 표시)**  
> HUD의 `Motion level`은 **현재 프레임**의 값입니다. 방금 전 프레임들에서 임계 초과 연속 조건이 이미 충족되었다면, **다음 프레임의 레벨이 낮아도** 녹화가 시작된 상태일 수 있습니다(정상 동작).

### 튜닝 가이드

- **민감도 높이기(더 쉽게 시작되게)**: `--motion-thresh` **낮추기** (예: `0.02 → 0.01`)
- **민감도 낮추기(더 둔감하게)**: `--motion-thresh` **높이기** (예: `0.02 → 0.05~0.10`)
- **오탐 줄이기**: `arm_delay` **증가** / `min_on_frames` **증가** *(코드 상수)*
