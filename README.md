# motion-recognition-video-recorder
My simple video recorder using OpenCV with motion recognition and Filter feature

✨ 주요 기능
Preview / Record 모드 (Space 토글, ESC 종료)
REC 인디케이터(좌상단 붉은 점) & HUD 오버레이(FPS/해상도/필터/모션 상태)

비디오 저장: 코덱(FourCC) / FPS / 해상도 지정

실시간 필터: flip, grayscale, blur, contrast, brightness

스냅샷 저장 (s, PNG)

일시정지/재개 (p)

모션-트리거 자동 녹화 (m)

워밍업 지연 + 임계 초과 연속 프레임 + 연속 무동작 종료 로 오탐 최소화

자동 파일 분할: --auto-split-min N
