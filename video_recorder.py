#!/usr/bin/env python3

#고급컴퓨터비전 과제 1
#video_recoder

###기능###
"""
- Preview / Record 모드 전환
- Red REC dot & HUD overlay
- video file 저장
- 필터: flip, grayscale, blur, contrast, brightness
- Snapshot
- Pause/Resume
- motion 인식 촬영
- Auto file split
"""
##########################
#!!!file 실행 방법!!!#
'''
$ python video_recorder.py --source auto --backend any --codec mp4v --fps 30 --width 1280 --height 720 \
    --outdir ./recordings --prefix session --auto-split-min 10
'''
##########################



###key Map(카메라 스크린 참고)###
'''
  ESC       : Exit
  Space     : Preview ↔ Record 전환
  p         : Pause/Resume (when recording)
  s         : Save snapshot
  m         : Motio인식 촬녕 on/off
  h         : hlep HUD on/off

  f         : flip ON/OFF
  g         : Grayscale ON/OFF
  b         : Blur ON/OFF
  [ / ]     : Blur kernel size - / +
  - / =     : Contrast - / +
  ; / '     : Brightness - / +
  n         : new flie로 spilt

'''


import argparse
from pathlib import Path
import sys
import time
import cv2 as cv
import numpy as np
from datetime import datetime, timedelta

FOURCC_MAP = {
    'mp4v': ('mp4v', '.mp4'),
    'xvid': ('XVID', '.avi'),
    'mjpg': ('MJPG', '.avi'),
    'avc1': ('avc1', '.mp4'),  
    'h264': ('H264', '.mp4'), 
}

def make_writer(out_path: Path, size, fps: float, codec: str):
    """Create VideoWriter. Fallback to MJPG(.avi) if it fails."""
    fourcc_str, ext = FOURCC_MAP.get(codec.lower(), FOURCC_MAP['mp4v'])
    file_path = out_path.with_suffix(ext)
    fourcc = cv.VideoWriter_fourcc(*fourcc_str)
    writer = cv.VideoWriter(str(file_path), fourcc, fps, size, isColor=True)

    if not writer.isOpened():
        fourcc_fallback, ext_fb = FOURCC_MAP['mjpg']
        file_path = out_path.with_suffix(ext_fb)
        writer = cv.VideoWriter(str(file_path), cv.VideoWriter_fourcc(*fourcc_fallback), fps, size, isColor=True)
    return writer, file_path

# ----------------------------
# Filters / HUD
# ----------------------------
class FilterState:
    def __init__(self):
        self.flip = False
        self.gray = False
        self.blur = False
        self.blur_k = 5  # odd
        self.contrast = 1.0  # alpha
        self.brightness = 0  # beta (-100~100)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        img = frame
        if self.flip:
            img = cv.flip(img, 1)
        if self.gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if self.blur and self.blur_k > 1:
            k = max(1, self.blur_k)
            if k % 2 == 0:
                k += 1
            img = cv.GaussianBlur(img, (k, k), 0)
        # contrast/brightness
        img = cv.convertScaleAbs(img, alpha=self.contrast, beta=self.brightness)
        return img

class HUD:
    def __init__(self):
        self.show_help = True
        self.show_timestamp = True

    def draw(self, frame: np.ndarray, *,
             recording: bool, paused: bool,
             mode_text: str,
             fps_disp: float,
             size: tuple,
             filters: FilterState,
             motion_on: bool,
             motion_level: float,
             rec_reason: str,
             split_info: str,
             ):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        # REC indicator
        if recording and not paused:
            cv.circle(overlay, (20, 20), 10, (0, 0, 255), thickness=-1)
        elif paused:
            cv.circle(overlay, (20, 20), 10, (0, 255, 255), thickness=-1)  
        alpha = 0.6
        frame[:] = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        if self.show_timestamp:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv.putText(frame, ts, (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        # Help HUD 
        lines = [
            f"Mode: {mode_text} | Reason: {rec_reason} | FPS: {fps_disp:.1f} | {size[0]}x{size[1]}",
            f"Filters: flip={filters.flip} gray={filters.gray} blur={filters.blur} k={filters.blur_k} alpha={filters.contrast:.2f} beta={filters.brightness}",
            f"Motion: {'ON' if motion_on else 'OFF'} level={motion_level:.3f} | Split: {split_info}",
            "Keys: Space(rec) p(pause) s(snap) m(motion) t(ts) h(help) ESC(quit)",
            "      f(flip) g(gray) b(blur) [/] blur-+/  -/= contrast-+/  ;/' bright-+/  n(new file)",
        ]
        y = 25
        for i, text in enumerate(lines[: (0 if not self.show_help else len(lines))]):
            cv.putText(frame, text, (40, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            y += 20

class Toast:
    def __init__(self):
        self.msg = None
        self.until = 0.0

    def show(self, text: str, sec: float = 1.5):
        self.msg = str(text)
        self.until = time.monotonic() + float(sec)

    def draw(self, frame: np.ndarray):
        if not self.msg:
            return
        now = time.monotonic()
        if now > self.until:
            self.msg = None
            return

        h, w = frame.shape[:2]
        box_w, box_h = int(w * 0.6), 50
        x1 = (w - box_w) // 2
        y1 = int(h * 0.85)
        x2, y2 = x1 + box_w, y1 + box_h

        overlay = frame.copy()
        cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame[:] = cv.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv.putText(frame, self.msg, (x1 + 15, y1 + 32),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

# ----------------------------
# Motion 인식
# ----------------------------
class MotionTrigger:
    def __init__(self, thresh=0.3, idle_seconds=5.0, enabled=False,
                 arm_delay=1.5,           # 모드 ON 후 이 시간 동안은 감지 무시(초)
                 min_on_frames=10):       # 시작 판정: 임계 초과 연속 프레임 수
        self.bg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.thresh = float(thresh)
        self.idle_seconds = float(idle_seconds)
        self.enabled = bool(enabled)
        self.arm_delay = float(arm_delay)
        self.min_on_frames = int(min_on_frames)

        self._enabled_at = 0.0
        self._above_cnt = 0
        self._below_since = None
        self._active = False
        self.last_level = 0.0

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)
        self._enabled_at = time.time()
        self._above_cnt = 0
        self._below_since = None
        self._active = False
        self.last_level = 0.0
        self.bg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def update(self, frame: np.ndarray) -> float:
        if not self.enabled:
            self.last_level = 0.0
            return 0.0

        mask = self.bg.apply(frame)
        level = float(np.count_nonzero(mask)) / float(mask.size)
        self.last_level = level

        if (time.time() - self._enabled_at) < self.arm_delay:
            return level

        if level >= self.thresh:
            self._above_cnt += 1
            self._below_since = None
        else:
            self._above_cnt = 0
            if self._active and self._below_since is None:
                self._below_since = time.time()

        if not self._active and self._above_cnt >= self.min_on_frames:
            self._active = True
            self._below_since = None

        if self._active and self._below_since is not None:
            if (time.time() - self._below_since) >= self.idle_seconds:
                self._active = False
                self._below_since = None

        return level

    def is_active(self) -> bool:
        return self.enabled and self._active


# ----------------------------
# RecorD staet
# ----------------------------
class RecordState:
    """idle / manual / auto + pause"""
    def __init__(self):
        self.mode = 'idle'   # 'idle' | 'manual' | 'auto'
        self.paused = False
        self.started_at = None

    @property
    def recording(self) -> bool:
        return self.mode in ('manual', 'auto')

    def human_mode(self):
        if self.paused:
            return 'PAUSED'
        return 'RECORD' if self.recording else 'PREVIEW'

    def reason(self):
        return self.mode.upper()

# ------------
# Input source
# ------------
def _backend_code(name: str) -> int:
    name = (name or 'any').lower()
    return {
        'any': cv.CAP_ANY,
        'v4l2': cv.CAP_V4L2,
        'gstreamer': cv.CAP_GSTREAMER,
        'ffmpeg': cv.CAP_FFMPEG,
        'dshow': cv.CAP_DSHOW,
        'msmf': cv.CAP_MSMF,
        'avfoundation': cv.CAP_AVFOUNDATION,
    }.get(name, cv.CAP_ANY)

def open_capture(args) -> cv.VideoCapture:
    be = _backend_code(args.backend)

    def _open_by_source(src):
        if isinstance(src, int):
            return cv.VideoCapture(int(src), be)
        try:
            i = int(str(src))
            return cv.VideoCapture(i, be)
        except ValueError:
            if be == cv.CAP_GSTREAMER:
                return cv.VideoCapture(str(src), be)
            else:
                return cv.VideoCapture(str(src))

    if args.source:
        if args.source.lower() == 'auto':
            for i in range(0, 10):
                cap = cv.VideoCapture(i, be)
                if cap.isOpened():
                    print(f"[INFO] Auto-picked camera index {i}")
                    return cap
            return cv.VideoCapture()  # not opened
        else:
            return _open_by_source(args.source)

    return cv.VideoCapture(args.device, be)




# Main $$$$

def parse_args():
    p = argparse.ArgumentParser(description='OpenCV Video Recorder')
    # Input source
    p.add_argument('--source', type=str, default='',
                   help="Input: integer index for camera, file path/RTSP/HTTP, or 'auto'")
    p.add_argument('--device', type=int, default=0, help='(legacy) camera index — ignored if --source is set')
    p.add_argument('--backend', type=str, default='any',
                   choices=['any','v4l2','gstreamer','ffmpeg','dshow','msmf','avfoundation'],
                   help='Select VideoCapture backend')

    # Recording / output
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=float, default=30.0, help='Output FPS (recommend explicit)')
    p.add_argument('--codec', type=str, default='mp4v', choices=list(FOURCC_MAP.keys()))
    p.add_argument('--outdir', type=str, default='./recordings')
    p.add_argument('--prefix', type=str, default='rec')
    p.add_argument('--auto-split-min', type=int, default=0, help='Split every N minutes (0=off)')

    # Motion 인식 초기화
    p.add_argument('--motion-thresh', type=float, default=0.02, help='Motion threshold (0~1)')
    p.add_argument('--motion-idle-sec', type=float, default=5.0, help='Stop after this many idle seconds')

    return p.parse_args()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def gen_filename(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    cap = open_capture(args)
    if not cap or not cap.isOpened():
        print(f"[ERROR] Failed to open input source. Try --source 'auto', index (0/1/2), file path, RTSP/HTTP, or GStreamer pipeline. backend={args.backend}")
        print("[HINT] Another app may be using the camera, permissions missing, device absent, or container lacks passthrough.")
        sys.exit(1)


    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv.CAP_PROP_FPS, args.fps)


    real_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or args.width)
    real_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or args.height)
    size = (real_w, real_h)

    filters = FilterState()
    hud = HUD()
    toast = Toast()
    motion = MotionTrigger(thresh=args.motion_thresh, idle_seconds=args.motion_idle_sec, enabled=False)  # start OFF
    rec = RecordState()

    writer = None
    current_path = None
    split_due = None

    # FPS display
    last_time = time.time()
    fps_disp = 0.0
    frames_counted = 0

    win = 'VideoRecorder'
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, real_w, real_h)

    def start_new_writer(mode_reason: str):
        nonlocal writer, current_path, split_due, rec
        filename = gen_filename(args.prefix)
        writer, current_path = make_writer(outdir / filename, size, args.fps, args.codec)
        rec.started_at = datetime.now()
        if args.auto_split_min > 0:
            split_due = datetime.now() + timedelta(minutes=args.auto_split_min)
        else:
            split_due = None
        print(f"[INFO] Start recording ({mode_reason}) -> {current_path}")

    def close_writer():
        nonlocal writer, current_path
        if writer is not None:
            writer.release()
            print(f"[INFO] Saved -> {current_path}")
        writer = None
        current_path = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print('[WARN] Failed to read frame. Exiting.')
            break

        # Filters
        frame = filters.apply(frame)

        # Motion
        motion_level = motion.update(frame)

        # Auto state transitions
        if rec.mode == 'idle' and motion.is_active():
            rec.mode = 'auto'
            rec.paused = False
            start_new_writer('AUTO')
        elif rec.mode == 'auto' and not motion.is_active():
            close_writer()
            rec.mode = 'idle'
            rec.paused = False

        # Auto split
        split_info = 'OFF'
        if rec.recording and not rec.paused and split_due is not None:
            remain = (split_due - datetime.now()).total_seconds()
            split_info = f"{remain:0.0f}s to roll"
            if remain <= 0:
                close_writer()
                start_new_writer(rec.reason())

        # Write frame
        if rec.recording and not rec.paused and writer is not None:
            if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                frame = cv.resize(frame, (size[0], size[1]))
            writer.write(frame)

        # FPS calc
        frames_counted += 1
        now = time.time()
        if now - last_time >= 0.5:
            fps_disp = frames_counted / (now - last_time)
            last_time = now
            frames_counted = 0

        # HUD + Toast
        hud.draw(frame,
                 recording=rec.recording,
                 paused=rec.paused,
                 mode_text=rec.human_mode(),
                 fps_disp=fps_disp,
                 size=size,
                 filters=filters,
                 motion_on=motion.enabled,
                 motion_level=motion_level,
                 rec_reason=rec.reason(),
                 split_info=split_info,
                 )
        toast.draw(frame)
        cv.imshow(win, frame)

        # Keys
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('h'):
            hud.show_help = not hud.show_help
        elif key == ord('t'):
            hud.show_timestamp = not hud.show_timestamp
        elif key == ord('f'):
            filters.flip = not filters.flip
        elif key == ord('g'):
            filters.gray = not filters.gray
        elif key == ord('b'):
            filters.blur = not filters.blur
        elif key == ord('['):
            filters.blur_k = max(1, filters.blur_k - 2)
        elif key == ord(']'):
            filters.blur_k = filters.blur_k + 2
        elif key == ord('-'):
            filters.contrast = max(0.2, filters.contrast - 0.05)
        elif key == ord('='):
            filters.contrast = min(3.0, filters.contrast + 0.05)
        elif key == ord(';'):
            filters.brightness = max(-100, filters.brightness - 2)
        elif key == ord("'"):
            filters.brightness = min(100, filters.brightness + 2)
        elif key == ord('s'):
            snap_name = f"{gen_filename(args.prefix)}.png"
            cv.imwrite(str(outdir / snap_name), frame)
            print(f"[SNAP] {outdir / snap_name}")
        elif key == ord('m'):
            motion.set_enabled(not motion.enabled)  # ← set_enabled() 사용
            if motion.enabled:
                toast.show('Motion auto recording: ON', 1.2)
                print('[MOTION] ON: armed, waiting for motion (with warm-up)')
            else:
                toast.show('Motion auto recording: OFF', 1.2)
                print('[MOTION] OFF: auto recording disabled')

        elif key == ord('p'):
            if rec.recording:
                rec.paused = not rec.paused
                toast.show('Paused' if rec.paused else 'Resumed', 1.2)
        elif key == ord('n'):
            if rec.recording and not rec.paused:
                close_writer()
                start_new_writer(rec.reason())
        elif key == 32:  # Space
            if rec.mode == 'manual':
                close_writer()
                rec.mode = 'idle'
                rec.paused = False
                toast.show('Recording stopped', 1.2)
            elif rec.mode in ('idle', 'auto'):
                if rec.mode == 'auto':
                    close_writer()
                rec.mode = 'manual'
                rec.paused = False
                start_new_writer('MANUAL')
                toast.show('Recording started', 1.2)

    # Cleanup
    close_writer()
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
