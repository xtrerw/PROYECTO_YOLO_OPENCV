"""驾驶视觉分析台。直接运行：python driving_vision.py

空格暂停，Q 退出，H 开关轨迹，R 开关观察区域，S 截图。
拖动窗口下方 Progress 进度条跳转；跳转会重置轨迹和面板累计计数。
无窗口导出示例：python driving_vision.py --headless --save --max-frames 300
依赖当前项目已有的 ultralytics、opencv-python、numpy 和 lap。
"""

import argparse
import csv
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
INK = (225, 235, 243)
CYAN = (235, 215, 45)
AMBER = (45, 175, 255)


def label(img, text, xy, scale=0.55, color=INK, thickness=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def filter_class_confidence(predictor, main_conf, other_conf):
    """在 ByteTrack 回调之前，按类别过滤检测结果。"""
    for i, result in enumerate(predictor.results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        main_ids = [cid for cid, name in result.names.items() if name in ('person', 'car')]
        is_main = boxes.cls == -1
        for cid in main_ids:
            is_main |= boxes.cls == cid
        keep = (is_main & (boxes.conf >= main_conf)) | (~is_main & (boxes.conf >= other_conf))
        predictor.results[i] = result[keep]


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--source', type=Path, default=ROOT / 'test.mp4')
    p.add_argument('--model', type=Path, default=ROOT / 'yolo11n.pt')
    p.add_argument('--output', type=Path, default=ROOT / 'vision_output')
    p.add_argument('--conf', type=float, default=0.6, help='person 和 car 的置信度阈值')
    p.add_argument('--other-conf', type=float, default=0.3, help='其他类别的置信度阈值')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--width', type=int, default=1120)
    p.add_argument('--start', type=float, default=0, help='从第几秒开始')
    p.add_argument('--max-frames', type=int, default=0, help='0 表示处理到结束')
    p.add_argument('--device', default=None, help='例如 cpu 或 0')
    p.add_argument('--save', action='store_true', help='保存带面板的视频')
    p.add_argument('--headless', action='store_true', help='无窗口运行')
    p.add_argument('--roi', type=float, nargs=8,
                   default=[0.39, 0.08, 0.47, 0.08, 0.66, 0.35, 0.25, 0.35],
                   metavar=('X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'),
                   help='观察区域四个顶点，相对画面宽高，范围 0 到 1')
    a = p.parse_args()
    if not 0 < a.conf <= 1 or not 0 < a.other_conf <= 1 or a.width < 640 or a.imgsz < 32:
        p.error('conf 和 other-conf 必须在 (0,1]，width >= 640，imgsz >= 32')
    if a.start < 0 or a.max_frames < 0 or any(v < 0 or v > 1 for v in a.roi):
        p.error('start/max-frames 不能为负，roi 坐标必须在 [0,1]')
    return a


def run(a):
    if not a.source.is_file() or not a.model.is_file():
        raise FileNotFoundError(f'请检查视频和模型路径：{a.source} / {a.model}')
    model = YOLO(str(a.model))
    # 首次 model.track 会追加跟踪回调，因此先注册类别过滤。
    model.add_callback('on_predict_postprocess_end',
                       lambda predictor: filter_class_confidence(predictor, a.conf, a.other_conf))
    cap = cv2.VideoCapture(str(a.source))
    writer = None
    logfile = None
    processed = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f'无法打开视频：{a.source}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = round(a.start * fps)
        if total > 0 and start_frame >= total:
            raise ValueError('--start 超过视频时长')
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = a.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out.mkdir(parents=True, exist_ok=True)
        logfile = (out / 'detections.csv').open('w', newline='', encoding='utf-8-sig')
        log = csv.writer(logfile)
        log.writerow(['frame', 'video_seconds', 'track_id', 'class', 'confidence',
                      'x1', 'y1', 'x2', 'y2', 'in_watch_region', 'segment'])
        histories, last_seen, seen = {}, {}, {}
        entries = set()
        show_trails, show_roi = True, True
        times = deque(maxlen=30)
        window = 'DRIVE / VISION LAB'
        pending_seek = None
        updating_progress = False
        paused = False
        segment = 0

        def on_progress(position):
            nonlocal pending_seek
            if not updating_progress:
                pending_seek = position

        if not a.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, a.width + 300, round(a.width * 9 / 16))
            if total > 1:
                updating_progress = True
                try:
                    cv2.createTrackbar('Progress', window, start_frame, total - 1, on_progress)
                finally:
                    updating_progress = False
        print(f'输出目录：{out}', flush=True)
        while not a.max_frames or processed < a.max_frames:
            tick = time.perf_counter()
            if pending_seek is not None:
                target = pending_seek
                pending_seek = None
                if cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                    for tracker in getattr(model.predictor, 'trackers', []):
                        tracker.reset()
                    histories.clear()
                    last_seen.clear()
                    seen.clear()
                    entries.clear()
                    times.clear()
                    segment += 1
                else:
                    print(f'无法跳转到第 {target} 帧', flush=True)
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ok, original = cap.read()
            if not ok:
                break
            # 保持视频比例；CSV 坐标转换回原始视频坐标。
            h = round(original.shape[0] * a.width / original.shape[1])
            h += h % 2
            w = a.width + a.width % 2
            frame = cv2.resize(original, (w, h))
            polygon = (np.array(a.roi).reshape(4, 2) * [w, h]).astype(np.int32)
            result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                 classes=None, conf=min(a.conf, a.other_conf),
                                 imgsz=a.imgsz, device=a.device, verbose=False)[0]
            boxes = result.boxes
            tracks = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
            active, watched = Counter(), 0
            dots = []
            if show_roi:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], CYAN)
                frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
                cv2.polylines(frame, [polygon], True, CYAN, 1, cv2.LINE_AA)
            for box, tid in zip(boxes, tracks):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                cls = result.names[int(box.cls.item())]
                confidence = float(box.conf.item())
                point = ((x1 + x2) // 2, y2)
                inside = cv2.pointPolygonTest(polygon, point, False) >= 0
                active[cls] += 1
                watched += int(inside)
                color = AMBER if inside else CYAN
                if tid >= 0:
                    seen.setdefault(tid, cls)
                    if inside:
                        entries.add(tid)
                    histories.setdefault(tid, deque(maxlen=40)).append(point)
                    last_seen[tid] = processed
                    if show_trails and len(histories[tid]) > 1:
                        cv2.polylines(frame, [np.array(histories[tid], dtype=np.int32)],
                                      False, color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                for x, y, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                      (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(frame, (x, y), (x + dx * 12, y), color, 3)
                    cv2.line(frame, (x, y), (x, y + dy * 12), color, 3)
                text = f'{cls} #{tid if tid >= 0 else "?"} {confidence:.0%}'
                ty = max(y1 - 7, 17)
                tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
                cv2.rectangle(frame, (x1, ty - 16), (min(w, x1 + tw + 8), ty + 4), (20, 27, 32), -1)
                label(frame, text, (x1 + 3, ty), .45, color)
                cv2.circle(frame, point, 3, color, -1)
                dots.append((point[0] / w, point[1] / h, color))
                sx, sy = original.shape[1] / w, original.shape[0] / h
                log.writerow([frame_no, round(frame_no / fps, 3), tid, cls,
                              round(confidence, 4), round(x1 * sx), round(y1 * sy),
                              round(x2 * sx), round(y2 * sy), int(inside), segment])
            for tid in list(last_seen):
                if processed - last_seen[tid] > fps * 3:
                    histories.pop(tid, None)
                    last_seen.pop(tid, None)
            times.append(time.perf_counter() - tick)
            panel = np.full((h, 300, 3), (23, 19, 16), dtype=np.uint8)
            label(panel, 'DRIVE / VISION', (20, 35), .7, CYAN, 2)
            label(panel, 'YOLO11 + ByteTrack', (20, 60), .45)
            label(panel, f'{frame_no / fps:07.1f}s  |  {1 / np.mean(times):.1f} FPS', (20, 92))
            label(panel, f'VISIBLE       {sum(active.values()):02d}', (20, 129), .65)
            label(panel, f'TRACK IDS     {len(seen):02d}', (20, 160), .65)
            label(panel, f'IN REGION     {watched:02d}', (20, 191), .65, AMBER if watched else CYAN)
            label(panel, 'REGION OCCUPIED' if watched else 'REGION EMPTY', (20, 226), .55, AMBER if watched else CYAN)
            label(panel, 'Image region only / not distance', (20, 250), .4)
            # 小地图是图像坐标分布，不是鸟瞰投影。
            map_y, map_h = 288, max(40, min(130, h - 370))
            if h >= 400:
                label(panel, 'IMAGE POSITION MAP', (20, map_y - 12), .45)
                cv2.rectangle(panel, (20, map_y), (280, map_y + map_h), (65, 65, 55), 1)
                mini_poly = (np.array(a.roi).reshape(4, 2) * [260, map_h] + [20, map_y]).astype(np.int32)
                cv2.polylines(panel, [mini_poly], True, CYAN, 1)
                for x, y, color in dots:
                    cv2.circle(panel, (int(20 + 260 * x), int(map_y + map_h * y)), 4, color, -1)
                for i, (name, count) in enumerate(active.most_common(4)):
                    yy = map_y + map_h + 25 + i * 21
                    if yy < h - 55:
                        label(panel, f'{name:<13} {count}', (20, yy), .45)
            label(panel, 'SPACE pause   Q quit   S snapshot', (16, h - 32), .4)
            label(panel, 'H trails      R region', (16, h - 12), .4)
            canvas = np.hstack((frame, panel))
            if total:
                cv2.rectangle(canvas, (0, h - 4), (int(canvas.shape[1] * (frame_no + 1) / total), h - 1), CYAN, -1)
            if a.save:
                if writer is None:
                    writer = cv2.VideoWriter(str(out / 'drive_vision.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (canvas.shape[1], h))
                    if not writer.isOpened():
                        raise RuntimeError('无法创建 MP4 输出，请检查编码器或输出路径')
                writer.write(canvas)
            processed += 1
            if processed == 1:
                cv2.imwrite(str(out / 'preview.jpg'), canvas)
            if processed % 100 == 0:
                print(f'已处理 {processed} 帧，跟踪 ID {len(seen)}，画面时间 {frame_no / fps:.1f}s', flush=True)
                logfile.flush()
            if not a.headless:
                cv2.imshow(window, canvas)
                if total > 1 and pending_seek is None:
                    updating_progress = True
                    try:
                        cv2.setTrackbarPos('Progress', window, frame_no)
                    finally:
                        updating_progress = False
                while True:
                    delay = 100 if paused else max(1, int(1000 / fps - (time.perf_counter() - tick) * 1000))
                    key = cv2.waitKey(delay) & 0xFF
                    if key == 32:
                        paused = not paused
                    if key == ord('h'):
                        show_trails = not show_trails
                    if key == ord('r'):
                        show_roi = not show_roi
                    if key == ord('s'):
                        cv2.imwrite(str(out / f'frame_{frame_no:06d}.jpg'), canvas)
                    if (not paused or pending_seek is not None or key in (ord('q'), 27)
                            or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1):
                        break
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        if not processed:
            raise RuntimeError('未能读取任何视频帧')
        summary = (f'Processed frames: {processed}\nTrack IDs: {len(seen)}\n'
                   f'Seeks: {segment} (counts below and Track IDs refer to the last segment)\n'
                   f'IDs entering region: {len(entries)}\nClasses by first observation: {dict(Counter(seen.values()))}\n'
                   'Track IDs may fragment; these are not exact unique object counts.\n'
                   'CSV segment distinguishes IDs after seeking; saved video follows playback order.\n'
                   'Region is a fixed image polygon, not lane detection or collision prediction.\n')
        (out / 'summary.txt').write_text(summary, encoding='utf-8')
        print(summary, flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if logfile is not None:
            logfile.close()
        if not a.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    run(arguments())
