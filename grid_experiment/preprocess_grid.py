"""
preprocess_grid.py — Предобработка GRID датасета без dlib.

Использует MediaPipe Face Mesh для извлечения области губ.
Результат: .npy файлы с кадрами губ для каждого видео.

Установка: pip install mediapipe

Использование:
    python preprocess_grid.py
    python preprocess_grid.py --speaker s1 --method mediapipe
    python preprocess_grid.py --method fixed   # без детектора, фиксированный crop
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


# ── Целевой размер области губ ──
TARGET_W = 96
TARGET_H = 64


# ====================== MediaPipe ======================

class MediaPipeLipExtractor:
    """Извлечение губ через MediaPipe Face Mesh (468 landmarks)."""

    # Индексы landmarks для контура губ (outer + inner)
    LIP_INDICES = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,  # outer upper
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,    # outer lower
    ]

    def __init__(self, margin=0.4):
        import mediapipe as mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,  # video mode — tracking between frames
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.margin = margin

    def extract(self, frame):
        """frame: BGR (H, W, 3) → lip ROI (TARGET_H, TARGET_W, 3) or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Координаты губ в пикселях
        lip_x = [landmarks[i].x * w for i in self.LIP_INDICES]
        lip_y = [landmarks[i].y * h for i in self.LIP_INDICES]

        cx = (min(lip_x) + max(lip_x)) / 2
        cy = (min(lip_y) + max(lip_y)) / 2
        lw = max(lip_x) - min(lip_x)
        lh = max(lip_y) - min(lip_y)

        # Расширяем с margin и подгоняем aspect ratio
        bw = lw * (1 + 2 * self.margin)
        bh = lh * (1 + 2 * self.margin)
        aspect = TARGET_W / TARGET_H

        if bw / bh > aspect:
            rw, rh = bw, bw / aspect
        else:
            rh, rw = bh, bh * aspect

        x1 = max(0, int(cx - rw / 2))
        y1 = max(0, int(cy - rh / 2))
        x2 = min(w, int(cx + rw / 2))
        y2 = min(h, int(cy + rh / 2))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        return cv2.resize(roi, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    def close(self):
        self.face_mesh.close()


# ====================== OpenCV DNN ======================

class OpenCVDNNLipExtractor:
    """
    Извлечение губ через OpenCV DNN face detector.
    Не требует дополнительных файлов — использует встроенную модель.
    Область рта = нижняя 1/3 лица + margin.
    """

    def __init__(self, margin=0.15):
        self.net = cv2.dnn.readNetFromCaffe(
            str(Path(__file__).parent / "deploy.prototxt"),
            str(Path(__file__).parent / "res10_300x300_ssd_iter_140000.caffemodel"),
        )
        self.margin = margin

    def extract(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        self.net.setInput(blob)
        detections = self.net.forward()

        if detections.shape[2] == 0:
            return None

        # Берём первое лицо с confidence > 0.5
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                face_h = y2 - y1

                # Область рта = нижняя треть лица
                mouth_y1 = y1 + int(face_h * 0.6)
                mouth_y2 = y2 + int(face_h * self.margin)
                mouth_cx = (x1 + x2) // 2
                mouth_w = int((x2 - x1) * 0.7)

                mx1 = max(0, mouth_cx - mouth_w // 2)
                mx2 = min(w, mouth_cx + mouth_w // 2)
                my1 = max(0, mouth_y1)
                my2 = min(h, mouth_y2)

                roi = frame[my1:my2, mx1:mx2]
                if roi.size == 0:
                    return None
                return cv2.resize(roi, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        return None

    def close(self):
        pass


# ====================== Fixed Crop ======================

class FixedCropLipExtractor:
    """
    Фиксированный crop для GRID — без детектора.
    GRID видео: 360×288, лицо в центре, рот примерно в нижней части.
    Координаты подбираются эмпирически.
    """

    def __init__(self):
        # Эмпирические координаты рта для GRID s1 (360×288)
        # Можно подстроить после визуальной проверки
        self.y1 = 190
        self.y2 = 260
        self.x1 = 115
        self.x2 = 245

    def extract(self, frame):
        roi = frame[self.y1:self.y2, self.x1:self.x2]
        if roi.size == 0:
            return None
        return cv2.resize(roi, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    def close(self):
        pass


# ====================== Обработка ======================

def process_speaker(video_dir, output_dir, extractor, skip_existing=True):
    """
    Обработать все видео спикера → сохранить .npy с кадрами губ.

    Каждый .npy: shape (T, H, W, 3), dtype uint8
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mpg"))
    print(f"Видео: {len(videos)} в {video_dir}")

    stats = {"processed": 0, "skipped": 0, "failed": 0, "no_face_frames": 0}

    for video_path in tqdm(videos, desc="Processing"):
        out_path = output_dir / f"{video_path.stem}.npy"

        if skip_existing and out_path.exists():
            stats["skipped"] += 1
            continue

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        no_face = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            roi = extractor.extract(frame)
            if roi is not None:
                frames.append(roi)
            else:
                no_face += 1

            frame_count += 1

        cap.release()

        if frames:
            arr = np.stack(frames)  # (T, H, W, C)
            np.save(out_path, arr)
            stats["processed"] += 1
            if no_face > 0:
                stats["no_face_frames"] += no_face
        else:
            stats["failed"] += 1
            print(f"  [!] Не удалось обработать {video_path.name}")

    print(f"\nГотово:")
    print(f"  Обработано:  {stats['processed']}")
    print(f"  Пропущено:   {stats['skipped']} (уже в кэше)")
    print(f"  Не удалось:  {stats['failed']}")
    print(f"  Кадров без лица: {stats['no_face_frames']}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess GRID dataset")
    parser.add_argument("--grid-path", type=str,
                        default="D:/GitHub/Datasets/GRID/data",
                        help="Path to GRID data directory")
    parser.add_argument("--speaker", type=str, default="s1",
                        help="Speaker folder name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: grid_path/../cache_SPEAKER)")
    parser.add_argument("--method", type=str, default="mediapipe",
                        choices=["mediapipe", "opencv_dnn", "fixed"],
                        help="Lip extraction method")
    parser.add_argument("--no-skip", action="store_true",
                        help="Reprocess existing files")
    args = parser.parse_args()

    grid_path = Path(args.grid_path)
    video_dir = grid_path / args.speaker

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = grid_path.parent / f"cache_{args.speaker}"

    print(f"Method:  {args.method}")
    print(f"Input:   {video_dir}")
    print(f"Output:  {output_dir}")
    print(f"Target:  {TARGET_W}×{TARGET_H}")
    print()

    if args.method == "mediapipe":
        extractor = MediaPipeLipExtractor(margin=0.4)
    elif args.method == "opencv_dnn":
        extractor = OpenCVDNNLipExtractor(margin=0.15)
    elif args.method == "fixed":
        extractor = FixedCropLipExtractor()

    process_speaker(video_dir, output_dir, extractor, skip_existing=not args.no_skip)
    extractor.close()


if __name__ == "__main__":
    main()
