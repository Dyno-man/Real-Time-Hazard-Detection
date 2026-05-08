"""Run the best YOLO model on a video with boxes and unique-object tallies."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_MODEL = Path("best.pt")
DEFAULT_OUTPUT_DIR = Path("runs/video_tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a YOLO model on video, draw detections, and tally newly tracked objects."
    )
    parser.add_argument("video", type=Path, help="Path to the input video.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="YOLO weights to use. Default: ./best.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold. Default: 0.25")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold. Default: 0.45")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size. Default: 640")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config. Default: bytetrack.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Default: runs/video_tests/<input>_detected.mp4",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write an annotated video file.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a live preview window.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")


def make_output_path(video_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{video_path.stem}_detected.mp4"


def open_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def draw_tally(frame, counts: Counter[str], total_seen: int) -> None:
    lines = [f"Unique objects: {total_seen}"]
    lines.extend(f"{name}: {count}" for name, count in counts.most_common())

    if not lines:
        return

    padding = 10
    line_height = 24
    panel_width = max(260, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for line in lines) + 24)
    panel_height = padding * 2 + line_height * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 10 + padding + 16
    for line in lines:
        cv2.putText(frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_height


def main() -> None:
    args = parse_args()
    require_file(args.video, "Video")
    require_file(args.model, "Model")

    if args.no_save and args.no_display:
        raise ValueError("Nothing to do: remove --no-save or --no-display.")

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = make_output_path(args.video, args.output)
    writer = None if args.no_save else open_writer(output_path, fps, width, height)

    seen_track_ids: set[int] = set()
    counts: Counter[str] = Counter()
    frame_number = 0

    print(f"Model: {args.model}")
    print(f"Video: {args.video}")
    if writer is not None:
        print(f"Saving annotated video to: {output_path}")
    print("Press q in the preview window to stop early.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            results = model.track(
                frame,
                persist=True,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                tracker=args.tracker,
                verbose=False,
            )
            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                for track_id, class_id in zip(track_ids, class_ids):
                    if track_id in seen_track_ids:
                        continue
                    seen_track_ids.add(track_id)
                    counts[model.names[int(class_id)]] += 1

            annotated = result.plot()
            draw_tally(annotated, counts, len(seen_track_ids))

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("YOLO video test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print("\nFinal unique-object tally")
    print(f"Total: {len(seen_track_ids)}")
    for name, count in counts.most_common():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
