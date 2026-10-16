"""
Utility script to generate a short synthetic traffic video for testing detection,
ByteTrack tracking, and virtual line crossing without external media.
"""

from pathlib import Path
import cv2
import numpy as np

from config import DATA_DIR


def generate_sample_traffic_video(output_path: Path = DATA_DIR / "sample_traffic.mp4", num_frames: int = 50) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = 640, 360
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Road background
    for i in range(num_frames):
        frame = np.full((h, w, 3), 45, dtype=np.uint8)

        # Road boundaries & lane dividers
        cv2.line(frame, (100, 0), (100, h), (255, 255, 255), 2)
        cv2.line(frame, (540, 0), (540, h), (255, 255, 255), 2)
        cv2.line(frame, (320, 0), (320, h), (200, 200, 0), 2)

        # Vehicle 1 (Car): moving downwards crossing line at Y=180
        y_car = int(30 + (i / num_frames) * 280)
        cv2.rectangle(frame, (180, y_car), (260, y_car + 70), (0, 200, 50), -1)
        cv2.rectangle(frame, (190, y_car + 10), (250, y_car + 40), (20, 50, 100), -1)

        # Vehicle 2 (Motorcycle): moving upwards
        y_moto = int(300 - (i / num_frames) * 260)
        cv2.rectangle(frame, (380, y_moto), (410, y_moto + 50), (200, 50, 200), -1)

        out.write(frame)

    out.release()
    print(f"Sample traffic video generated at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_traffic_video()
