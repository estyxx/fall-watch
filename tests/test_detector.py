"""
Test the fall detector using your Mac webcam.
- Green box + label = standing/sitting (ok)
- Red box + label = ON FLOOR (would trigger alert)

Press Q to quit.
"""

import cv2

from fall_watch.detector import _is_lying_down, load_model


def main() -> None:
    model = load_model()
    cap = cv2.VideoCapture(0)  # 0 = built-in webcam

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("✅ Webcam open — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []

            for i, person_kps in enumerate(keypoints_data):
                on_floor = _is_lying_down(person_kps, frame.shape[0])
                color = (0, 0, 255) if on_floor else (0, 200, 0)
                label = "⚠ ON FLOOR" if on_floor else "ok"

                # Draw bounding box
                if i < len(boxes):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                    )

                # Draw keypoints
                for kp in person_kps:
                    x, y, conf = kp
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        cv2.imshow("fall-watch — webcam test (Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
