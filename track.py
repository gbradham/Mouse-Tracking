import argparse
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Live tracking viewer")
	parser.add_argument("--model", type=str, default="./model.pt", help="Path to YOLO model")
	parser.add_argument(
		"--source",
		type=str,
		default="0",
		help="Video source: webcam index (e.g. 0) or path/URL",
	)
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
	parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
	parser.add_argument("--window", type=str, default="Live Tracking", help="Display window title")
	return parser.parse_args()


def main():
	args = parse_args()
	# Use int for webcam index when possible
	source: str | int
	try:
		source = int(args.source)
	except ValueError:
		source = args.source

	model = YOLO(args.model)
	stream = model.track(
		source=source,
		conf=args.conf,
		iou=args.iou,
		stream=True,
		show=False,
		verbose=False,
		persist=True,
	)

	for result in stream:
		frame = result.orig_img.copy() if hasattr(result, "orig_img") else None
		if frame is not None:
			# Draw a dot at the center of each bounding box
			if hasattr(result, "boxes") and result.boxes is not None:
				for box in result.boxes.xyxy:
					x1, y1, x2, y2 = box[:4]
					cx = int((x1 + x2) / 2)
					cy = int((y1 + y2) / 2)
					cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot
			cv2.imshow(args.window, frame)
			key = cv2.waitKey(1) & 0xFF
			if key in (ord("q"), 27):  # q or ESC to exit
				break

	cv2.destroyAllWindows()
	# Print the lowest confidence seen so far
	if hasattr(main, "min_conf"):
		print(f"Lowest confidence seen: {main.min_conf:.4f}")

if __name__ == "__main__":
	main()

