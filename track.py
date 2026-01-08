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
	parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
	parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
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
		frame = result.plot()  # Render boxes, masks, and IDs
		cv2.imshow(args.window, frame)
		key = cv2.waitKey(1) & 0xFF
		if key in (ord("q"), 27):  # q or ESC to exit
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

