import argparse

from ultralytics import YOLO
import easyocr
import cv2

# Load ocr model
reader = easyocr.Reader(["en"], gpu=True)

def filter_text(height_threshold, boxes):
    filtered = []

    for box in boxes:
        height = box[0][2][1] - box[0][1][1]

        if height > height_threshold:
            filtered.append(box[1])
    
    return filtered

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=2,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

def plate_processing(frame, coordinates, height_threshold):
    x1, y1, x2, y2 = coordinates
    min_y = abs(y2-y1) * height_threshold

    crop_plate = frame[y1:y1+int(min_y), x1:x2]

    gray = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)
    
    return gray

def plate_detection(model, frame, height_threshold):
    # Detect plate -> plates[List]
    plates = model(frame, conf=0.7)[0]

    # Image processing for detected plate -> processed_plate[List]
    for plate in plates:
        x1, y1, x2, y2 = [int(i) for i in plate.boxes.xyxy[0]]
        
        processed_plate = plate_processing(frame, (x1, y1, x2, y2), 0.5)

        # Extract text from processed image -> text_plate[List]
        text_plate = reader.readtext(processed_plate, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", detail=0)
        """
        filter_threshold = (y2 - y1) * height_threshold

        filtered_text = filter_text(filter_threshold, text_plate)
        """

        # Render text on frame -> new_frame
        if len(text_plate) > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            draw_text(frame, "".join(text_plate), text_color=(255, 255, 255), text_color_bg=(255, 0, 0), pos=(x1, y1-20))
            #cv2.putText(frame, "".join(filtered_text), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            draw_text(frame, "Number not detected", text_color=(255, 255, 255), text_color_bg=(0, 0, 255), pos=(x1, y1-20))
            #cv2.putText(frame, "Number not detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
def load_video(args):
    # Load video
    cap = cv2.VideoCapture(args.file_path)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        return None
    
    return cap

def main(args):
    # Load fine tuned yolo model for license plate
    plate_detection_model = YOLO(args.model_path)

    if args.mode == "video":
        cap = load_video(args)    
        # Read frame -> frame
        while(cap.isOpened()):
            ret, frame = cap.read()
            plate_detection(plate_detection_model, frame, 0.5)

            cv2.imshow('License Plate Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('License Plate Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.mode == "image":
        frame = cv2.imread(args.file_path)
        plate_detection(plate_detection_model, frame, 0.5)

        cv2.imshow('License Plate Detection', frame)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("Please provide correct arguments")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plate Number Detection"
    )

    parser.add_argument("--mode", type=str, help="Mode to run", choices=["image", "video"], default="image")
    parser.add_argument("--file_path", type=str, default=None, help="Path to video file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to object detection (YOLO) model file")
    args = parser.parse_args()

    main(args)