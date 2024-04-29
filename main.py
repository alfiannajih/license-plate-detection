import argparse
import imutils
from ultralytics import YOLO
import easyocr
import cv2

# Load ocr model
reader = easyocr.Reader(
    ['en'],
    user_network_directory="ocr_model",
    model_storage_directory="ocr_model",
    recog_network='finetuned_model')

#reader = easyocr.Reader(["en"], gpu=True)

def filter_text(
    filter_threshold,
    boxes
):
    filtered = []
    box_size = []
    max_size = 0

    for box in boxes:
        width = box[0][2][0] - box[0][0][0]
        height = box[0][2][1] - box[0][1][1]
        current_size = width*height

        if max_size < current_size:
            max_size = current_size

        box_size.append((current_size, box[1]))
    
    for size in box_size:
        if size[0] > max_size * filter_threshold:
            filtered.append(size[1])

    return filtered

def text_is_valid(
    text
):
    if len(text) >= 7 and len(text) <= 9:
        return True
    return False

def draw_text(
    img,
    text,
    font=cv2.FONT_HERSHEY_PLAIN,
    pos=(0, 0),
    font_scale=1.5,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0)
):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + int(font_scale) - 1), font, font_scale, text_color, font_thickness)

def plate_processing(
    frame,
    coordinates,
    height_threshold=0.25
):
    x1, y1, x2, y2 = coordinates
    #threshold = int(((x2 - x1)*tolerance)/2)
    min_y = abs(y2-y1) * height_threshold

    crop_plate = frame[y1:y2-int(min_y), x1:x2]
    gray = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    return thresh

def plate_detection(
    args,
    model,
    frame
):
    # Detect plate -> plates[List]
    plates = model(frame, conf=0.7)[0]

    # Image processing for detected plate -> processed_plate[List]
    for plate in plates:
        x1, y1, x2, y2 = [int(i) for i in plate.boxes.xyxy[0]]
        
        processed_plate = plate_processing(frame, (x1, y1, x2, y2))
        
        # Extract text from processed image -> text_plate[List]
        #text_plate = reader.readtext(processed_plate, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        final_text = reader.recognize(processed_plate)[0][1].upper()
        #filter_threshold = ((x2 - x1)*(y2 - y1)) * args.height_threshold

        #filtered_text = filter_text(args.height_threshold, text_plate)
        #final_text = "".join(filtered_text)

        if text_is_valid(final_text):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            draw_text(frame, final_text, text_color=(255, 255, 255), text_color_bg=(255, 0, 0), pos=(x1, y1-20))
            #cv2.putText(frame, "".join(filtered_text), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            draw_text(frame, "Number not detected", text_color=(255, 255, 255), text_color_bg=(0, 0, 255), pos=(x1, y1-20))
            #cv2.putText(frame, "Number not detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def main(args):
    # Load fine tuned yolo model for license plate
    plate_detection_model = YOLO(args.model_path)

    if args.mode == "video":
        cap = cv2.VideoCapture(args.file_path)
        
        if cap.isOpened() == False:
            print("Error opening video stream or file")
        # Read frame -> frame
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            frame = imutils.resize(frame, width=720)
            plate_detection(args, plate_detection_model, frame)

            cv2.imshow('License Plate Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('License Plate Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.mode == "image":
        frame = cv2.imread(args.file_path)
        plate_detection(args, plate_detection_model, frame)

        cv2.imshow('License Plate Detection', frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(args.webcam_id)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        if cap.isOpened() == False:
            print("Webcam not found")

        # Read frame -> frame
        while(cap.isOpened()):
            ret, frame = cap.read()
            plate_detection(args, plate_detection_model, frame)

            cv2.imshow('License Plate Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('License Plate Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Please specify mode: image, video or webcam")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plate Number Detection"
    )

    parser.add_argument("--mode", type=str, help="Mode to run", choices=["image", "video", "webcam"], default="image")
    parser.add_argument("--webcam_id", type=int, default=0, help="ID of the webcam")
    parser.add_argument("--file_path", type=str, default=None, help="Path to image file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to object detection (YOLO) model file")
    parser.add_argument("--height_threshold", type=float, default=0.5, help="Height threshold for filtering text detection")
    args = parser.parse_args()

    main(args)