import argparse
import imutils
from ultralytics import YOLO
import easyocr
import cv2

#reader = easyocr.Reader(["en"], gpu=True)
def filter_text(
    filter_threshold,
    boxes
):
    """
    Filters a list of text boxes based on a filter threshold.

    Args:
        filter_threshold (float): The threshold value used to determine which boxes to filter.
        boxes (List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], str]]): A list of text boxes, where each box is represented as a tuple containing the coordinates of the top-left and bottom-right corners of the box and the text associated with the box.

    Returns:
        List[str]: A list of the filtered text boxes.

    """
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
    """
    Check if the given text is valid by verifying if its length is between 7 and 9 characters (inclusive).
    
    Parameters:
        text (str): The text to be checked.
    
    Returns:
        bool: True if the text is valid, False otherwise.
    """
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
    """
    Draws text on an image at a specified position with a given font, scale, thickness, and colors.

    Parameters:
        img (numpy.ndarray): The image on which the text will be drawn.
        text (str): The text to be drawn.
        font (int, optional): The font type. Defaults to cv2.FONT_HERSHEY_PLAIN.
        pos (tuple, optional): The position where the text will be drawn. Defaults to (0, 0).
        font_scale (float, optional): The font scale. Defaults to 1.5.
        font_thickness (int, optional): The font thickness. Defaults to 2.
        text_color (tuple, optional): The color of the text. Defaults to (0, 255, 0).
        text_color_bg (tuple, optional): The background color of the text. Defaults to (0, 0, 0).

    Returns:
        None
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + int(font_scale) - 1), font, font_scale, text_color, font_thickness)

def plate_processing(
    frame,
    coordinates,
    height_threshold
):
    """
    Processes a license plate image by cropping, converting to grayscale, and applying binary thresholding.

    Args:
        frame (numpy.ndarray): The input image containing the license plate.
        coordinates (tuple): The coordinates of the license plate in the form (x1, y1, x2, y2).
        height_threshold (float): The threshold for adjusting the height of the cropped license plate.

    Returns:
        numpy.ndarray: The processed license plate image.
    """
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
    frame,
    reader
):
    """
    Detects plates in a frame and performs image processing on each detected plate.
    
    Args:
        args (object): An object containing various arguments.
        model (object): The model used for plate detection.
        frame (numpy.ndarray): The frame containing the plates.
        reader (object): The reader used for extracting text from processed plates.
        
    Returns:
        None
    """
    # Detect plate -> plates[List]
    plates = model(frame, conf=0.7)[0]

    # Image processing for detected plate -> processed_plate[List]
    for plate in plates:
        x1, y1, x2, y2 = [int(i) for i in plate.boxes.xyxy[0]]
        
        processed_plate = plate_processing(frame, (x1, y1, x2, y2), args.height_threshold)
        
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
    # Load ocr model
    reader = easyocr.Reader(
        ['en'],
        user_network_directory="text_recognition_model",
        model_storage_directory="text_recognition_model",
        recog_network='finetuned_model'
    )

    # Load fine tuned yolo model for license plate
    plate_detection_model = YOLO(args.plate_detection_model_path)

    if args.mode == "video":
        # Load video
        cap = cv2.VideoCapture(args.file_path)
        
        if cap.isOpened() == False:
            print("Error opening video stream or file")
        # Read frame -> frame
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            frame = imutils.resize(frame, width=720)
            plate_detection(args, plate_detection_model, frame, reader)

            cv2.imshow('License Plate Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('License Plate Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.mode == "image":
        # Load image
        frame = cv2.imread(args.file_path)
        plate_detection(args, plate_detection_model, frame, reader)

        cv2.imshow('License Plate Detection', frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == "webcam":
        # Load webcam
        cap = cv2.VideoCapture(args.webcam_id)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        if cap.isOpened() == False:
            print("Webcam not found")

        # Read frame -> frame
        while(cap.isOpened()):
            ret, frame = cap.read()
            plate_detection(args, plate_detection_model, frame, reader)

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
    parser.add_argument("--plate_detection_model_path", type=str, default="object_detection_model/plate_detection.pt", help="Path to object detection (YOLO) model file")
    parser.add_argument("--text_recognition_model_path", type=str, default="text_recognition_model", help="Path to text recognition model directory")
    parser.add_argument("--height_threshold", type=float, default=0.3, help="Height threshold for filtering text detection")
    args = parser.parse_args()

    main(args)