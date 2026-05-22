import cv2
import os

# This automatically finds the correct path on YOUR computer
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def detect_face(image_path):

    print("=== DEBUG ===")
    print("Image path:", image_path)
    print("Image exists:", os.path.exists(image_path))
    print("Cascade:", CASCADE_PATH)

    result = {
        'face_found': False,
        'face_count': 0,
        'processed_image_url': None,
        'message': ''
    }

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        result['message'] = 'Cannot read image. Use JPG or PNG only.'
        print("ERROR: Cannot read image")
        return result

    print("Image shape:", image.shape)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast so face detects better
    gray = cv2.equalizeHist(gray)

    # Load cascade with exact path
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():
        result['message'] = 'Cascade file failed to load.'
        print("ERROR: Cascade is empty")
        return result

    # Detect faces with relaxed settings
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Total faces found:", len(faces))

    if len(faces) == 0:
        result['message'] = 'No face detected. Use a clear front-facing photo.'
        return result

    # Draw green box around face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (46, 125, 94),
            3
        )
        cv2.putText(
            image,
            'Face Detected',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (46, 125, 94),
            2
        )

    # Save processed image
    save_folder = os.path.join(BASE_DIR, 'media', 'processed')
    os.makedirs(save_folder, exist_ok=True)

    filename = 'detected_' + os.path.basename(image_path)
    save_path = os.path.join(save_folder, filename)

    success = cv2.imwrite(save_path, image)
    print("Image saved:", success)
    print("Saved path:", save_path)

    result['face_found'] = True
    result['face_count'] = len(faces)
    result['processed_image_url'] = '/media/processed/' + filename
    result['message'] = str(len(faces)) + ' face detected successfully!'

    return result