import cv2
import numpy as np
import os
from home.ml.feature_extractor import extract_features, features_to_array
from home.ml.ml_models import (load_models, predict_skin_type,
                                predict_dark_spots, predict_eye_bags)

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load models when server starts
load_models()


def detect_skin_tone(face_roi):
    lab     = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_f     = L.astype(np.float32)
    B_f     = B.astype(np.float32) - 128
    with np.errstate(divide='ignore', invalid='ignore'):
        ita_map = np.degrees(np.arctan2(L_f - 50, B_f))
    ita = np.mean(ita_map[np.isfinite(ita_map)])
    if ita > 55:   return 'Very Fair'
    elif ita > 41: return 'Fair'
    elif ita > 28: return 'Medium'
    elif ita > 10: return 'Olive'
    elif ita > -30:return 'Brown'
    else:          return 'Dark'


def analyze_skin(image_path):
    print("\n=== SKIN ANALYSIS START ===")

    result = {
        'face_found'         : False,
        'skin_type'          : '',
        'skin_type_confidence': 0,
        'dark_spots'         : '',
        'dark_spot_count'    : 0,
        'eye_bags'           : '',
        'skin_quality'       : '',
        'skin_quality_score' : 0,
        'skin_tone'          : '',
        'processed_image_url': None,
        'message'            : ''
    }

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        result['message'] = 'Cannot read image.'
        return result

    # Resize for consistency
    h0, w0 = image.shape[:2]
    if max(h0, w0) > 800:
        scale = 800 / max(h0, w0)
        image = cv2.resize(image, (int(w0*scale), int(h0*scale)))

    print("Image loaded:", image.shape)

    # Face detection
    gray         = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq      = cv2.equalizeHist(gray)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    faces = face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.05,
        minNeighbors=4, minSize=(60,60)
    )
    print("Faces found:", len(faces))

    if len(faces) == 0:
        result['message'] = 'No face detected. Use a clear front-facing photo.'
        return result

    result['face_found'] = True
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_roi    = image[y:y+h, x:x+w]
    face_gray   = gray[y:y+h, x:x+w]

    # ── Extract all 25 features ─────────────────────────────
    features      = extract_features(face_roi, face_gray)
    features_arr  = features_to_array(features)

    # ── ML Predictions ──────────────────────────────────────
    skin_type, confidence = predict_skin_type(features_arr)

    dark_spots = predict_dark_spots(
        features['spot_count'],
        features['spot_density'],
        features['dark_mean'],
        features['dark_std']
    )

    eye_bags = predict_eye_bags(
        features['eye_darkness'],
        features['eye_gradient'],
        features['eye_bag_score'],
        features['eyes_detected']
    )

    skin_tone = detect_skin_tone(face_roi)
    spot_count = int(features['spot_count'])

    print(f"Skin Type  : {skin_type} ({confidence}%)")
    print(f"Dark Spots : {dark_spots} ({spot_count} spots)")
    print(f"Eye Bags   : {eye_bags}")
    print(f"Skin Tone  : {skin_tone}")

    # ── Quality score ────────────────────────────────────────
    score = 100
    score -= {'None':0,'Mild':12,'Moderate':25,'Severe':40}.get(dark_spots, 0)
    score -= {'None':0,'Mild':8,'Moderate':18,'Severe':28,'Not Detected':0}.get(eye_bags, 0)
    score -= {'Normal':0,'Oily':8,'Dry':8,'Combination':5,'Sensitive':10}.get(skin_type, 0)
    score  = max(0, min(100, score))

    if score >= 85:   quality = 'Excellent'
    elif score >= 70: quality = 'Good'
    elif score >= 50: quality = 'Fair'
    else:             quality = 'Needs Care'

    result.update({
        'skin_type'           : skin_type,
        'skin_type_confidence': confidence,
        'dark_spots'          : dark_spots,
        'dark_spot_count'     : spot_count,
        'eye_bags'            : eye_bags,
        'skin_tone'           : skin_tone,
        'skin_quality'        : quality,
        'skin_quality_score'  : score,
    })

    # ── Draw on image ─────────────────────────────────────────
    cv2.rectangle(image, (x,y), (x+w, y+h), (46,125,94), 3)

    labels = [
        f"Type : {skin_type} ({confidence}%)",
        f"Tone : {skin_tone}",
        f"Spots: {dark_spots} ({spot_count})",
        f"Eyes : {eye_bags}",
        f"Score: {score}/100",
    ]
    panel_x = x + w + 8
    if panel_x + 200 > image.shape[1]:
        panel_x = max(0, x - 210)

    for i, lbl in enumerate(labels):
        cv2.putText(image, lbl,
                    (panel_x, y + 22 + i*26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (30, 100, 200), 2)

    # Save
    save_folder = os.path.join(BASE_DIR, 'media', 'processed')
    os.makedirs(save_folder, exist_ok=True)
    filename  = 'analyzed_' + os.path.basename(image_path)
    save_path = os.path.join(save_folder, filename)
    cv2.imwrite(save_path, image)

    result['processed_image_url'] = '/media/processed/' + filename
    result['message']             = 'Skin analysis complete!'

    print(f"Score: {score} Quality: {quality}")
    print("=== SKIN ANALYSIS END ===\n")
    return result