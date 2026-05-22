import cv2
import numpy as np

def extract_features(face_roi, face_gray):
    """
    Extracts 25 numerical features from a face image.
    These features feed into our ML models.
    """

    features = {}

    # ══════════════════════════════════════════════
    # GROUP 1 — LAB COLOR FEATURES (5 features)
    # LAB is the most accurate color space for skin
    # ══════════════════════════════════════════════
    lab     = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    features['l_mean']  = float(np.mean(L))       # Lightness
    features['l_std']   = float(np.std(L))        # Lightness variation
    features['a_mean']  = float(np.mean(A))       # Red-Green axis
    features['b_mean']  = float(np.mean(B))       # Blue-Yellow axis
    features['ab_ratio']= float(np.mean(A) / (np.mean(B) + 1e-5))

    # ══════════════════════════════════════════════
    # GROUP 2 — HSV FEATURES (4 features)
    # Hue Saturation Value — good for oily detection
    # ══════════════════════════════════════════════
    hsv     = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    features['saturation_mean'] = float(np.mean(S))
    features['saturation_std']  = float(np.std(S))
    features['value_mean']      = float(np.mean(V))
    features['value_std']       = float(np.std(V))

    # ══════════════════════════════════════════════
    # GROUP 3 — TEXTURE FEATURES (5 features)
    # Texture tells us oily vs dry vs normal
    # ══════════════════════════════════════════════
    laplacian       = cv2.Laplacian(face_gray, cv2.CV_64F)
    features['texture_var']  = float(np.var(laplacian))
    features['texture_mean'] = float(np.mean(np.abs(laplacian)))

    sobelx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    features['gradient_mean'] = float(np.mean(gradient))
    features['gradient_std']  = float(np.std(gradient))

    blur   = cv2.GaussianBlur(face_gray, (5,5), 0)
    diff   = cv2.absdiff(face_gray, blur)
    features['roughness'] = float(np.mean(diff))

    # ══════════════════════════════════════════════
    # GROUP 4 — HIGHLIGHT FEATURES (3 features)
    # Highlights = oily/shiny skin indicator
    # ══════════════════════════════════════════════
    _, high220 = cv2.threshold(face_gray, 220, 255, cv2.THRESH_BINARY)
    _, high200 = cv2.threshold(face_gray, 200, 255, cv2.THRESH_BINARY)
    _, high180 = cv2.threshold(face_gray, 180, 255, cv2.THRESH_BINARY)

    total = face_gray.size
    features['highlight_220'] = float(np.sum(high220 > 0) / total)
    features['highlight_200'] = float(np.sum(high200 > 0) / total)
    features['highlight_180'] = float(np.sum(high180 > 0) / total)

    # ══════════════════════════════════════════════
    # GROUP 5 — DARK SPOT FEATURES (4 features)
    # ══════════════════════════════════════════════
    h, w    = face_gray.shape
    roi     = face_gray[int(h*0.2):int(h*0.9), int(w*0.1):int(w*0.9)]

    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced= clahe.apply(roi)

    mean_v  = np.mean(enhanced)
    std_v   = np.std(enhanced)
    thresh  = max(0, mean_v - 1.5 * std_v)

    _, dark_mask = cv2.threshold(
        enhanced, int(thresh), 255, cv2.THRESH_BINARY_INV
    )
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    spot_cnts = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15 or area > 600:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        if 4 * np.pi * area / (perimeter**2) > 0.3:
            spot_cnts.append(c)

    features['spot_count']    = float(len(spot_cnts))
    features['spot_density']  = float(len(spot_cnts) / (roi.size + 1e-5))
    features['dark_mean']     = float(mean_v)
    features['dark_std']      = float(std_v)

    # ══════════════════════════════════════════════
    # GROUP 6 — EYE BAG FEATURES (4 features)
    # ══════════════════════════════════════════════
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    eyes = eye_cascade.detectMultiScale(
        face_gray, 1.1, 5, minSize=(20,20)
    )

    bag_darkness  = []
    bag_gradients = []

    for (ex, ey, ew, eh) in eyes:
        y1 = min(ey + eh, face_gray.shape[0])
        y2 = min(ey + eh + int(eh*0.8), face_gray.shape[0])
        x1 = ex + int(ew*0.1)
        x2 = ex + ew - int(ew*0.1)
        if y2 <= y1 or x2 <= x1:
            continue
        region = face_gray[y1:y2, x1:x2]
        if region.size == 0:
            continue
        bag_darkness.append(255 - np.mean(region))
        sx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        bag_gradients.append(np.mean(np.sqrt(sx**2 + sy**2)))

    features['eye_darkness']  = float(np.mean(bag_darkness))  if bag_darkness  else 0.0
    features['eye_gradient']  = float(np.mean(bag_gradients)) if bag_gradients else 0.0
    features['eye_bag_score'] = float(
        features['eye_darkness'] * 0.6 + features['eye_gradient'] * 0.4
    )
    features['eyes_detected'] = float(len(eyes))

    print(f"[Features] Extracted {len(features)} features")
    return features


def features_to_array(features):
    """Convert features dict to numpy array for ML model"""
    keys = sorted(features.keys())
    return np.array([features[k] for k in keys]).reshape(1, -1)