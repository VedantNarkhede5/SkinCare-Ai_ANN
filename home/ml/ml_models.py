import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'home', 'ml', 'saved_model')
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  GENERATE SYNTHETIC TRAINING DATA
#  Based on real dermatology feature ranges
# ══════════════════════════════════════════════════════════════
def generate_training_data(n_samples=2000):
    """
    Generate realistic synthetic training data
    based on known dermatology feature ranges.
    Each sample = 25 features matching feature_extractor.py
    """
    np.random.seed(42)

    X_skin_type  = []
    y_skin_type  = []
    X_dark_spots = []
    y_dark_spots = []
    X_eye_bags   = []
    y_eye_bags   = []

    samples_per_class = n_samples // 4

    # ── OILY SKIN ────────────────────────────────────────────
    for _ in range(samples_per_class):
        f = generate_sample(
            l_mean         = np.random.uniform(155, 200),
            l_std          = np.random.uniform(8,  20),
            a_mean         = np.random.uniform(128, 145),
            b_mean         = np.random.uniform(128, 145),
            saturation_mean= np.random.uniform(85, 140),
            saturation_std = np.random.uniform(15, 35),
            value_mean     = np.random.uniform(155, 210),
            value_std      = np.random.uniform(10, 25),
            texture_var    = np.random.uniform(100, 600),
            texture_mean   = np.random.uniform(5,  18),
            gradient_mean  = np.random.uniform(8,  20),
            gradient_std   = np.random.uniform(5,  15),
            roughness      = np.random.uniform(3,  10),
            highlight_220  = np.random.uniform(0.05, 0.18),
            highlight_200  = np.random.uniform(0.10, 0.28),
            highlight_180  = np.random.uniform(0.18, 0.40),
            spot_count     = np.random.uniform(0, 6),
            spot_density   = np.random.uniform(0, 0.0002),
            dark_mean      = np.random.uniform(130, 170),
            dark_std       = np.random.uniform(15, 30),
            eye_darkness   = np.random.uniform(20, 60),
            eye_gradient   = np.random.uniform(10, 30),
            eye_bag_score  = np.random.uniform(15, 48),
            eyes_detected  = np.random.choice([0, 1, 2]),
            ab_ratio       = np.random.uniform(0.9, 1.1),
        )
        X_skin_type.append(f)
        y_skin_type.append('Oily')

    # ── DRY SKIN ─────────────────────────────────────────────
    for _ in range(samples_per_class):
        f = generate_sample(
            l_mean         = np.random.uniform(80, 125),
            l_std          = np.random.uniform(18, 40),
            a_mean         = np.random.uniform(125, 138),
            b_mean         = np.random.uniform(120, 135),
            saturation_mean= np.random.uniform(20, 65),
            saturation_std = np.random.uniform(10, 28),
            value_mean     = np.random.uniform(80, 128),
            value_std      = np.random.uniform(18, 40),
            texture_var    = np.random.uniform(900, 2500),
            texture_mean   = np.random.uniform(18, 45),
            gradient_mean  = np.random.uniform(20, 45),
            gradient_std   = np.random.uniform(15, 35),
            roughness      = np.random.uniform(14, 30),
            highlight_220  = np.random.uniform(0.0, 0.02),
            highlight_200  = np.random.uniform(0.0, 0.05),
            highlight_180  = np.random.uniform(0.01, 0.08),
            spot_count     = np.random.uniform(2, 12),
            spot_density   = np.random.uniform(0.0001, 0.0005),
            dark_mean      = np.random.uniform(80, 120),
            dark_std       = np.random.uniform(22, 42),
            eye_darkness   = np.random.uniform(35, 80),
            eye_gradient   = np.random.uniform(18, 45),
            eye_bag_score  = np.random.uniform(30, 68),
            eyes_detected  = np.random.choice([0, 1, 2]),
            ab_ratio       = np.random.uniform(0.85, 1.05),
        )
        X_skin_type.append(f)
        y_skin_type.append('Dry')

    # ── NORMAL SKIN ──────────────────────────────────────────
    for _ in range(samples_per_class):
        f = generate_sample(
            l_mean         = np.random.uniform(125, 158),
            l_std          = np.random.uniform(10, 22),
            a_mean         = np.random.uniform(126, 136),
            b_mean         = np.random.uniform(125, 138),
            saturation_mean= np.random.uniform(55, 88),
            saturation_std = np.random.uniform(10, 22),
            value_mean     = np.random.uniform(125, 158),
            value_std      = np.random.uniform(10, 22),
            texture_var    = np.random.uniform(400, 900),
            texture_mean   = np.random.uniform(10, 22),
            gradient_mean  = np.random.uniform(12, 25),
            gradient_std   = np.random.uniform(8,  18),
            roughness      = np.random.uniform(7,  15),
            highlight_220  = np.random.uniform(0.01, 0.04),
            highlight_200  = np.random.uniform(0.02, 0.08),
            highlight_180  = np.random.uniform(0.05, 0.15),
            spot_count     = np.random.uniform(0, 4),
            spot_density   = np.random.uniform(0, 0.00015),
            dark_mean      = np.random.uniform(118, 148),
            dark_std       = np.random.uniform(12, 22),
            eye_darkness   = np.random.uniform(15, 45),
            eye_gradient   = np.random.uniform(8,  25),
            eye_bag_score  = np.random.uniform(10, 38),
            eyes_detected  = np.random.choice([0, 1, 2]),
            ab_ratio       = np.random.uniform(0.92, 1.08),
        )
        X_skin_type.append(f)
        y_skin_type.append('Normal')

    # ── COMBINATION SKIN ─────────────────────────────────────
    for _ in range(samples_per_class):
        f = generate_sample(
            l_mean         = np.random.uniform(138, 168),
            l_std          = np.random.uniform(14, 28),
            a_mean         = np.random.uniform(127, 140),
            b_mean         = np.random.uniform(126, 140),
            saturation_mean= np.random.uniform(68, 105),
            saturation_std = np.random.uniform(18, 35),
            value_mean     = np.random.uniform(138, 168),
            value_std      = np.random.uniform(14, 28),
            texture_var    = np.random.uniform(550, 1100),
            texture_mean   = np.random.uniform(14, 28),
            gradient_mean  = np.random.uniform(15, 30),
            gradient_std   = np.random.uniform(10, 22),
            roughness      = np.random.uniform(9,  18),
            highlight_220  = np.random.uniform(0.025, 0.07),
            highlight_200  = np.random.uniform(0.05,  0.14),
            highlight_180  = np.random.uniform(0.10,  0.22),
            spot_count     = np.random.uniform(1, 7),
            spot_density   = np.random.uniform(0.00005, 0.00025),
            dark_mean      = np.random.uniform(122, 152),
            dark_std       = np.random.uniform(16, 28),
            eye_darkness   = np.random.uniform(22, 58),
            eye_gradient   = np.random.uniform(12, 32),
            eye_bag_score  = np.random.uniform(18, 50),
            eyes_detected  = np.random.choice([0, 1, 2]),
            ab_ratio       = np.random.uniform(0.90, 1.10),
        )
        X_skin_type.append(f)
        y_skin_type.append('Combination')

    # ── DARK SPOT LEVELS ─────────────────────────────────────
    for count, label in [
        ((0, 1),   'None'),
        ((2, 5),   'Mild'),
        ((6, 14),  'Moderate'),
        ((15, 40), 'Severe'),
    ]:
        for _ in range(500):
            spots  = np.random.uniform(count[0], count[1])
            density= spots / 50000
            X_dark_spots.append([
                spots,
                density,
                np.random.uniform(80, 160),
                np.random.uniform(10, 40),
            ])
            y_dark_spots.append(label)

    # ── EYE BAG LEVELS ───────────────────────────────────────
    for score_range, label in [
        ((5,  38), 'None'),
        ((39, 62), 'Mild'),
        ((63, 88), 'Moderate'),
        ((89, 130),'Severe'),
    ]:
        for _ in range(500):
            s = np.random.uniform(score_range[0], score_range[1])
            X_eye_bags.append([
                s * 0.6 + np.random.normal(0, 3),
                s * 0.4 + np.random.normal(0, 2),
                s        + np.random.normal(0, 4),
                float(np.random.choice([0, 1, 2])),
            ])
            y_eye_bags.append(label)

    return (
        np.array(X_skin_type),  np.array(y_skin_type),
        np.array(X_dark_spots), np.array(y_dark_spots),
        np.array(X_eye_bags),   np.array(y_eye_bags),
    )


def generate_sample(**kwargs):
    """Build ordered feature array matching feature_extractor keys"""
    keys = [
        'ab_ratio','a_mean','b_mean',
        'dark_mean','dark_std',
        'eye_bag_score','eye_darkness','eye_gradient','eyes_detected',
        'gradient_mean','gradient_std',
        'highlight_180','highlight_200','highlight_220',
        'l_mean','l_std',
        'roughness',
        'saturation_mean','saturation_std',
        'spot_count','spot_density',
        'texture_mean','texture_var',
        'value_mean','value_std',
    ]
    return [kwargs.get(k, 0.0) for k in keys]


# ══════════════════════════════════════════════════════════════
#  TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════
def train_all_models():
    print("\n=== GENERATING TRAINING DATA ===")
    (X_skin, y_skin,
     X_spots, y_spots,
     X_eyes,  y_eyes) = generate_training_data(2000)

    print(f"Skin type samples : {len(X_skin)}")
    print(f"Dark spot samples : {len(X_spots)}")
    print(f"Eye bag samples   : {len(X_eyes)}")

    # ── 1. Skin Type — SVM ───────────────────────────────────
    print("\n--- Training Skin Type SVM ---")
    skin_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10,
                       gamma='scale', probability=True))
    ])
    skin_scores = cross_val_score(skin_pipeline, X_skin, y_skin, cv=5)
    skin_pipeline.fit(X_skin, y_skin)
    print(f"Skin Type CV Accuracy: {skin_scores.mean()*100:.1f}%")
    joblib.dump(skin_pipeline,
                os.path.join(MODEL_DIR, 'skin_type_svm.pkl'))

    # ── 2. Dark Spots — Random Forest ────────────────────────
    print("\n--- Training Dark Spots Random Forest ---")
    spot_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf',     RandomForestClassifier(
                       n_estimators=200,
                       max_depth=8,
                       random_state=42))
    ])
    spot_scores = cross_val_score(spot_pipeline, X_spots, y_spots, cv=5)
    spot_pipeline.fit(X_spots, y_spots)
    print(f"Dark Spots CV Accuracy: {spot_scores.mean()*100:.1f}%")
    joblib.dump(spot_pipeline,
                os.path.join(MODEL_DIR, 'dark_spots_rf.pkl'))

    # ── 3. Eye Bags — Gradient Boosting ──────────────────────
    print("\n--- Training Eye Bags Gradient Boosting ---")
    eye_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb',     GradientBoostingClassifier(
                       n_estimators=150,
                       learning_rate=0.1,
                       max_depth=4,
                       random_state=42))
    ])
    eye_scores = cross_val_score(eye_pipeline, X_eyes, y_eyes, cv=5)
    eye_pipeline.fit(X_eyes, y_eyes)
    print(f"Eye Bags CV Accuracy: {eye_scores.mean()*100:.1f}%")
    joblib.dump(eye_pipeline,
                os.path.join(MODEL_DIR, 'eye_bags_gb.pkl'))

    print("\n✅ All models trained and saved!")
    print(f"Location: {MODEL_DIR}")
    return True


# ══════════════════════════════════════════════════════════════
#  PREDICT FUNCTIONS
# ══════════════════════════════════════════════════════════════
_skin_model  = None
_spot_model  = None
_eye_model   = None


def load_models():
    global _skin_model, _spot_model, _eye_model

    skin_path = os.path.join(MODEL_DIR, 'skin_type_svm.pkl')

    if not os.path.exists(skin_path):
        print("Models not found — training now...")
        train_all_models()

    _skin_model = joblib.load(os.path.join(MODEL_DIR, 'skin_type_svm.pkl'))
    _spot_model = joblib.load(os.path.join(MODEL_DIR, 'dark_spots_rf.pkl'))
    _eye_model  = joblib.load(os.path.join(MODEL_DIR, 'eye_bags_gb.pkl'))
    print("✅ All ML models loaded!")


def predict_skin_type(features_array):
    global _skin_model
    if _skin_model is None:
        load_models()
    pred   = _skin_model.predict(features_array)[0]
    proba  = _skin_model.predict_proba(features_array)[0]
    conf   = round(float(max(proba)) * 100, 1)
    return str(pred), conf


def predict_dark_spots(spot_count, spot_density, dark_mean, dark_std):
    global _spot_model
    if _spot_model is None:
        load_models()
    X    = np.array([[spot_count, spot_density, dark_mean, dark_std]])
    pred = _spot_model.predict(X)[0]
    return str(pred)


def predict_eye_bags(eye_darkness, eye_gradient, eye_bag_score, eyes_detected):
    global _eye_model
    if _eye_model is None:
        load_models()
    X    = np.array([[eye_darkness, eye_gradient, eye_bag_score, eyes_detected]])
    pred = _eye_model.predict(X)[0]
    return str(pred)


if __name__ == '__main__':
    train_all_models()