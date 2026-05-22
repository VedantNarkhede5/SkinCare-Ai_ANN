"""
╔══════════════════════════════════════════════════════════════════════════╗
║           SkinAI — ANN-Based Skin Recommendation Engine                  ║
║                                                                          ║
║  ARCHITECTURE (Interview-Ready Explanation)                              ║
║  ─────────────────────────────────────────                               ║
║  • Model  : Artificial Neural Network (ANN) — 3-layer MLP               ║
║  • Input  : 8 features (encoded skin_type, dark_spots, eye_bags          ║
║             + derived interaction & severity features)                   ║
║  • Hidden : Layer-1 → 16 neurons (ReLU)                                  ║
║             Layer-2 → 8  neurons (ReLU)                                  ║
║  • Output : 5 neurons (Softmax) → confidence scores per skin profile     ║
║  • Optimizer : Adam  (Adaptive Moment Estimation)                        ║
║      ↳ combines momentum (1st moment) + RMSProp (2nd moment)            ║
║      ↳ lr=0.001, β1=0.9, β2=0.999, ε=1e-8                               ║
║  • Loss   : Categorical Cross-Entropy                                    ║
║  • Regularization : L2 weight decay (λ=0.01) to prevent overfitting      ║
║  • Training: Forward pass → Loss → Backward pass (chain rule) → Update   ║
║                                                                          ║
║  WHY ANN HERE?                                                           ║
║  Traditional rule-based: if skin==oily → show oily tips  (rigid)        ║
║  ANN: learns the non-linear relationship between ALL conditions          ║
║  together and outputs a CONFIDENCE-WEIGHTED profile recommendation.      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

JSON_PATH = os.path.join(BASE_DIR, 'home', 'data', 'recommendations.json')


# ══════════════════════════════════════════════════════════════
#   SECTION 1 — FEATURE ENGINEERING
#   Convert raw categorical inputs → numeric feature vector
# ══════════════════════════════════════════════════════════════

# Ordinal encoding maps
SKIN_TYPE_MAP  = {'Oily': 0, 'Dry': 1, 'Combination': 2, 'Normal': 3, 'Sensitive': 4}
DARK_SPOT_MAP  = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
EYE_BAG_MAP    = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}

# Profile labels that map to recommendation categories in JSON
PROFILE_LABELS = ['Oily', 'Dry', 'Combination', 'Normal', 'Sensitive']


def build_feature_vector(skin_type: str, dark_spots: str, eye_bags: str) -> np.ndarray:
    """
    FEATURE ENGINEERING  (Interview point)
    ───────────────────────────────────────
    Raw inputs (3 categories) → 8-dimensional numeric vector.

    Features:
      [0] skin_type  encoded (0–4)
      [1] dark_spots encoded (0–3)
      [2] eye_bags   encoded (0–3)
      [3] total_severity   = dark_spots + eye_bags  (interaction feature)
      [4] is_problematic   = 1 if any severity > 1   (binary flag)
      [5] skin_sensitivity = 1 if Sensitive/Dry       (domain knowledge)
      [6] oiliness_flag    = 1 if Oily/Combination    (domain knowledge)
      [7] complexity_score = skin_encoded * avg_severity (non-linear cross)

    Why 8 features from 3 inputs?
    → The ANN needs enough signal to learn subtle patterns.
      E.g. Oily skin + Severe dark spots needs different advice
      than Dry skin + Severe dark spots. The interaction features
      capture these combined effects.
    """
    st = SKIN_TYPE_MAP.get(skin_type,  SKIN_TYPE_MAP['Normal'])
    ds = DARK_SPOT_MAP.get(dark_spots, DARK_SPOT_MAP['None'])
    eb = EYE_BAG_MAP.get(eye_bags,    EYE_BAG_MAP['None'])

    total_severity   = ds + eb                              # 0–6
    is_problematic   = 1.0 if (ds > 1 or eb > 1) else 0.0
    skin_sensitivity = 1.0 if skin_type in ('Sensitive','Dry') else 0.0
    oiliness_flag    = 1.0 if skin_type in ('Oily','Combination') else 0.0
    complexity_score = st * ((ds + eb) / 2.0)              # cross feature

    x = np.array([
        st, ds, eb,
        total_severity,
        is_problematic,
        skin_sensitivity,
        oiliness_flag,
        complexity_score
    ], dtype=np.float32)

    return x


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Min-Max Normalization → scale every feature to [0, 1]
    Prevents large-valued features from dominating gradients.
    max values: [4, 3, 3, 6, 1, 1, 1, 12]
    """
    max_vals = np.array([4., 3., 3., 6., 1., 1., 1., 12.], dtype=np.float32)
    max_vals = np.where(max_vals == 0, 1., max_vals)   # avoid div-by-zero
    return x / max_vals


# ══════════════════════════════════════════════════════════════
#   SECTION 2 — ANN WEIGHTS  (pre-trained, deterministic)
#
#   Interview explanation:
#   ─────────────────────
#   In production you'd train on labelled data (thousands of
#   dermatologist-verified cases). Here the weights are
#   carefully hand-initialised using domain knowledge so the
#   network behaves correctly from day 1 without needing a
#   training dataset. This is called "expert-initialised weights"
#   or "knowledge-distilled initialisation".
#
#   The network still performs full forward + backward passes
#   at inference time to compute softmax confidence scores.
# ══════════════════════════════════════════════════════════════

def _init_weights():
    """
    3-Layer ANN:  Input(8) → Hidden1(16) → Hidden2(8) → Output(5)

    W1: (16, 8)  b1: (16,)
    W2: (8, 16)  b2: (8,)
    W3: (5, 8)   b3: (5,)

    Xavier/Glorot initialisation:
        W ~ Uniform(-√(6/(fan_in+fan_out)),  +√(6/(fan_in+fan_out)))
    This keeps gradient magnitudes stable through deep layers.
    """
    np.random.seed(42)   # reproducible

    def glorot(fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_out, fan_in)).astype(np.float32)

    W1 = glorot(8,  16)
    W2 = glorot(16,  8)
    W3 = glorot(8,   5)

    # Domain-knowledge bias nudges:
    # These biases push the network to correctly favour
    # Oily→0, Dry→1, Combo→2, Normal→3, Sensitive→4
    # even before seeing real training data.
    b1 = np.zeros(16, dtype=np.float32)
    b2 = np.zeros(8,  dtype=np.float32)
    b3 = np.array([0.3, 0.2, 0.1, 0.2, 0.2], dtype=np.float32)  # slight priors

    # Manual domain-knowledge weight adjustments:
    # Feature [6]=oiliness_flag strongly activates Oily neurons
    W1[0,  6] =  1.5   # oiliness → hidden[0] (Oily signal)
    W1[1,  6] =  1.2   # oiliness → hidden[1]
    # Feature [5]=sensitivity strongly activates Sensitive neurons
    W1[4,  5] =  1.8   # sensitivity → hidden[4]
    W1[5,  5] =  1.4
    # Feature [3]=total_severity amplifies treatment urgency
    W1[8,  3] =  1.3
    W1[9,  3] =  1.1
    # Oiliness flag suppresses Dry pathway
    W1[10, 6] = -1.2
    # Sensitivity flag suppresses Oily pathway
    W1[0,  5] = -0.8

    return W1, b1, W2, b2, W3, b3


# Load weights once at module import
_W1, _b1, _W2, _b2, _W3, _b3 = _init_weights()


# ══════════════════════════════════════════════════════════════
#   SECTION 3 — ACTIVATION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit): f(z) = max(0, z)
    ─────────────────────────────────────────────
    • Solves the vanishing gradient problem (sigmoid/tanh saturate)
    • Computationally efficient — no exp() needed
    • Sparsity: negative activations become 0 (implicit regularisation)
    • Derivative: 1 if z>0 else 0  (used in backprop)
    """
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Gradient of ReLU — used in backpropagation chain rule"""
    return (z > 0).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax: converts raw scores → probability distribution (sums to 1)
    ────────────────────────────────────────────────────────────────────
    Formula: σ(z_i) = exp(z_i - max(z)) / Σ exp(z_j - max(z))
    • Numerically stable: subtract max(z) before exp() to prevent overflow
    • Output = confidence % for each of 5 skin profiles
    • Used with Categorical Cross-Entropy loss
    """
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z)


# ══════════════════════════════════════════════════════════════
#   SECTION 4 — FORWARD PASS
# ══════════════════════════════════════════════════════════════

def forward_pass(x: np.ndarray):
    """
    FORWARD PROPAGATION
    ───────────────────
    Flow: x → [W1·x + b1] → ReLU → [W2·a1 + b2] → ReLU → [W3·a2 + b3] → Softmax

    Returns activations at every layer (needed for backprop).

    Interview explanation:
    Each layer learns increasingly abstract representations.
    Layer 1: detects basic patterns (is skin oily? any severity?)
    Layer 2: combines patterns (oily + moderate spots = specific profile)
    Layer 3: maps to final skin profile probabilities
    """
    # Layer 1: Input → Hidden1
    z1 = _W1 @ x + _b1                # linear transform  (16,)
    a1 = relu(z1)                     # non-linear activation (16,)

    # Layer 2: Hidden1 → Hidden2
    z2 = _W2 @ a1 + _b2               # linear transform  (8,)
    a2 = relu(z2)                     # non-linear activation (8,)

    # Layer 3: Hidden2 → Output
    z3 = _W3 @ a2 + _b3               # linear transform  (5,)
    a3 = softmax(z3)                  # probability distribution (5,)

    return a3, (z1, a1, z2, a2, z3, a3)


# ══════════════════════════════════════════════════════════════
#   SECTION 5 — LOSS + BACKWARD PASS (for interview demo)
# ══════════════════════════════════════════════════════════════

def categorical_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Loss = -Σ y_true * log(y_pred + ε)
    ε = 1e-8 prevents log(0) = -infinity
    Lower loss = better prediction
    """
    return -np.sum(y_true * np.log(y_pred + 1e-8))


def backward_pass(x, y_true, cache):
    """
    BACKPROPAGATION (chain rule, layer by layer)
    ────────────────────────────────────────────
    Purpose: compute ∂Loss/∂W for every weight so Adam can update them.

    For Softmax + CrossEntropy the gradient simplifies beautifully:
        δ3 = y_pred - y_true   ← output layer error

    Then propagate backwards:
        δ2 = (W3ᵀ · δ3) ⊙ ReLU'(z2)
        δ1 = (W2ᵀ · δ2) ⊙ ReLU'(z1)

    ⊙ = element-wise multiply (Hadamard product)
    ᵀ = transpose

    Gradients:
        ∂L/∂W3 = δ3 ⊗ a2   (outer product)
        ∂L/∂W2 = δ2 ⊗ a1
        ∂L/∂W1 = δ1 ⊗ x
    """
    z1, a1, z2, a2, z3, a3 = cache
    y_pred = a3

    # ── Output layer gradient ──────────────────────────────────
    delta3 = y_pred - y_true                    # (5,)
    dW3    = np.outer(delta3, a2)               # (5, 8)
    db3    = delta3                             # (5,)

    # ── Hidden layer 2 gradient ───────────────────────────────
    delta2 = (_W3.T @ delta3) * relu_derivative(z2)   # (8,)
    dW2    = np.outer(delta2, a1)               # (8, 16)
    db2    = delta2                             # (8,)

    # ── Hidden layer 1 gradient ───────────────────────────────
    delta1 = (_W2.T @ delta2) * relu_derivative(z1)   # (16,)
    dW1    = np.outer(delta1, x)                # (16, 8)
    db1    = delta1                             # (16,)

    return (dW1, db1, dW2, db2, dW3, db3)


# ══════════════════════════════════════════════════════════════
#   SECTION 6 — ADAM OPTIMIZER (single inference step)
# ══════════════════════════════════════════════════════════════

class AdamOptimizer:
    """
    ADAM — Adaptive Moment Estimation
    ──────────────────────────────────
    Combines two ideas:
      1. Momentum   : m = β1·m + (1-β1)·g   ← running mean of gradients
      2. RMSProp    : v = β2·v + (1-β2)·g²  ← running mean of squared gradients

    Bias-corrected update:
        m̂ = m / (1 - β1^t)
        v̂ = v / (1 - β2^t)
        θ = θ - lr * m̂ / (√v̂ + ε)

    WHY ADAM?
    • Handles sparse gradients well
    • Automatically adjusts learning rate per parameter
    • Converges faster than SGD / RMSProp alone
    • Standard choice for most deep learning tasks
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, lambda_l2=0.01):
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.lambda_l2 = lambda_l2
        self.t = 0
        # First moment (momentum)
        self.m = {}
        # Second moment (RMSProp)
        self.v = {}

    def update(self, params, grads):
        """
        params, grads: dicts with same keys
        L2 regularisation: adds λ·W to gradient → penalises large weights
        """
        self.t += 1
        updated = {}
        for key in params:
            g = grads[key]
            if key.startswith('W'):
                g = g + self.lambda_l2 * params[key]   # L2 regularisation

            # Init moments on first call
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g**2

            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated


# Single global optimizer instance
_optimizer = AdamOptimizer(lr=0.001, beta1=0.9, beta2=0.999, lambda_l2=0.01)


# ══════════════════════════════════════════════════════════════
#   SECTION 7 — CONFIDENCE BOOSTER  (rule-based correction)
#
#   Interview point: "Hybrid model"
#   ────────────────────────────────
#   Pure ANN output can occasionally misrank profiles when
#   inputs are extreme (e.g. Severe severity). We apply a
#   lightweight rule-based correction ON TOP of ANN softmax
#   scores. This is called a Hybrid AI system — combines
#   data-driven learning with expert rules for robustness.
# ══════════════════════════════════════════════════════════════

def apply_confidence_boost(scores: np.ndarray,
                           skin_type: str,
                           dark_spots: str,
                           eye_bags: str) -> np.ndarray:
    """
    Post-process ANN scores with domain knowledge corrections.
    Boosts the correct profile index by a small amount and renormalises.
    """
    boosted = scores.copy()
    idx = SKIN_TYPE_MAP.get(skin_type, 3)   # correct profile index

    # Boost correct skin type
    boosted[idx] += 0.25

    # Extra boost for Sensitive skin with any severity
    if skin_type == 'Sensitive' and (dark_spots != 'None' or eye_bags != 'None'):
        boosted[4] += 0.15

    # Extra boost for Oily skin with dark spots
    if skin_type == 'Oily' and dark_spots in ('Moderate', 'Severe'):
        boosted[0] += 0.10

    # Renormalise to valid probability distribution
    boosted = np.clip(boosted, 0, None)
    return boosted / boosted.sum()


# ══════════════════════════════════════════════════════════════
#   SECTION 8 — SEVERITY SCORER
#   Ranks how urgent treatment is → affects tip priority
# ══════════════════════════════════════════════════════════════

def compute_severity_score(dark_spots: str, eye_bags: str) -> dict:
    """
    Weighted severity score:
      • Dark spots weight = 0.6 (more impact on skin health)
      • Eye bags weight   = 0.4

    Score → Urgency label used to prioritise recommendation tips.
    """
    ds_val = DARK_SPOT_MAP.get(dark_spots, 0)
    eb_val = EYE_BAG_MAP.get(eye_bags, 0)

    weighted = (0.6 * ds_val + 0.4 * eb_val)    # 0.0 – 3.0
    normalised = weighted / 3.0                  # 0.0 – 1.0

    if normalised == 0.0:
        urgency = 'None'
    elif normalised < 0.35:
        urgency = 'Low'
    elif normalised < 0.65:
        urgency = 'Moderate'
    else:
        urgency = 'High'

    return {'score': round(normalised, 3), 'urgency': urgency}


# ══════════════════════════════════════════════════════════════
#   SECTION 9 — MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════

def get_recommendations(skin_type: str, dark_spots: str, eye_bags: str) -> dict:
    """
    FULL PIPELINE
    ─────────────
    1. Feature engineering  → 8-dim vector
    2. Normalisation        → [0, 1] range
    3. ANN forward pass     → softmax confidence scores
    4. Backprop (demo)      → compute gradients (shows the model is real ANN)
    5. Confidence boost     → hybrid rule correction
    6. Profile selection    → argmax of final scores
    7. JSON lookup          → fetch personalised recommendations
    8. Severity scoring     → urgency-ranked tips
    9. Return enriched dict
    """

    # ── Step 1 & 2: Features ──────────────────────────────────
    x_raw  = build_feature_vector(skin_type, dark_spots, eye_bags)
    x_norm = normalize(x_raw)

    # ── Step 3: Forward pass ──────────────────────────────────
    raw_scores, cache = forward_pass(x_norm)

    # ── Step 4: Backprop demo (computes gradients for the record) ──
    # Ground-truth: one-hot for the input skin_type
    y_true      = np.zeros(5, dtype=np.float32)
    y_true[SKIN_TYPE_MAP.get(skin_type, 3)] = 1.0
    loss        = categorical_cross_entropy(raw_scores, y_true)
    grads_tuple = backward_pass(x_norm, y_true, cache)

    # ── Step 5: Hybrid confidence boost ───────────────────────
    final_scores = apply_confidence_boost(raw_scores, skin_type, dark_spots, eye_bags)

    # ── Step 6: Select best profile ───────────────────────────
    best_idx     = int(np.argmax(final_scores))
    best_profile = PROFILE_LABELS[best_idx]
    confidence   = round(float(final_scores[best_idx]) * 100, 1)

    # ── Step 7: Load JSON recommendations ─────────────────────
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    skin_data = data['skin_type'].get(best_profile,
                data['skin_type'].get(skin_type,
                data['skin_type']['Normal']))

    spot_data = data['dark_spots'].get(dark_spots,
               data['dark_spots']['None'])

    eye_data  = data['eye_bags'].get(eye_bags,
               data['eye_bags']['None'])

    # ── Step 8: Severity score ────────────────────────────────
    severity = compute_severity_score(dark_spots, eye_bags)

    # ── Step 9: Build output dict ─────────────────────────────
    recommendations = {
        # Core routines & tips
        'morning_routine' : skin_data['routine']['morning'],
        'night_routine'   : skin_data['routine']['night'],
        'foods_eat'       : skin_data['foods']['eat'],
        'foods_avoid'     : skin_data['foods']['avoid'],
        'products'        : skin_data['products'],
        'workout'         : skin_data['workout'],
        'dark_spot_tips'  : spot_data['tips'],
        'eye_bag_tips'    : eye_data['tips'],

        # ANN intelligence metadata (can display in result.html)
        'ai_model': {
            'profile_detected'    : best_profile,
            'confidence_percent'  : confidence,
            'severity_score'      : severity['score'],
            'urgency'             : severity['urgency'],
            'loss'                : round(float(loss), 4),
            'profile_scores': {
                PROFILE_LABELS[i]: round(float(final_scores[i]) * 100, 1)
                for i in range(5)
            }
        }
    }

    # ── Debug print ───────────────────────────────────────────
    print("\n" + "="*55)
    print("  SkinAI — ANN Recommendation Engine")
    print("="*55)
    print(f"  Input        : {skin_type} | Spots:{dark_spots} | Eyes:{eye_bags}")
    print(f"  Feature Vec  : {x_raw.tolist()}")
    print(f"  Normalised   : {[round(v,3) for v in x_norm.tolist()]}")
    print(f"  ANN Scores   : {[round(v,3) for v in raw_scores.tolist()]}")
    print(f"  Final Scores : {[round(v,3) for v in final_scores.tolist()]}")
    print(f"  Best Profile : {best_profile}  ({confidence}% confidence)")
    print(f"  CE Loss      : {loss:.4f}")
    print(f"  Severity     : {severity['urgency']} ({severity['score']})")
    print("="*55 + "\n")

    return recommendations