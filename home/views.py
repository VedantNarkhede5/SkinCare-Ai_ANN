from django.shortcuts import render, redirect
from django.contrib import messages
from home.ml.skin_analyzer import analyze_skin
from home.ml.recommender import get_recommendations
import os
import json
import hashlib
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_FILE = os.path.join(BASE_DIR, 'home', 'data', 'users.json')


# ════════════════════════════════════════════════════════════
#   JSON HELPERS
# ════════════════════════════════════════════════════════════

def load_users():
    if not os.path.exists(USERS_FILE):
        save_users({"users": []})
        return {"users": []}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)


def save_users(data):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def hash_password(password):
    salt = "skinai_salt_2025"
    return hashlib.sha256((salt + password).encode()).hexdigest()


def find_user_by_username(username):
    data = load_users()
    for user in data["users"]:
        if user["username"].lower() == username.lower():
            return user
    return None


def find_user_by_email(email):
    data = load_users()
    for user in data["users"]:
        if user["email"].lower() == email.lower():
            return user
    return None


# ════════════════════════════════════════════════════════════
#   SESSION HELPERS  — FIXED & HARDENED
# ════════════════════════════════════════════════════════════

def session_login(request, user):
    # Flush any old session first — prevents session fixation attacks
    request.session.flush()
    request.session['user_id']      = user['id']
    request.session['username']     = user['username']
    request.session['first_name']   = user['first_name']
    request.session['is_logged_in'] = True
    request.session.modified        = True   # force Django to save it


def session_logout(request):
    request.session.flush()


def is_logged_in(request):
    """
    3-layer session validation:
    Layer 1 — flag check
    Layer 2 — user_id present
    Layer 3 — user_id actually exists in users.json
    Stale/corrupt sessions are flushed immediately.
    """
    if not request.session.get('is_logged_in', False):
        return False

    user_id = request.session.get('user_id')
    if not user_id:
        request.session.flush()
        return False

    # Cross-check against users.json
    data = load_users()
    user_exists = any(u['id'] == user_id for u in data['users'])
    if not user_exists:
        request.session.flush()
        return False

    return True


def login_required_json(view_func):
    """
    Decorator — invalid session → flush + redirect to welcome.
    """
    def wrapper(request, *args, **kwargs):
        if not is_logged_in(request):
            request.session.flush()
            return redirect('welcome')
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


# ════════════════════════════════════════════════════════════
#   VIEWS
# ════════════════════════════════════════════════════════════

def welcome(request):
    if is_logged_in(request):
        return redirect('home')
    return render(request, 'welcome.html')


def login_view(request):
    if is_logged_in(request):
        return redirect('home')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if not username or not password:
            messages.error(request, 'Please fill in all fields.')
            return render(request, 'login.html')

        user = find_user_by_username(username)

        if user and user['password'] == hash_password(password):
            session_login(request, user)
            messages.success(request, f"Welcome back, {user['first_name'] or user['username']}! 🌿")
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password. Please try again.')

    return render(request, 'login.html')


def signup_view(request):
    if is_logged_in(request):
        return redirect('home')

    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name  = request.POST.get('last_name',  '').strip()
        username   = request.POST.get('username',   '').strip()
        email      = request.POST.get('email',      '').strip()
        password1  = request.POST.get('password1',  '')
        password2  = request.POST.get('password2',  '')

        if not username or not email or not password1:
            messages.error(request, 'Please fill in all required fields.')
        elif len(username) < 3:
            messages.error(request, 'Username must be at least 3 characters.')
        elif find_user_by_username(username):
            messages.error(request, 'That username is already taken. Please choose another.')
        elif find_user_by_email(email):
            messages.error(request, 'An account with this email already exists.')
        elif password1 != password2:
            messages.error(request, 'Passwords do not match.')
        elif len(password1) < 8:
            messages.error(request, 'Password must be at least 8 characters.')
        else:
            new_user = {
                "id"         : str(uuid.uuid4()),
                "username"   : username,
                "email"      : email,
                "password"   : hash_password(password1),
                "first_name" : first_name,
                "last_name"  : last_name,
                "created_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            data = load_users()
            data["users"].append(new_user)
            save_users(data)

            session_login(request, new_user)
            messages.success(request, f"Account created! Welcome, {first_name or username}. 🌿")
            return redirect('home')

    return render(request, 'signup.html')

def privacy(request):
    return render(request, 'privacy.html')

def logout_view(request):
    session_logout(request)
    return redirect('welcome')


# ════════════════════════════════════════════════════════════
#   PROTECTED VIEWS
# ════════════════════════════════════════════════════════════

@login_required_json
def home(request):
    return render(request, 'index.html')

def upload(request):
    if request.method == 'POST':
        photo = request.FILES.get('photo')
        consent = request.POST.get('consent')

        # Check consent
        if not consent:
            return render(request, 'index.html', {
                'error': 'Please accept the privacy policy to continue.'
            })

        if photo:
            save_folder = os.path.join(BASE_DIR, 'media', 'uploads')
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, photo.name)

            with open(save_path, 'wb+') as f:
                for chunk in photo.chunks():
                    f.write(chunk)

            # Run skin analysis
            result = analyze_skin(save_path)

            # Get recommendations
            recommendations = {}
            if result['face_found']:
                recommendations = get_recommendations(
                    skin_type  = result['skin_type'],
                    dark_spots = result['dark_spots'],
                    eye_bags   = result['eye_bags']
                )

            # ✅ Delete original upload after analysis
            if os.path.exists(save_path):
                os.remove(save_path)
                print("Original photo deleted after analysis ✅")

            return render(request, 'result.html', {
                'photo_url'       : result.get('processed_image_url', ''),
                'photo_name'      : photo.name,
                'result'          : result,
                'recommendations' : recommendations,
            })

        else:
            return render(request, 'index.html', {
                'error': 'Please select a photo first.'
            })

    return render(request, 'index.html')