import cv2
import numpy as np
import tempfile
import pytesseract
import mahotas.features
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
import streamlit as st
import hashlib
import json
import os
import pandas as pd
import pyotp
import qrcode
import io
import base64
import cirq
import matplotlib.pyplot as plt
import time
from scipy import signal
import hashlib
import json
import pyotp
import qrcode
from io import BytesIO
from cryptography.fernet import Fernet
import pandas as pd
from collections import defaultdict
from datetime import datetime

USER_FILE = "users.json"
RESET_REQUEST_FILE = "reset_requests.json"
ADMIN_PASSWORD = "Dk}5^;r9XzT.q{h"
ADMIN_EMAIL = "athish.ns@expressanalytics.net"

chat_rooms = defaultdict(list)

def hash_sha3_512(text):
    return hashlib.sha3_512(text.encode()).hexdigest()

def get_user_key(username):
    key_file = f"{username}_key.key"
    if not os.path.exists(key_file):
        key = Fernet.generate_key()
        with open(key_file, "wb") as f:
            f.write(key)
    else:
        with open(key_file, "rb") as f:
            key = f.read()
    return Fernet(key)

def get_vault_file(username):
    return f"vault_{username}.json"

def load_vault(username):
    fernet = get_user_key(username)
    file = get_vault_file(username)
    if os.path.exists(file):
        with open(file, "rb") as f:
            return json.loads(fernet.decrypt(f.read()).decode())
    return {}

def save_vault(username, data):
    fernet = get_user_key(username)
    file = get_vault_file(username)
    encrypted = fernet.encrypt(json.dumps(data).encode())
    with open(file, "wb") as f:
        f.write(encrypted)

def load_reset_requests():
    if os.path.exists(RESET_REQUEST_FILE):
        with open(RESET_REQUEST_FILE, "r") as f:
            return json.load(f)
    return {}

def save_reset_requests(requests):
    with open(RESET_REQUEST_FILE, "w") as f:
        json.dump(requests, f)

def generate_qr_code_uri(username, secret):
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(name=username, issuer_name="QuantumPasswordManager")
    qr = qrcode.make(uri)
    buffer = BytesIO()
    qr.save(buffer)
    return buffer

def chat_interface():
    current_user = st.session_state['authenticated_user']
    st.markdown("## 💬 Chat Center")
    users = load_users()
    other_users = [u['username'] for u in users if u['username'] != current_user]
    if "chats" not in st.session_state:
        st.session_state["chats"] = {}
    if "active_chat" not in st.session_state:
        st.session_state["active_chat"] = None
    st.sidebar.markdown("### 📇 Start New Chat")
    chat_type = st.sidebar.radio("Type", ["Individual", "Group"], key="chat_type")
    if chat_type == "Individual":
        recipient = st.sidebar.selectbox("Choose User", other_users, key="chat_user")
        chat_id = f"chat_{'_'.join(sorted([current_user, recipient]))}"
        if st.sidebar.button("Start Chat"):
            if chat_id not in st.session_state["chats"]:
                st.session_state["chats"][chat_id] = {
                    "participants": [current_user, recipient],
                    "messages": []
                }
            st.session_state["active_chat"] = chat_id
    else:
        group_name = st.sidebar.text_input("Group Name", key="group_name")
        members = st.sidebar.multiselect("Group Members", other_users, key="group_members")
        if st.sidebar.button("Create Group"):
            if group_name and members:
                group_id = f"group_{group_name.lower().replace(' ', '_')}"
                if group_id not in st.session_state["chats"]:
                    st.session_state["chats"][group_id] = {
                        "participants": members + [current_user],
                        "messages": []
                    }
                st.session_state["active_chat"] = group_id
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Active Chats")
    for chat_id, chat_data in st.session_state["chats"].items():
        if current_user in chat_data["participants"]:
            display_name = chat_id.replace("chat_", "").replace("group_", "Group: ").replace("_", " ")
            if st.sidebar.button(display_name, key=f"switch_{chat_id}"):
                st.session_state["active_chat"] = chat_id
    if st.session_state["active_chat"]:
        chat_id = st.session_state["active_chat"]
        chat = st.session_state["chats"][chat_id]
        is_group = chat_id.startswith("group_")
        title = "Group Chat" if is_group else "Private Chat"
        st.markdown(f"### 📢 {title} - `{chat_id}`")
        st.divider()
        st.markdown("#### 📨 Messages")
        for sender, message, timestamp in chat["messages"]:
            align = "🟢" if sender == current_user else "🔵"
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M")
            st.markdown(f"""
                <div style="margin-bottom: 10px;">
                    <strong>{align} {sender}</strong><span style="float: right; color: gray;">{time_str}</span><br>
                    <div style="margin-left: 10px; background-color: #0e1117; padding: 8px 12px; border-radius: 10px; display: inline-block;">{message}</div>
                </div>
            """, unsafe_allow_html=True)
        st.divider()
        msg_input = st.text_input("Type your message", key="message_input")
        if st.button("Send Message"):
            if msg_input.strip():
                timestamp = datetime.now().timestamp()
                chat["messages"].append((current_user, msg_input.strip(), timestamp))
                st.rerun()
    else:
        st.info("Start or open a chat to begin messaging.")

def password_vault(username):
    st.title(f"🔐 Welcome, {username}")
    menu = st.sidebar.selectbox("Menu", ["Add", "View", "Delete", "Request Password Reset", "Chat", "Logout"])
    vault = load_vault(username)
    if menu == "Add":
        st.subheader("Add New Password")
        site = st.text_input("Site", key="add_site")
        user = st.text_input("Site Username", key="add_username")
        pw = st.text_input("Password", type="password", key="add_password")
        if st.button("Save"):
            if site and user and pw:
                fernet = get_user_key(username)
                encrypted_pw = fernet.encrypt(pw.encode()).decode()
                vault[site] = {"username": user, "password": encrypted_pw}
                save_vault(username, vault)
                st.success(f"Saved password for {site}")
            else:
                st.warning("All fields required.")
    elif menu == "View":
        st.subheader("🔍 View Stored Passwords")
        code = st.text_input("Re-enter 2FA Code", max_chars=6, key="view_2fa")
        user = get_user_by_username(username)
        secret = user["totp_secret"]
        totp = pyotp.TOTP(secret)
        if st.button("Verify & View"):
            if totp.verify(code):
                st.success("2FA verified. Displaying passwords.")
                if vault:
                    fernet = get_user_key(username)
                    data = [[site, data['username'], fernet.decrypt(data['password'].encode()).decode()] for site, data in vault.items()]
                    df = pd.DataFrame(data, columns=["Website", "Username/Email", "Password"])
                    st.table(df)
                else:
                    st.info("No passwords stored.")
            else:
                st.error("Invalid 2FA code.")
    elif menu == "Delete":
        st.subheader("Delete Entry")
        if vault:
            site = st.selectbox("Choose site", list(vault.keys()), key="delete_site")
            if st.button("Delete"):
                vault.pop(site, None)
                save_vault(username, vault)
                st.success(f"{site} deleted.")
        else:
            st.info("No passwords to delete.")
    elif menu == "Request Password Reset":
        st.subheader("Request Password Reset")
        reason = st.text_area("Why do you need a reset?", key="reset_reason")
        if st.button("Submit Request"):
            requests = load_reset_requests()
            requests[username] = reason
            save_reset_requests(requests)
            st.success("Request submitted. Admin will review it.")
    elif menu == "Chat":
        chat_interface()
    elif menu == "Logout":
        if 'authenticated_user' in st.session_state:
            del st.session_state['authenticated_user']
        st.session_state['page'] = None
        st.rerun()

def admin_mode():
    st.title("🔧 Admin Control Panel")
    email_admin = st.text_input("Enter Admin Email", type="default", key="admin_email")
    admin_pass = st.text_input("Enter Admin Password", type="password", key="admin_password")
    if admin_pass != ADMIN_PASSWORD or email_admin != ADMIN_EMAIL:
        st.warning("Incorrect admin credentials.")
        return
    st.success("Admin access granted.")
    users = load_users()
    requests = load_reset_requests()
    st.subheader("👥 All Users")
    user_to_delete = st.selectbox("Select User to Delete", [u['username'] for u in users], key="delete_user")
    if st.button(f"Delete User: {user_to_delete}"):
        users = [u for u in users if u['username'] != user_to_delete]
        save_users(users)
        try:
            os.remove(f"{user_to_delete}_key.key")
            os.remove(get_vault_file(user_to_delete))
        except:
            pass
        st.success(f"User {user_to_delete} deleted.")
    st.subheader("🔑 Delete Password for User")
    target_user = st.selectbox("Select User", [u['username'] for u in users], key="select_target_user")
    if target_user:
        vault = load_vault(target_user)
        if vault:
            site_to_delete = st.selectbox("Select Site", list(vault.keys()), key="delete_password_site")
            if st.button("Delete Password"):
                vault.pop(site_to_delete, None)
                save_vault(target_user, vault)
                st.success("Password deleted.")
        else:
            st.info("No passwords stored.")
    st.subheader("📬 Password Reset Requests")
    if not requests:
        st.info("No reset requests.")
    else:
        for user, reason in requests.items():
            st.markdown(f"**{user}**: _{reason}_")
            new_pw = st.text_input(f"New password for {user}", key=f"reset_{user}")
            if st.button(f"Reset password for {user}", key=f"reset_button_{user}"):
                users = [u for u in users if u['username'] != user]
                users.append({
                    "username": user,
                    "password_hash": hash_sha3_512(new_pw),
                    "totp_secret": pyotp.random_base32()
                })
                save_users(users)
                requests.pop(user)
                save_reset_requests(requests)
                st.success(f"Password for {user} updated.")
    st.subheader("🔑 Create Password for Multiple Users")
    selected_users = st.multiselect("Select Users", [u['username'] for u in users], key="multi_user_select")
    site = st.text_input("Site", key="admin_multi_create_site")
    username = st.text_input("Site Username", key="admin_multi_create_username")
    password = st.text_input("Password", type="password", key="admin_multi_create_password")
    if st.button("Add Password"):
        if selected_users and site and username and password:
            for user in selected_users:
                fernet = get_user_key(user)
                encrypted_pw = fernet.encrypt(password.encode()).decode()
                vault = load_vault(user)
                vault[site] = {"username": username, "password": encrypted_pw}
                save_vault(user, vault)
            st.success(f"Password for '{site}' added to selected users.")
        else:
            st.warning("Fill all fields.")
    if st.button("Clear All Reset Requests"):
        save_reset_requests({})
        st.info("All reset requests cleared.")
# --- End Password Manager Logic ---

def resize_image(image, target_size=(1000, 1000)):
    return cv2.resize(image, target_size)

def compute_features(image):
    resized_image = resize_image(image)

    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(resized_image)

    gray_for_lbp = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    lbp_features = mahotas.features.lbp(gray_for_lbp, radius=8, points=8)
    # OpenCV Sobel for gradient magnitude (edge detection)
    gradient_magnitude = cv2.Sobel(resized_image, cv2.CV_64F, 1, 1, ksize=3)
    # OpenCV integral image
    integral_img = cv2.integral(resized_image)
    integral_img = integral_img[1:, 1:]  # Remove extra row and column

    hog_features = normalize(hog_features.reshape(1, -1))
    lbp_features = normalize(lbp_features.reshape(1, -1))
    gradient_magnitude = normalize(gradient_magnitude.reshape(1, -1))
    integral_img = normalize(integral_img.reshape(1, -1))

    feature_vector = np.concatenate((hog_features, lbp_features, gradient_magnitude, integral_img), axis=1)

    return feature_vector.ravel()

def detect_tampering(image_path, comparison_img_path):
    img = cv2.imread(image_path)
    comparison_img = cv2.imread(comparison_img_path)
    
    if img is None or comparison_img is None:
        print("Error loading images.")
        return
    img = resize_image(img)
    comparison_img = resize_image(comparison_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comparison_gray = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2GRAY)
    
    feature_vector = compute_features(gray)
    
    feature_vector_comparison = compute_features(comparison_gray)
    
    if np.any(np.isnan(feature_vector)) or np.any(np.isnan(feature_vector_comparison)):
        print("Error computing features. Check the input images.")
        return
    
    similarity_score = 1 - cosine(feature_vector, feature_vector_comparison)
    
    print(f"Similarity score: {similarity_score}")
    
    threshold = 0.70
    if similarity_score < threshold:
        print("Tampering detected.")
        analyze_tampering(img, comparison_img)
        detect_bad_name(img, comparison_img)
    else:
        print("Images seem authentic.")

def analyze_tampering(original_img, comparison_img):
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    comparison_gray = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2GRAY)
    
    abs_diff = cv2.absdiff(original_gray, comparison_gray)
    
    num_different_pixels = np.count_nonzero(abs_diff)
    
    total_pixels = original_gray.size
    
    percentage_difference = (num_different_pixels / total_pixels) * 100
    print(f"Percentage difference in pixel values: {percentage_difference:.2f}%")
    
    # Removed nudenet nudity detection
    return percentage_difference

def detect_bad_name(original_img, comparison_img):
    
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    (H, W) = original_img.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    blob = cv2.dnn.blobFromImage(original_img, 1.0, (newW, newH),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    rects, confidences = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    detected_texts = []

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        roi = original_img[startY:endY, startX:endX]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            detected_texts.append(gray_roi[y:y+h, x:x+w])
    bad_keywords = ["bad", "shame", "disgrace", "disgraceful", "shameful", "unethical", "dishonest"]
    found_bad_name = False
    for text in detected_texts:
        for keyword in bad_keywords:
            if keyword in text:
                found_bad_name = True
                break

    if found_bad_name:
        print("The tampered image may be used to bring a bad name to a person.")
    else:
        print("The tampered image does not appear to be used for bringing a bad name to a person.")
def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rects, confidences
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    if probs is not None:
        return boxes[pick].astype("int"), probs[pick]

    return boxes[pick].astype("int")

def hash_features(features):
    m = hashlib.sha3_512()
    m.update(features.tobytes())
    return m.hexdigest()

def eeg_similarity(features1, features2):
    # Use cosine similarity for EEG features
    if np.linalg.norm(features1) == 0 or np.linalg.norm(features2) == 0:
        return 0.0
    sim = 1 - cosine(features1, features2)
    return sim

EEG_SIMILARITY_THRESHOLD = 0.7

USERS_FILE = 'users.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, 'r') as f:
        data = json.load(f)
        return data.get('users', [])

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump({'users': users}, f, indent=2)

def get_user_by_username(username):
    users = load_users()
    return next((u for u in users if u['username'] == username), None)

def extract_eeg_features(csv_file):
    df = pd.read_csv(csv_file)
    # Use all numeric columns for features
    numeric_df = df.select_dtypes(include=[np.number])
    # Simple features: mean, std, min, max for each channel
    features = []
    for col in numeric_df.columns:
        features.extend([
            numeric_df[col].mean(),
            numeric_df[col].std(),
            numeric_df[col].min(),
            numeric_df[col].max()
        ])
    return np.array(features, dtype=np.float32)

def quantum_enhanced_hash(data):
    """Use quantum circuits to enhance hashing"""
    # Convert data to binary string
    if isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    else:
        data_bytes = str(data).encode()
    
    # Create quantum circuit
    qubits = cirq.LineQubit.range(8)  # 8 qubits for simplicity
    circuit = cirq.Circuit()
    
    # Apply Hadamard gates to create superposition
    for qubit in qubits:
        circuit.append(cirq.H(qubit))
    
    # Apply controlled operations based on data
    for i, byte in enumerate(data_bytes[:8]):  # Use first 8 bytes
        for j in range(8):
            if byte & (1 << j):
                circuit.append(cirq.X(qubits[j]))
    
    # Apply more quantum operations
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[2], qubits[3]))
    circuit.append(cirq.CNOT(qubits[4], qubits[5]))
    circuit.append(cirq.CNOT(qubits[6], qubits[7]))
    
    # Measure all qubits
    circuit.append(cirq.measure(*qubits, key='result'))
    
    # Simulate the circuit
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1)
    measurement = result.measurements['result'][0]
    
    # Convert measurement to hash
    quantum_hash = ''.join(map(str, measurement))
    return quantum_hash

def generate_totp_secret():
    """Generate a new TOTP secret for Google Authenticator"""
    return pyotp.random_base32()

def create_qr_code(secret, username):
    """Create QR code for Google Authenticator setup"""
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=username,
        issuer_name="EEG Auth System"
    )
    
    # Create QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 for display
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def verify_totp(secret, code):
    """Verify TOTP code"""
    totp = pyotp.TOTP(secret)
    return totp.verify(code)

def generate_simulated_eeg(duration=10, sampling_rate=256, channels=4):
    """Generate realistic simulated EEG data"""
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate different frequency components for each channel
    eeg_data = {}
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4']
    
    for i, channel in enumerate(channel_names):
        # Alpha waves (8-13 Hz)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        # Beta waves (13-30 Hz)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
        # Theta waves (4-8 Hz)
        theta = 0.4 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
        # Delta waves (0.5-4 Hz)
        delta = 0.2 * np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi)
        
        # Add some noise
        noise = 0.1 * np.random.randn(len(t))
        
        # Combine all components
        signal_component = alpha + beta + theta + delta + noise
        
        # Add some artifacts and variations
        if np.random.rand() > 0.7:
            # Add eye blink artifact
            blink_times = np.random.choice(len(t), size=3, replace=False)
            for blink_time in blink_times:
                start = max(0, blink_time - 10)
                end = min(len(t), blink_time + 10)
                blink_range = np.arange(end - start)
                signal_component[start:end] += 2 * np.exp(-0.1 * blink_range)
        
        eeg_data[channel] = signal_component
    
    return eeg_data, t

def plot_eeg_waveform(eeg_data, t, title="Simulated EEG Signal"):
    """Create a realistic EEG waveform plot"""
    fig, axes = plt.subplots(len(eeg_data), 1, figsize=(12, 8), sharex=True)
    if len(eeg_data) == 1:
        axes = [axes]
    
    for i, (channel, signal_data) in enumerate(eeg_data.items()):
        axes[i].plot(t, signal_data, linewidth=0.8, color='blue', alpha=0.8)
        axes[i].set_ylabel(f'{channel} (μV)')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-50, 50)
        
        # Add some realistic EEG annotations
        if i == 0:
            axes[i].set_title(title, fontsize=14, fontweight='bold')
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    return fig

def simulate_eeg_reading():
    """Simulate real-time EEG reading"""
    st.write("### Simulating EEG Signal Reading...")
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Generate EEG data
    eeg_data, t = generate_simulated_eeg(duration=5, sampling_rate=256, channels=4)
    
    # Simulate real-time reading
    for i in range(101):
        time.sleep(0.05)  # Simulate processing time
        progress_bar.progress(i)
        status_text.text(f"Reading brain activity... {i}%")
    
    status_text.text("✅ EEG signal captured successfully!")
    
    # Display the EEG waveform
    fig = plot_eeg_waveform(eeg_data, t, "Real-time EEG Signal - Memory Trigger Response")
    st.pyplot(fig)
    
    # Extract features from the simulated data
    features = []
    for channel, signal_data in eeg_data.items():
        features.extend([
            np.mean(signal_data),
            np.std(signal_data),
            np.min(signal_data),
            np.max(signal_data),
            np.percentile(signal_data, 25),
            np.percentile(signal_data, 75)
        ])
    
    return np.array(features, dtype=np.float32)

def signup(username, password, trigger_image):
    users = load_users()
    if any(u['username'] == username for u in users):
        return False, 'Username already exists.', None
    
    # Generate TOTP secret
    totp_secret = generate_totp_secret()
    
    # Save trigger image for later display
    image_path = f"trigger_{username}.png"
    with open(image_path, 'wb') as f:
        f.write(trigger_image.read())
    
    # Simulate EEG reading
    eeg_features = simulate_eeg_reading()
    quantum_hash = quantum_enhanced_hash(eeg_features)
    password_hash = hashlib.sha3_512(password.encode()).hexdigest()
    
    users.append({
        'username': username, 
        'password_hash': password_hash, 
        'quantum_hash': quantum_hash, 
        'eeg_features': eeg_features.tolist(), 
        'trigger_image': image_path,
        'totp_secret': totp_secret
    })
    save_users(users)
    
    # Generate QR code for Google Authenticator
    qr_code = create_qr_code(totp_secret, username)
    
    return True, 'Signup successful!', qr_code

def login(username, password):
    users = load_users()
    user = next((u for u in users if u['username'] == username), None)
    if not user:
        return False, 'User not found.'
    
    # Verify password
    password_hash = hashlib.sha3_512(password.encode()).hexdigest()
    if user['password_hash'] != password_hash:
        return False, 'Incorrect password.'
    
    # Simulate EEG reading for login
    st.write("### Authenticating with EEG Signal...")
    eeg_features = simulate_eeg_reading()
    quantum_hash = quantum_enhanced_hash(eeg_features)
    
    if quantum_hash != user['quantum_hash']:
        # Fallback to similarity check
        stored_features = np.array(user['eeg_features'], dtype=np.float32)
        sim = eeg_similarity(eeg_features, stored_features)
        if sim < EEG_SIMILARITY_THRESHOLD:
            return False, f'EEG signal does not match (similarity: {sim:.2f}).'
        return True, f'Login successful! (similarity: {sim:.2f})'
    
    return True, 'Login successful! (quantum verified)'

def main():
    st.set_page_config(page_title="Quantum-Enhanced EEG-Based Authentication System", layout="centered")
    st.title("Quantum-Enhanced EEG-Based Authentication System with MFA")
    st.markdown("### *Simulating Real-time Brain-Computer Interface*")
    
    if st.session_state.get('page') == 'password_manager':
        if 'authenticated_user' in st.session_state:
            password_vault(st.session_state['authenticated_user'])
        else:
            st.error("No authenticated user found. Please login through the EEG system.")
            if st.button("Go Back to EEG Login"):
                st.session_state['page'] = None
                st.rerun()
        return
    if st.session_state.get('page') == 'admin_panel':
        admin_mode()
        return
    
    # Handle password manager redirect
    if st.session_state.get('redirect_to_manager', False):
        st.session_state['page'] = 'password_manager'
        st.session_state['redirect_to_manager'] = False
        st.rerun()
    
    # Check if user is authenticated
    if 'authenticated_user' in st.session_state:
        st.success(f"Welcome {st.session_state['authenticated_user']}!")
        
        # Show three buttons after authentication
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔑 Login (Password Manager)", use_container_width=True):
                st.session_state['page'] = 'password_manager'
                st.rerun()
        
        with col2:
            if st.button("📝 Signup (New User)", use_container_width=True):
                st.session_state['show_signup_form'] = True
                st.rerun()
        
        with col3:
            if st.button("⚙️ Admin Panel", use_container_width=True):
                st.session_state['page'] = 'admin_panel'
                st.rerun()
        
        # Handle signup form after authentication
        if st.session_state.get('show_signup_form', False):
            st.header("📝 New User Registration")
            new_username = st.text_input("New Username", key="new_signup_user")
            new_password = st.text_input("New Password", type="password", key="new_signup_pass")
            new_trigger_image = st.file_uploader("Upload Memory Trigger Image", type=["png", "jpg", "jpeg"], key="new_signup_img")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Create New User"):
                    if not new_username or not new_password or not new_trigger_image:
                        st.error("All fields are required.")
                    else:
                        st.write("### 📸 Memory Trigger Setup")
                        st.image(new_trigger_image, caption="Memory Trigger Image", use_container_width=True)
                        st.write("**Focus on this image and think about the memory it evokes...**")
                        
                        ok, msg, qr_code = signup(new_username, new_password, new_trigger_image)
                        if ok:
                            st.success(msg)
                            st.write("### 📱 Google Authenticator Setup")
                            st.write("Scan this QR code with Google Authenticator:")
                            st.image(f"data:image/png;base64,{qr_code}", caption="QR Code for Google Authenticator")
                            st.write("**Important:** Save this QR code or add the account to Google Authenticator before proceeding.")
                        else:
                            st.error(msg)
            
            with col2:
                if st.button("❌ Cancel"):
                    del st.session_state['show_signup_form']
                    st.rerun()
        
        # Handle admin form after authentication
        if st.session_state.get('show_admin_form', False):
            st.header("⚙️ Admin Panel Access")
            admin_username = st.text_input("Admin Username", key="admin_user")
            admin_password = st.text_input("Admin Password", type="password", key="admin_pass")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Access Admin Panel"):
                    if not admin_username or not admin_password:
                        st.error("Admin credentials are required.")
                    else:
                        # Simple admin check (you can enhance this)
                        if admin_username == "admin" and admin_password == "admin123":
                            st.success("✅ Admin access granted!")
                            st.write("### 🔧 Admin Panel Features:")
                            st.write("- User Management")
                            st.write("- System Configuration")
                            st.write("- Security Settings")
                            st.write("- Audit Logs")
                            # Add more admin features here
                        else:
                            st.error("❌ Invalid admin credentials.")
            
            with col2:
                if st.button("❌ Cancel"):
                    del st.session_state['show_admin_form']
                    st.rerun()
        return  # Prevent showing the sidebar menu when authenticated
    
    menu = st.sidebar.selectbox("Menu", ["Signup", "Login"])
    
    if menu == "Signup":
        st.header("🔐 User Registration")
        username = st.text_input("Username", key="signup_user")
        password = st.text_input("Password", type="password", key="signup_pass")
        trigger_image = st.file_uploader("Upload Memory Trigger Image", type=["png", "jpg", "jpeg"], key="signup_img")
        
        if st.button("🚀 Start Registration"):
            if not username or not password or not trigger_image:
                st.error("All fields are required.")
            else:
                st.write("### 📸 Memory Trigger Setup")
                st.image(trigger_image, caption="Your Memory Trigger Image", use_container_width=True)
                st.write("**Focus on this image and think about the memory it evokes...**")
                
                ok, msg, qr_code = signup(username, password, trigger_image)
                if ok:
                    st.success(msg)
                    st.write("### 📱 Google Authenticator Setup")
                    st.write("Scan this QR code with Google Authenticator:")
                    st.image(f"data:image/png;base64,{qr_code}", caption="QR Code for Google Authenticator")
                    st.write("**Important:** Save this QR code or add the account to Google Authenticator before proceeding.")
                else:
                    st.error(msg)
    
    elif menu == "Login":
        st.header("🔓 User Authentication")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        
        # Show trigger image if username exists
        users = load_users()
        user = next((u for u in users if u['username'] == username), None)
        if user and os.path.exists(user['trigger_image']):
            st.write("### 📸 Memory Trigger")
            st.image(user['trigger_image'], caption="Focus on this image and recall the memory...", use_container_width=True)
        
        if st.button("Authenticate"):
            if not username or not password:
                st.error("Username and password are required.")
            else:
                # First do EEG authentication
                st.write("### Authenticating with EEG Signal...")
                ok, msg = login(username, password)
                if ok:
                    st.success(msg)
                    st.write("🔐 **Access Granted!** Welcome to the secure system.")
                    
                    # Set authenticated user and show the three buttons
                    st.session_state['authenticated_user'] = username
                    st.rerun()
                else:
                    st.error(msg)

if __name__ == "__main__":
    
    main()
