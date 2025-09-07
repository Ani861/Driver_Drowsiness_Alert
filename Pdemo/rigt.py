import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
import threading
from queue import Queue, Empty
from collections import deque
import blynklib
 # Import the Blynk library

# --- BLYNK CONSTANTS ---
BLYNK_TEMPLATE_ID = "TMPL3pw1yJ2gD"
BLYNK_TEMPLATE_NAME = "Sleep Alert"
BLYNK_AUTH_TOKEN = "xyerCeiN6CBFVmSKUVfoKuZsZ8s1TndY"

# --- NEW: Blynk Integration Class ---
class BlynkManager:
    """
    Manages all communication with the Blynk server in a separate thread.
    """
    def __init__(self, auth_token):
        self.blynk = blynklib.Blynk(auth_token, server='blynk.cloud', port=80)
        self.is_active = False
        self.stop_event = threading.Event()
        self.thread = None
        self.last_green_led_toggle = 0
        self.green_led_state = 0

        # Define Virtual Pins
        self.V_GREEN_LED = 1
        self.V_YELLOW_LED = 2
        self.V_ALARM = 3

        # Timers for temporary alerts
        self.yellow_led_timer = None
        self.alarm_timer = None

    def _run_blynk(self):
        """The main loop that runs in the background thread."""
        while not self.stop_event.is_set():
            try:
                self.blynk.run()
                current_time = time.time()

                # Handle green LED heartbeat (blinks every 2 seconds)
                if self.is_active and (current_time - self.last_green_led_toggle > 1):
                    self.green_led_state = 255 if self.green_led_state == 0 else 0
                    self.blynk.virtual_write(self.V_GREEN_LED, self.green_led_state)
                    self.last_green_led_toggle = current_time
                elif not self.is_active and self.green_led_state != 0:
                    # Ensure green LED is off when stopped
                    self.green_led_state = 0
                    self.blynk.virtual_write(self.V_GREEN_LED, 0)

            except Exception as e:
                print(f"Blynk connection error: {e}")
                time.sleep(5) # Wait before retrying
            time.sleep(0.02)

    def start(self):
        """Starts the Blynk communication thread."""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run_blynk, daemon=True)
            self.thread.start()
            print("Blynk Manager started.")

    def stop(self):
        """Stops the Blynk communication thread gracefully."""
        self.is_active = False
        self.stop_event.set()
        # Turn off all indicators when stopping
        try:
            self.blynk.virtual_write(self.V_GREEN_LED, 0)
            self.blynk.virtual_write(self.V_YELLOW_LED, 0)
            self.blynk.virtual_write(self.V_ALARM, 0)
        except:
            pass # Ignore errors on shutdown
        if self.thread:
            self.thread.join(timeout=2)
        print("Blynk Manager stopped.")

    def set_active_status(self, active: bool):
        """Tells the manager if the main detection is running."""
        self.is_active = active

    def _turn_off_yellow_led(self):
        self.blynk.virtual_write(self.V_YELLOW_LED, 0)

    def _turn_off_alarm(self):
        self.blynk.virtual_write(self.V_ALARM, 0)

    def trigger_yawn_alert(self):
        """Blinks the yellow LED for 4 seconds."""
        if self.is_active:
            print("Blynk: Yawn detected, triggering V2.")
            self.blynk.virtual_write(self.V_YELLOW_LED, 255) # Turn on
            # Cancel previous timer if it exists
            if self.yellow_led_timer and self.yellow_led_timer.is_alive():
                self.yellow_led_timer.cancel()
            self.yellow_led_timer = threading.Timer(4.0, self._turn_off_yellow_led)
            self.yellow_led_timer.start()

    def trigger_high_risk_alert(self):
        """Turns on the alarm for 6 seconds."""
        if self.is_active:
            print("Blynk: High risk detected, triggering V3.")
            self.blynk.virtual_write(self.V_ALARM, 255) # Turn on
            # Cancel previous timer if it exists
            if self.alarm_timer and self.alarm_timer.is_alive():
                self.alarm_timer.cancel()
            self.alarm_timer = threading.Timer(6.0, self._turn_off_alarm)
            self.alarm_timer.start()


# --- Core Drowsiness Detection Logic ---
class DrowsinessDetector:
    """
    Handles the core logic for eye state, yawn detection, and drowsiness evaluation.
    This class is designed to be run in a background thread.
    """
    # MODIFIED: Added blynk_manager to __init__
    def __init__(self, eye_model_path="weights/best.pt", yawn_model_path="weights2/best.pt", blynk_manager=None):
        # --- Model Initialization ---
        self.detect_eye = YOLO(eye_model_path)
        self.detect_yawn = YOLO(yawn_model_path)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Landmark points for ROIs ---
        self.left_eye_points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.right_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.mouth_points = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


        # --- State Variables ---
        self.left_eye_state = 'Open Eye'
        self.right_eye_state = 'Open Eye'
        self.yawn_state = 'No Yawn'
        self.yawning = False
        self.eyes_closed_start_time = None
        self.blinks = 0
        self.eyes_were_open = True # State for original blink logic

        # --- Advanced Logic: Event History Tracking ---
        self.time_window_seconds = 300  # 5 minutes
        self.microsleep_threshold_seconds = 1.0
        self.yawn_timestamps = deque()
        self.microsleep_timestamps = deque()
        self.high_risk_timestamps = deque()
        self.alert_message = ""
        
        # --- NEW: Blynk Manager instance ---
        self.blynk_manager = blynk_manager

    def predict_state(self, model, frame, current_state, open_class_id, closed_class_id, conf_threshold):
        """Generic prediction function for eye or yawn models."""
        if frame is None or frame.size == 0:
            return current_state
        try:
            results = model.predict(frame, verbose=False)
            if not results[0].boxes:
                return current_state
            
            box = results[0].boxes[0]
            class_id = int(box.cls.item())
            confidence = box.conf.item()

            if class_id == closed_class_id:
                return "Close Eye" if model == self.detect_eye else "Yawn"
            elif class_id == open_class_id and confidence > conf_threshold:
                return "Open Eye" if model == self.detect_eye else "No Yawn"
            
            return current_state
        except Exception:
            return current_state

    def extract_rois(self, frame, face_landmarks):
        """Extracts Regions of Interest (ROIs) for eyes and mouth."""
        ih, iw, _ = frame.shape
        padding = 10
        
        def get_bbox(points_ids):
            coords = [(int(face_landmarks.landmark[p].x * iw), int(face_landmarks.landmark[p].y * ih)) for p in points_ids]
            if not coords: return None
            x_coords, y_coords = zip(*coords)
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(iw, x2 + padding), min(ih, y2 + padding)
            return frame[y1:y2, x1:x2]

        left_eye = get_bbox(self.left_eye_points)
        right_eye = get_bbox(self.right_eye_points)
        mouth = get_bbox(self.mouth_points)
        return left_eye, right_eye, mouth

    def update_and_register_events(self):
        """Updates internal state based on predictions and registers significant events."""
        current_time = time.time()
        are_both_eyes_closed = self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye"
        is_currently_yawning = self.yawn_state == "Yawn"

        # --- Original Blink Counting Logic ---
        if are_both_eyes_closed and self.eyes_were_open:
            self.blinks += 1
        self.eyes_were_open = not are_both_eyes_closed

        # Handle Eye Closure and Microsleeps
        if are_both_eyes_closed:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
        else: # This block runs when eyes are open
            if self.eyes_closed_start_time is not None:
                duration = current_time - self.eyes_closed_start_time
                if duration >= self.microsleep_threshold_seconds:
                    
                    # --- FIX START: Correct High-Risk Event Logic ---
                    microsleep_start = self.eyes_closed_start_time
                    microsleep_end = current_time
                    
                    yawn_during_microsleep = False
                    if self.yawn_timestamps:
                        last_yawn_time = self.yawn_timestamps[-1]
                        if microsleep_start <= last_yawn_time <= microsleep_end:
                            yawn_during_microsleep = True

                    if yawn_during_microsleep:
                        self.high_risk_timestamps.append(current_time)
                    else:
                        self.microsleep_timestamps.append(current_time)
                    # --- FIX END ---
            
            # Reset the timer once the eyes are confirmed open
            self.eyes_closed_start_time = None

        # Handle Yawning (register on the transition to yawning)
        if is_currently_yawning and not self.yawning:
            self.yawn_timestamps.append(current_time)
            # --- NEW: Trigger Blynk Yawn Alert ---
            if self.blynk_manager:
                self.blynk_manager.trigger_yawn_alert()
        self.yawning = is_currently_yawning


    def evaluate_drowsiness_state(self):
        """Evaluates the overall drowsiness level based on the history of events."""
        current_time = time.time()
        # Clean up old events from the deques
        for dq in [self.yawn_timestamps, self.microsleep_timestamps, self.high_risk_timestamps]:
            while dq and current_time - dq[0] > self.time_window_seconds:
                dq.popleft()

        # Determine the alert message based on event frequency
        self.alert_message = ""
        if len(self.high_risk_timestamps) >= 2:
            self.alert_message = "ALERT: High Drowsiness Risk!"
            # --- NEW: Trigger Blynk High Risk Alert ---
            if self.blynk_manager:
                self.blynk_manager.trigger_high_risk_alert()
        elif len(self.microsleep_timestamps) >= 3:
            self.alert_message = "ALERT: Microsleep Risk Detected!"
        elif len(self.yawn_timestamps) >= 4:
            self.alert_message = "Warning: Frequent Yawning!"
            # We already trigger on single yawns, but you could add another trigger here if desired
            # for "frequent" yawns, e.g., a different color LED. For now, we'll keep it simple.

    def process_frame(self, frame):
        """Processes a single frame to detect face, eyes, mouth, and update drowsiness stats."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        
        display_message = ""

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_eye_roi, right_eye_roi, mouth_roi = self.extract_rois(frame, face_landmarks)

            # Predictions
            self.left_eye_state = self.predict_state(self.detect_eye, left_eye_roi, self.left_eye_state, 0, 1, 0.55)
            self.right_eye_state = self.predict_state(self.detect_eye, right_eye_roi, self.right_eye_state, 0, 1, 0.55)
            self.yawn_state = self.predict_state(self.detect_yawn, mouth_roi, self.yawn_state, 0, 1, 0.55)

            # Update logic
            self.update_and_register_events()
            self.evaluate_drowsiness_state()

            # Create a simple status message for the video overlay
            is_closed = self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye"
            if is_closed and self.yawning: display_message = "HIGH DROWSINESS"
            elif is_closed: display_message = "Eyes Closed"
            elif self.yawning: display_message = "Yawning"

        return frame, display_message

    def reset_counters(self):
        """Resets all counters and state variables."""
        self.blinks = 0
        self.yawn_timestamps.clear()
        self.microsleep_timestamps.clear()
        self.high_risk_timestamps.clear()
        self.alert_message = ""
        self.yawning = False
        self.eyes_closed_start_time = None
        self.left_eye_state = 'Open Eye'
        self.right_eye_state = 'Open Eye'
        self.yawn_state = 'No Yawn'
        self.eyes_were_open = True

# --- Video Processing Thread ---
def video_processor(detector, stop_event, result_queue, process_every_n_frames):
    """
    Runs in a separate thread to handle video capture and processing,
    preventing the Streamlit UI from freezing.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    frame_counter = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display_message = ""
        
        # Frame skipping for performance
        if frame_counter % process_every_n_frames == 0:
            processed_frame, display_message = detector.process_frame(frame)
        else:
            processed_frame = frame

        result = {
            "frame": cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
            "display_message": display_message,
            "blinks": detector.blinks,
            "yawns": len(detector.yawn_timestamps),
            "microsleeps": len(detector.microsleep_timestamps),
            "high_risk": len(detector.high_risk_timestamps),
            "alert": detector.alert_message
        }
        # Clear the queue to only keep the latest frame, preventing lag
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Empty:
                break
        result_queue.put(result)
        frame_counter += 1
        time.sleep(0.01)

    cap.release()

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Drowsiness Detection", page_icon="👁️", layout="wide")

    # Enhanced CSS styling with animations
    st.markdown("""
        <style>
            /* (Your existing CSS is unchanged) */
            .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .main-container { padding: 2rem; }
            .card { background-color: rgba(255, 255, 255, 0.95); border-radius: 12px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2); margin-bottom: 1rem; backdrop-filter: blur(10px); transition: all 0.3s ease; }
            .card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.15); }
            .status-box { border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; border-width: 2px; border-style: solid; transition: all 0.3s ease; position: relative; overflow: hidden; }
            .status-box::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent); transform: rotate(45deg); transition: all 0.6s ease; }
            .status-box:hover::before { top: -100%; left: -100%; }
            .status-attentive { background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%); border-color: #4CAF50; color: #2E7D32; animation: pulse-green 2s infinite; }
            .status-warning { background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%); border-color: #FF9800; color: #E65100; animation: pulse-orange 2s infinite; }
            .status-alert { background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); border-color: #F44336; color: #C62828; animation: pulse-red 1.5s infinite; }
            @keyframes pulse-green { 0%, 100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); } 50% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); } }
            @keyframes pulse-orange { 0%, 100% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.4); } 50% { box-shadow: 0 0 0 10px rgba(255, 152, 0, 0); } }
            @keyframes pulse-red { 0%, 100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); } 50% { box-shadow: 0 0 0 15px rgba(244, 67, 54, 0); } }
            .status-box h3 { font-size: 1.2rem; font-weight: 600; margin: 0; position: relative; z-index: 1; }
            .stButton>button { border-radius: 8px; padding: 10px 20px; font-weight: 500; border: 1px solid rgba(255,255,255,0.3); transition: all 0.3s ease; background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); color: #333; }
            .stButton>button:hover { transform: translateY(-2px) scale(1.02); box-shadow: 0 8px 25px rgba(0,0,0,0.15); background: rgba(255,255,255,0.2); }
            .stButton>button[kind="primary"] { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; border: none; }
            .stButton>button[kind="primary"]:hover { background: linear-gradient(135deg, #45a049 0%, #3d8b40 100%); }
            .stMetric { background: rgba(255,255,255,0.1) !important; border-radius: 8px !important; padding: 15px !important; margin: 5px 0 !important; border: 1px solid rgba(255,255,255,0.2) !important; transition: all 0.3s ease !important; backdrop-filter: blur(5px) !important; }
            .stMetric:hover { transform: translateY(-2px) !important; background: rgba(255,255,255,0.15) !important; box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important; }
            .stMetric label { color: rgba(255,255,255,0.9) !important; font-weight: 600 !important; }
            .stMetric [data-testid="metric-value"] { color: white !important; font-size: 1.5rem !important; font-weight: 700 !important; }
            .css-1d391kg { background: rgba(255,255,255,0.1) !important; backdrop-filter: blur(10px) !important; }
            .logo-container { display: flex; align-items: center; justify-content: center; padding: 10px; }
            .logo-container img { border-radius: 50%; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: all 0.3s ease; }
            .logo-container img:hover { transform: scale(1.05) rotate(5deg); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
            .main-title { animation: fadeInDown 1s ease-out; text-align: left !important; margin-left: 20px; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-size: 2.2rem; font-weight: 700; margin-top: 20px; }
            @keyframes fadeInDown { from { opacity: 0; transform: translate3d(0, -100%, 0); } to { opacity: 1; transform: translate3d(0, 0, 0); } }
            .video-container { border-radius: 12px; overflow: hidden; transition: all 0.3s ease; animation: fadeIn 1s ease-out; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            .video-container:hover { transform: scale(1.02); box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        </style>
    """, unsafe_allow_html=True)

    col_logo, col_title = st.columns([1, 4])
    with col_title:
             st.markdown('<h1 class="main-title">Driver Drowsiness Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Session State Initialization (with BlynkManager)
    if 'detection_active' not in st.session_state: st.session_state.detection_active = False
    if 'detector' not in st.session_state: st.session_state.detector = None
    if 'thread' not in st.session_state: st.session_state.thread = None
    if 'stop_event' not in st.session_state: st.session_state.stop_event = None
    if 'result_queue' not in st.session_state: st.session_state.result_queue = Queue(maxsize=1)
    if 'blynk_manager' not in st.session_state:
        st.session_state.blynk_manager = BlynkManager(BLYNK_AUTH_TOKEN)

    with st.sidebar:
        st.header("Performance Settings")
        st.markdown("---")
        process_each_frame = st.checkbox("Process Each Frame", value=False, help="Process every frame for maximum accuracy. May reduce performance.")
        if not process_each_frame:
            process_every_n_frames = st.slider("Process Every Nth Frame", min_value=2, max_value=10, value=4, step=1, help="Skip frames to improve performance")
        else:
            process_every_n_frames = 1

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Video Feed")
        video_placeholder = st.empty()
    with col2:
        st.subheader("Detection Status")
        status_placeholder = st.empty()
        st.markdown("**Controls**")
        c1, c2, c3 = st.columns(3)
        start_button = c1.button("Start", use_container_width=True, type="primary")
        stop_button = c2.button("Stop", use_container_width=True)
        reset_button = c3.button("Reset", use_container_width=True)
        st.markdown("---")
        st.markdown("**Live Metrics (5-minute window)**")
        metrics_placeholder = st.empty()

    if start_button and not st.session_state.detection_active:
        try:
            st.session_state.detection_active = True
            
            # Start and activate Blynk Manager
            st.session_state.blynk_manager.start()
            st.session_state.blynk_manager.set_active_status(True)

            # Pass the manager to the detector
            detector = DrowsinessDetector(blynk_manager=st.session_state.blynk_manager)
            
            st.session_state.detector = detector
            st.session_state.stop_event = threading.Event()
            st.session_state.result_queue = Queue(maxsize=1)
            st.session_state.thread = threading.Thread(
                target=video_processor,
                args=(detector, st.session_state.stop_event, st.session_state.result_queue, process_every_n_frames)
            )
            st.session_state.thread.start()
            st.rerun()
        except Exception as e:
            st.error(f"Error starting detection: {e}")
            st.session_state.detection_active = False

    if stop_button and st.session_state.detection_active:
        st.session_state.detection_active = False
        st.session_state.stop_event.set()
        
        # Deactivate and stop Blynk Manager
        st.session_state.blynk_manager.set_active_status(False)
        st.session_state.blynk_manager.stop()
        
        st.session_state.thread.join()
        st.rerun()

    if reset_button and st.session_state.detector:
        st.session_state.detector.reset_counters()
        st.success("Counters reset successfully!")

    if st.session_state.detection_active:
        while st.session_state.detection_active:
            try:
                result = st.session_state.result_queue.get(timeout=0.1)
                
                with video_placeholder.container():
                    st.markdown('<div class="video-container">', unsafe_allow_html=True)
                    st.image(result["frame"], channels="RGB", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                alert_msg = result["alert"]
                if alert_msg:
                    if "Warning" in alert_msg:
                        status_text, status_class = ("Warning: Frequent Yawning", "status-warning")
                    else:
                        status_text, status_class = ("ALERT: Drowsiness Detected", "status-alert")
                else:
                    status_text, status_class = ("Driver Alert", "status-attentive")
                
                status_placeholder.markdown(f'<div class="status-box {status_class}"><h3>{status_text}</h3></div>', unsafe_allow_html=True)
                
                with metrics_placeholder.container():
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Blinks", result["blinks"])
                    mc2.metric("Yawns", result["yawns"])
                    mc3, mc4 = st.columns(2)
                    mc3.metric("Microsleeps", result["microsleeps"])
                    mc4.metric("High Risk Events", result["high_risk"])

            except Empty:
                time.sleep(0.01)
                if not st.session_state.detection_active:
                    break
    else:
        with video_placeholder.container():
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.image("https://images.vexels.com/media/users/3/134989/isolated/preview/90c81e3560efefd0819e1c71feecf3be-camera-off-button-flat-icon.png", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        status_placeholder.markdown('<div class="status-box" style="background-color: #F5F5F5; border-color: #CCCCCC; color: #666666;"><h3>Detection Stopped</h3></div>', unsafe_allow_html=True)
        
        with metrics_placeholder.container():
            mc1, mc2 = st.columns(2)
            mc1.metric("Blinks", 0)
            mc2.metric("Yawns", 0)
            mc3, mc4 = st.columns(2)
            mc3.metric("Microsleeps", 0)
            mc4.metric("High Risk Events", 0)

if __name__ == "__main__":
    main()




