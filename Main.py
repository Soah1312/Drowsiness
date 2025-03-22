from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.uix.switch import Switch
from kivy.uix.label import Label
from kivy.clock import Clock
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

KV = """
BoxLayout:
    orientation: 'vertical'
    padding: 20
    spacing: 20

    Label:
        text: "Drowsiness Detection"
        font_size: '24sp'
        bold: True
        size_hint_y: None
        height: 50

    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: 50
        Label:
            text: "Enable Monitoring"
        Switch:
            id: monitoring_switch
            active: False

    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: 50
        Label:
            text: "Enable Beep Sound"
        Switch:
            id: beep_switch
            active: False

    Label:
        id: status_label
        text: "Monitoring is OFF"
        font_size: '18sp'
        bold: True
        color: (1, 0, 0, 1)  # Red when OFF
"""


class DrowsinessApp(App):
    def build(self):
        self.ui = Builder.load_string(KV)
        self.monitoring = False
        self.beep_enabled = False
        self.cap = None
        self.eye_start_time = None
        self.tilt_start_time = None

        self.monitoring_switch = self.ui.ids.monitoring_switch
        self.beep_switch = self.ui.ids.beep_switch
        self.status_label = self.ui.ids.status_label

        self.monitoring_switch.bind(active=self.toggle_monitoring)
        self.beep_switch.bind(active=self.toggle_beep)

        return self.ui

    def toggle_monitoring(self, switch, value):
        if value:
            self.monitoring = True
            self.status_label.text = "Monitoring is ON"
            self.status_label.color = (0, 1, 0, 1)
            self.cap = cv2.VideoCapture(0)
            Clock.schedule_interval(self.detect_drowsiness, 1.0 / 30.0)
        else:
            self.monitoring = False
            self.status_label.text = "Monitoring is OFF"
            self.status_label.color = (1, 0, 0, 1)
            Clock.unschedule(self.detect_drowsiness)
            if self.cap:
                self.cap.release()
                self.cap = None

    def toggle_beep(self, switch, value):
        self.beep_enabled = value

    def detect_drowsiness(self, dt):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                height, width, _ = frame.shape
                landmarks = [(int(l.x * width), int(l.y * height)) for l in face_landmarks.landmark]

                LEFT_EYE = [362, 385, 387, 263, 373, 380]
                RIGHT_EYE = [33, 160, 158, 133, 153, 144]
                HEAD_TILT_POINTS = [10, 152]

                def calculate_ear(eye_points, landmarks):
                    A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
                    B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
                    C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
                    return (A + B) / (2.0 * C)

                left_ear = calculate_ear(LEFT_EYE, landmarks)
                right_ear = calculate_ear(RIGHT_EYE, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                top_point = landmarks[HEAD_TILT_POINTS[0]]
                bottom_point = landmarks[HEAD_TILT_POINTS[1]]
                angle = np.degrees(np.arctan2(bottom_point[0] - top_point[0], bottom_point[1] - top_point[1]))

                DROWSY_THRESHOLD = 0.25
                TIME_THRESHOLD = 5.0
                TILT_THRESHOLD = 15

                is_drowsy = False

                if avg_ear < DROWSY_THRESHOLD:
                    if self.eye_start_time is None:
                        self.eye_start_time = time.time()
                    elapsed_eye_time = time.time() - self.eye_start_time

                    if elapsed_eye_time >= TIME_THRESHOLD:
                        is_drowsy = True

                else:
                    self.eye_start_time = None

                if abs(angle) > TILT_THRESHOLD:
                    if self.tilt_start_time is None:
                        self.tilt_start_time = time.time()
                    elapsed_tilt_time = time.time() - self.tilt_start_time

                    if elapsed_tilt_time >= TIME_THRESHOLD:
                        is_drowsy = True

                else:
                    self.tilt_start_time = None

                if is_drowsy:
                    self.status_label.text = "ALERT! Drowsiness Detected!"
                    self.status_label.color = (1, 0, 0, 1)
                    if self.beep_enabled:
                        winsound.Beep(1000, 1000)
                else:
                    self.status_label.text = "Monitoring is ON"
                    self.status_label.color = (0, 1, 0, 1)


if __name__ == "__main__":
    DrowsinessApp().run()