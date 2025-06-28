

import cv2

class CameraStream:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture = None

    def start_stream(self):
        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.capture.isOpened():
            raise Exception("Unable to open camera")

        print(f"[CameraStream] Started stream on camera index {self.camera_index}")

    def read_frame(self):
        if not self.capture:
            raise Exception("Camera stream not started")

        ret, frame = self.capture.read()
        if not ret:
            raise Exception("Failed to read frame from camera")

        return frame

    def stop_stream(self):
        if self.capture:
            self.capture.release()
            self.capture = None
            print("[CameraStream] Stopped stream")

    def __del__(self):
        self.stop_stream()