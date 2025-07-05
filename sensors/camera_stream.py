import logging
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) not available. Camera stream will be limited.")

class CameraStream:
    def __init__(self, camera_index=0, width=640, height=480):
        self.capture = None
        self.camera_index = camera_index
        self.width = width
        self.height = height
        if CV2_AVAILABLE:
            self.capture = cv2.VideoCapture(self.camera_index)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def start_stream(self):
        if not CV2_AVAILABLE or self.capture is None:
            logging.warning("OpenCV (cv2) not available or camera not initialized. Cannot start camera stream.")
            return
        print(f"[CameraStream] Started stream on camera index {self.camera_index}")

    def read(self):
        if not CV2_AVAILABLE or self.capture is None:
            logging.warning("OpenCV (cv2) not available or camera not initialized. Cannot read camera stream.")
            return None
        ret, frame = self.capture.read()
        return frame if ret else None

    def stop_stream(self):
        if self.capture:
            self.capture.release()
            self.capture = None
            print("[CameraStream] Stopped stream")

    def __del__(self):
        self.stop_stream()