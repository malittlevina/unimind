import logging

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) not available. Scene classification will be limited.")

class SceneClassifier:
    def __init__(self, model_path: str = "mobilenet.onnx", labels_path: str = "labels.txt", confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        
        if CV2_AVAILABLE:
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            except Exception as e:
                logging.warning(f"Could not load model: {e}")
                self.net = None
        else:
            self.net = None
            
        self.labels = self._load_labels()
        logging.info("SceneClassifier initialized.")

    def _load_labels(self):
        try:
            with open(self.labels_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            logging.warning("Labels file not found.")
            return ["Unknown"] * 1000

    def classify_frame(self, frame):
        if not CV2_AVAILABLE or self.net is None:
            return {
                "label": "OpenCV not available",
                "confidence": 0.0
            }
            
        try:
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/255, size=(224, 224), mean=(0, 0, 0), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward()

            class_id = np.argmax(outputs)
            confidence = outputs[0][class_id]

            if confidence > self.confidence_threshold:
                label = self.labels[class_id] if class_id < len(self.labels) else "Unknown"
                return {
                    "label": label,
                    "confidence": float(confidence)
                }
            else:
                return {
                    "label": "Uncertain",
                    "confidence": float(confidence)
                }
        except Exception as e:
            logging.error(f"Error in classify_frame: {e}")
            return {
                "label": "Error during classification",
                "confidence": 0.0
            }

    def classify_from_camera(self, camera_index: int = 0):
        if not CV2_AVAILABLE:
            return {
                "label": "OpenCV not available",
                "confidence": 0.0
            }
            
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise IOError("Cannot open camera")

            ret, frame = cap.read()
            cap.release()

            if ret:
                return self.classify_frame(frame)
            else:
                return {
                    "label": "No frame captured",
                    "confidence": 0.0
                }
        except Exception as e:
            logging.error(f"Error in classify_from_camera: {e}")
            return {
                "label": "Camera error",
                "confidence": 0.0
            }