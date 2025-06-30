import logging

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) not available. Object tracking will be limited.")

class ObjectTracker:
    def __init__(self, tracker_type="CSRT"):
        self.tracker_type = tracker_type
        self.trackers = {}
        self.next_id = 0
        
        if CV2_AVAILABLE:
            self.tracker_available = True
        else:
            self.tracker_available = False
            
        logging.info("ObjectTracker initialized.")

    def add_tracker(self, frame, bbox):
        if not CV2_AVAILABLE:
            return None
            
        try:
            tracker = cv2.TrackerCSRT_create()
            success = tracker.init(frame, bbox)
            if success:
                tracker_id = self.next_id
                self.trackers[tracker_id] = tracker
                self.next_id += 1
                return tracker_id
        except Exception as e:
            logging.error(f"Error adding tracker: {e}")
        return None

    def update_trackers(self, frame):
        if not CV2_AVAILABLE:
            return {"tracked_objects": [], "message": "OpenCV not available"}
            
        results = []
        for tracker_id, tracker in self.trackers.items():
            try:
                success, bbox = tracker.update(frame)
                if success:
                    results.append({
                        "id": tracker_id,
                        "bbox": bbox,
                        "status": "tracking"
                    })
                else:
                    results.append({
                        "id": tracker_id,
                        "status": "lost"
                    })
            except Exception as e:
                logging.error(f"Error updating tracker {tracker_id}: {e}")
                results.append({
                    "id": tracker_id,
                    "status": "error"
                })
        
        return {"tracked_objects": results}

    def remove_tracker(self, tracker_id):
        if tracker_id in self.trackers:
            del self.trackers[tracker_id]
            return True
        return False