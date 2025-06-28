


# Placeholder configuration structure for all supported sensors
SENSOR_CONFIG = {
    "camera": {
        "enabled": True,
        "resolution": "720p",
        "fps": 30,
        "device_index": 0
    },
    "microphone": {
        "enabled": True,
        "sample_rate": 44100,
        "channels": 1,
        "device_index": None
    },
    "temperature": {
        "enabled": False,
        "unit": "Celsius",
        "pin": "A0"
    },
    "motion": {
        "enabled": False,
        "type": "accelerometer",
        "interface": "I2C"
    }
}


# Helper function to retrieve sensor settings
def get_sensor_settings(sensor_name):
    return SENSOR_CONFIG.get(sensor_name, {})