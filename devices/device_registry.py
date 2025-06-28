

# device_registry.py

class DeviceRegistry:
    def __init__(self):
        self.devices = {}

    def register_device(self, device_id, device_info):
        """Registers a new device with the given ID and metadata."""
        self.devices[device_id] = device_info

    def get_device(self, device_id):
        """Retrieves information about a registered device."""
        return self.devices.get(device_id, None)

    def remove_device(self, device_id):
        """Removes a device from the registry."""
        if device_id in self.devices:
            del self.devices[device_id]

    def list_devices(self):
        """Returns a list of all registered device IDs."""
        return list(self.devices.keys())