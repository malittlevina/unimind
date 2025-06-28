

# prosthetic_interface.py

import time
import serial  # For USB or Bluetooth serial connections

class ProstheticInterface:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.connection = None

    def connect(self):
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"[ProstheticInterface] Connected to {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"[ProstheticInterface] Connection error: {e}")

    def disconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            print(f"[ProstheticInterface] Disconnected from {self.port}.")

    def send_command(self, command):
        if self.connection and self.connection.is_open:
            self.connection.write(command.encode('utf-8'))
            print(f"[ProstheticInterface] Sent command: {command}")
        else:
            print("[ProstheticInterface] No active connection.")

    def read_data(self):
        if self.connection and self.connection.in_waiting > 0:
            data = self.connection.readline().decode('utf-8').strip()
            print(f"[ProstheticInterface] Received: {data}")
            return data
        return None

    def run_diagnostics(self):
        print("[ProstheticInterface] Running diagnostics...")
        self.send_command("DIAGNOSE")
        time.sleep(2)
        response = self.read_data()
        print(f"[ProstheticInterface] Diagnostic result: {response}")
        return response