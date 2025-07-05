"""
thothos_integration.py – ThothOS integration for 3D realm-building system.
Provides seamless integration between ThothOS and the realm-building service.
"""

import sys
import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from realm_service import get_service, create_realm, place_object, cast_glyph, list_realms
    SERVICE_AVAILABLE = True
except ImportError:
    SERVICE_AVAILABLE = False
    print("Warning: realm_service not available")

class ThothOSIntegration:
    """Integration layer between ThothOS and the realm-building system."""
    
    def __init__(self):
        """Initialize the ThothOS integration."""
        self.service = get_service() if SERVICE_AVAILABLE else None
        self.active_realms = {}
        self.request_history = []
        self.integration_status = "initializing"
        
        if self.service:
            self.integration_status = "ready"
            print("✅ ThothOS integration initialized successfully")
        else:
            self.integration_status = "error"
            print("❌ Failed to initialize ThothOS integration")
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            "integration_status": self.integration_status,
            "service_available": SERVICE_AVAILABLE,
            "active_realms": len(self.active_realms),
            "request_history": len(self.request_history),
            "timestamp": time.time()
        }
    
    def handle_thothos_command(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle commands from ThothOS."""
        if not self.service:
            return {"success": False, "error": "Service not available"}
        
        parameters = parameters or {}
        command_lower = command.lower()
        
        # Log the command
        self.request_history.append({
            "command": command,
            "parameters": parameters,
            "timestamp": time.time()
        })
        
        try:
            if "create realm" in command_lower or "new realm" in command_lower:
                return self._handle_create_realm(parameters)
            elif "place object" in command_lower or "add object" in command_lower:
                return self._handle_place_object(parameters)
            elif "cast glyph" in command_lower or "cast spell" in command_lower:
                return self._handle_cast_glyph(parameters)
            elif "list realms" in command_lower or "show realms" in command_lower:
                return self._handle_list_realms()
            elif "realm info" in command_lower or "get realm" in command_lower:
                return self._handle_get_realm_info(parameters)
            elif "demo realm" in command_lower or "create demo" in command_lower:
                return self._handle_create_demo_realm()
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_create_realm(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle realm creation command."""
        name = parameters.get("name", "ThothOS Realm")
        archetype = parameters.get("archetype", "forest_glade")
        description = parameters.get("description", "Realm created from ThothOS")
        properties = parameters.get("properties", {})
        
        result = create_realm(name, archetype, description, properties)
        
        if result["success"]:
            realm_id = result["realm_id"]
            self.active_realms[realm_id] = {
                "name": name,
                "archetype": archetype,
                "created_at": time.time(),
                "source": "thothos"
            }
        
        return result
    
    def _handle_place_object(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle object placement command."""
        realm_id = parameters.get("realm_id")
        if not realm_id:
            # Try to use the most recent realm
            if self.active_realms:
                realm_id = list(self.active_realms.keys())[-1]
            else:
                return {"success": False, "error": "No realm specified and no active realms"}
        
        object_type = parameters.get("object_type", "crystal")
        coordinates = parameters.get("coordinates", {"x": 0, "y": 0, "z": 0})
        properties = parameters.get("properties", {})
        
        return place_object(realm_id, object_type, coordinates, properties)
    
    def _handle_cast_glyph(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle glyph casting command."""
        realm_id = parameters.get("realm_id")
        if not realm_id:
            # Try to use the most recent realm
            if self.active_realms:
                realm_id = list(self.active_realms.keys())[-1]
            else:
                return {"success": False, "error": "No realm specified and no active realms"}
        
        glyph_type = parameters.get("glyph_type", "illumination")
        location = parameters.get("location", {"x": 0, "y": 0, "z": 0})
        caster = parameters.get("caster", "thothos_user")
        duration = parameters.get("duration", 3600)
        properties = parameters.get("properties", {})
        
        return cast_glyph(realm_id, glyph_type, location, caster, duration, properties)
    
    def _handle_list_realms(self) -> Dict[str, Any]:
        """Handle realm listing command."""
        result = list_realms()
        
        if result["success"]:
            # Update active realms
            for realm in result["realms"]:
                realm_id = realm["realm_id"]
                if realm_id not in self.active_realms:
                    self.active_realms[realm_id] = {
                        "name": realm["name"],
                        "archetype": realm["archetype"],
                        "created_at": realm["created_at"],
                        "source": "existing"
                    }
        
        return result
    
    def _handle_get_realm_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle realm info command."""
        realm_id = parameters.get("realm_id")
        if not realm_id:
            return {"success": False, "error": "No realm ID specified"}
        
        from realm_service import get_realm_info
        return get_realm_info(realm_id)
    
    def _handle_create_demo_realm(self) -> Dict[str, Any]:
        """Handle demo realm creation command."""
        from realm_service import create_demo_realm
        result = create_demo_realm()
        
        if result["success"]:
            realm_id = result["realm_id"]
            self.active_realms[realm_id] = {
                "name": "Demo Realm",
                "archetype": "forest_glade",
                "created_at": time.time(),
                "source": "thothos_demo"
            }
        
        return result
    
    def get_pending_requests(self) -> Dict[str, Any]:
        """Get pending requests for ThothOS to process."""
        if not self.service:
            return {"success": False, "error": "Service not available"}
        
        from realm_service import get_pending_requests
        return get_pending_requests()
    
    def export_for_thothos(self, filepath: str = None) -> Dict[str, Any]:
        """Export realm data in ThothOS-compatible format."""
        if not filepath:
            filepath = f"thothos_realms_{int(time.time())}.json"
        
        export_data = {
            "export_timestamp": time.time(),
            "integration_status": self.get_status(),
            "active_realms": self.active_realms,
            "request_history": self.request_history[-100:],  # Last 100 requests
            "realms_data": []
        }
        
        # Get detailed realm data
        realms_result = list_realms()
        if realms_result["success"]:
            for realm in realms_result["realms"]:
                realm_info = self._handle_get_realm_info({"realm_id": realm["realm_id"]})
                if realm_info["success"]:
                    export_data["realms_data"].append(realm_info["realm"])
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                "success": True,
                "message": f"ThothOS data exported to {filepath}",
                "filepath": filepath
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def import_from_thothos(self, filepath: str) -> Dict[str, Any]:
        """Import realm data from ThothOS export."""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Restore active realms
            if "active_realms" in import_data:
                self.active_realms.update(import_data["active_realms"])
            
            # Restore request history
            if "request_history" in import_data:
                self.request_history.extend(import_data["request_history"])
            
            return {
                "success": True,
                "message": f"ThothOS data imported from {filepath}",
                "realms_restored": len(import_data.get("active_realms", {}))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Global integration instance
thothos_integration = ThothOSIntegration()

def get_integration() -> ThothOSIntegration:
    """Get the global ThothOS integration instance."""
    return thothos_integration

def handle_thothos_command(command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle a command from ThothOS."""
    return thothos_integration.handle_thothos_command(command, parameters)

def get_integration_status() -> Dict[str, Any]:
    """Get ThothOS integration status."""
    return thothos_integration.get_status()

def export_for_thothos(filepath: str = None) -> Dict[str, Any]:
    """Export data for ThothOS."""
    return thothos_integration.export_for_thothos(filepath)

def import_from_thothos(filepath: str) -> Dict[str, Any]:
    """Import data from ThothOS."""
    return thothos_integration.import_from_thothos(filepath)

def get_pending_requests() -> Dict[str, Any]:
    """Get pending requests for ThothOS."""
    return thothos_integration.get_pending_requests()

# Example ThothOS command handlers
def thothos_create_realm(name: str, archetype: str = "forest_glade") -> Dict[str, Any]:
    """ThothOS command: Create a new realm."""
    return handle_thothos_command("create realm", {
        "name": name,
        "archetype": archetype,
        "description": f"Realm created by ThothOS: {name}"
    })

def thothos_place_object(object_type: str, x: float = 0, y: float = 0, z: float = 0) -> Dict[str, Any]:
    """ThothOS command: Place an object in the current realm."""
    return handle_thothos_command("place object", {
        "object_type": object_type,
        "coordinates": {"x": x, "y": y, "z": z}
    })

def thothos_cast_glyph(glyph_type: str, x: float = 0, y: float = 0, z: float = 0) -> Dict[str, Any]:
    """ThothOS command: Cast a glyph in the current realm."""
    return handle_thothos_command("cast glyph", {
        "glyph_type": glyph_type,
        "location": {"x": x, "y": y, "z": z}
    })

def thothos_list_realms() -> Dict[str, Any]:
    """ThothOS command: List all realms."""
    return handle_thothos_command("list realms")

def thothos_create_demo() -> Dict[str, Any]:
    """ThothOS command: Create a demo realm."""
    return handle_thothos_command("create demo realm")

# Test function
def test_thothos_integration():
    """Test the ThothOS integration."""
    print("🔗 Testing ThothOS Integration")
    print("=" * 40)
    
    # Check status
    status = get_integration_status()
    print(f"Integration Status: {status['integration_status']}")
    print(f"Service Available: {status['service_available']}")
    
    if not status['service_available']:
        print("❌ Service not available - cannot test integration")
        return False
    
    # Test commands
    print("\n🎮 Testing ThothOS Commands:")
    
    # Create realm
    print("\n1. Creating realm...")
    result = thothos_create_realm("ThothOS Test Realm", "mountain_peak")
    if result["success"]:
        print(f"✅ {result['message']}")
        realm_id = result["realm_id"]
    else:
        print(f"❌ Failed: {result['error']}")
        return False
    
    # Place object
    print("\n2. Placing object...")
    result = thothos_place_object("crystal", 5, 0, 5)
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # Cast glyph
    print("\n3. Casting glyph...")
    result = thothos_cast_glyph("illumination", 0, 0, 0)
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # List realms
    print("\n4. Listing realms...")
    result = thothos_list_realms()
    if result["success"]:
        print(f"✅ Found {result['count']} realms")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # Export data
    print("\n5. Exporting data...")
    result = export_for_thothos("thothos_test_export.json")
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    print("\n✅ ThothOS integration test completed")
    return True

if __name__ == "__main__":
    success = test_thothos_integration()
    sys.exit(0 if success else 1) 