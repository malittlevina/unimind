"""
storyrealms_bridge.py – Symbolic bridge for 3D realm-building systems.
Provides symbolic interfaces for Unity, Unreal, and other 3D engines.
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

class RealmArchetype(Enum):
    """Predefined realm archetypes for quick creation."""
    FOREST_GLADE = "forest_glade"
    MOUNTAIN_PEAK = "mountain_peak"
    OCEAN_DEPTHS = "ocean_depths"
    DESERT_DUNES = "desert_dunes"
    COSMIC_VOID = "cosmic_void"
    CRYSTAL_CAVE = "crystal_cave"
    FLOATING_ISLANDS = "floating_islands"
    UNDERWATER_CITY = "underwater_city"
    TIME_TEMPLE = "time_temple"
    DREAM_GARDEN = "dream_garden"

class ObjectType(Enum):
    """Predefined object types for realm placement."""
    # Natural objects
    TREE = "tree"
    ROCK = "rock"
    WATER = "water"
    FIRE = "fire"
    CRYSTAL = "crystal"
    FLOWER = "flower"
    
    # Architectural elements
    PILLAR = "pillar"
    ARCHWAY = "archway"
    STAIRS = "stairs"
    BRIDGE = "bridge"
    TOWER = "tower"
    TEMPLE = "temple"
    
    # Interactive elements
    PORTAL = "portal"
    GATEWAY = "gateway"
    ALTAR = "altar"
    FOUNTAIN = "fountain"
    MIRROR = "mirror"
    DOOR = "door"
    
    # Magical elements
    GLYPH = "glyph"
    RUNE = "rune"
    ORB = "orb"
    WARD = "ward"
    BEACON = "beacon"
    NEXUS = "nexus"

class GlyphType(Enum):
    """Types of magical glyphs that can be cast in realms."""
    PROTECTION = "protection"
    ILLUMINATION = "illumination"
    TELEPORTATION = "teleportation"
    TRANSFORMATION = "transformation"
    COMMUNICATION = "communication"
    HEALING = "healing"
    WARDING = "warding"
    SUMMONING = "summoning"
    BINDING = "binding"
    REVELATION = "revelation"

@dataclass
class Coordinates:
    """3D coordinates for object placement."""
    x: float
    y: float
    z: float
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0

@dataclass
class RealmObject:
    """Represents an object placed in a realm."""
    object_id: str
    object_type: ObjectType
    coordinates: Coordinates
    properties: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class Realm:
    """Represents a 3D realm with its properties and objects."""
    realm_id: str
    name: str
    archetype: RealmArchetype
    description: str
    created_at: float
    modified_at: float
    objects: List[RealmObject]
    properties: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class GlyphCast:
    """Represents a glyph cast in a realm."""
    glyph_id: str
    glyph_type: GlyphType
    location: Coordinates
    caster: str
    target: Optional[str] = None
    duration: Optional[float] = None
    properties: Dict[str, Any] = None

class StoryrealmsBridge:
    """
    Symbolic bridge for 3D realm-building systems.
    Provides interfaces for Unity, Unreal, and other 3D engines.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the bridge with configuration."""
        self.logger = logging.getLogger('StoryrealmsBridge')
        self.realms: Dict[str, Realm] = {}
        self.active_realm: Optional[str] = None
        self.engine_type: str = "symbolic"  # symbolic, unity, unreal
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize request queue for external engines
        self.request_queue: List[Dict[str, Any]] = []
        
        # Create output directory for engine requests
        self.output_dir = Path("unimind/bridge/engine_requests")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("StoryrealmsBridge initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load bridge configuration."""
        default_config = {
            "engine_type": "symbolic",
            "unity_project_path": "external_engines/unity_project",
            "unreal_project_path": "external_engines/unreal_project",
            "auto_save": True,
            "log_requests": True,
            "max_objects_per_realm": 1000,
            "default_realm_properties": {
                "ambient_lighting": "natural",
                "weather": "clear",
                "time_of_day": "noon",
                "atmosphere": "peaceful"
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def create_realm(self, name: str, archetype: Union[RealmArchetype, str], 
                    description: str = "", properties: Dict[str, Any] = None) -> str:
        """
        Create a new 3D realm.
        
        Args:
            name: Name of the realm
            archetype: Predefined realm archetype
            description: Description of the realm
            properties: Additional realm properties
            
        Returns:
            Realm ID
        """
        try:
            # Convert string to enum if needed
            if isinstance(archetype, str):
                archetype = RealmArchetype(archetype)
            
            # Generate unique realm ID
            realm_id = str(uuid.uuid4())
            current_time = time.time()
            
            # Create realm object
            realm = Realm(
                realm_id=realm_id,
                name=name,
                archetype=archetype,
                description=description,
                created_at=current_time,
                modified_at=current_time,
                objects=[],
                properties=properties or self.config["default_realm_properties"].copy(),
                metadata={
                    "created_by": "storyrealms_bridge",
                    "engine_type": self.engine_type,
                    "version": "1.0.0"
                }
            )
            
            # Store realm
            self.realms[realm_id] = realm
            
            # Log symbolic action
            self.logger.info(f"Created realm: {name} ({archetype.value})")
            
            # Generate engine request
            request = {
                "action": "create_realm",
                "realm_id": realm_id,
                "name": name,
                "archetype": archetype.value,
                "description": description,
                "properties": realm.properties,
                "timestamp": current_time
            }
            
            self._queue_request(request)
            
            return realm_id
            
        except Exception as e:
            self.logger.error(f"Failed to create realm: {e}")
            raise
    
    def place_object(self, realm_id: str, object_type: Union[ObjectType, str], 
                    coordinates: Union[Coordinates, Dict[str, float]], 
                    properties: Dict[str, Any] = None) -> str:
        """
        Place an object in a realm.
        
        Args:
            realm_id: ID of the target realm
            object_type: Type of object to place
            coordinates: 3D coordinates for placement
            properties: Object-specific properties
            
        Returns:
            Object ID
        """
        try:
            # Validate realm exists
            if realm_id not in self.realms:
                raise ValueError(f"Realm not found: {realm_id}")
            
            # Convert string to enum if needed
            if isinstance(object_type, str):
                object_type = ObjectType(object_type)
            
            # Convert dict to Coordinates if needed
            if isinstance(coordinates, dict):
                coordinates = Coordinates(**coordinates)
            
            # Generate unique object ID
            object_id = str(uuid.uuid4())
            
            # Create realm object
            realm_object = RealmObject(
                object_id=object_id,
                object_type=object_type,
                coordinates=coordinates,
                properties=properties or {},
                metadata={
                    "placed_by": "storyrealms_bridge",
                    "timestamp": time.time()
                }
            )
            
            # Add to realm
            self.realms[realm_id].objects.append(realm_object)
            self.realms[realm_id].modified_at = time.time()
            
            # Log symbolic action
            self.logger.info(f"Placed {object_type.value} in realm {realm_id}")
            
            # Generate engine request
            request = {
                "action": "place_object",
                "realm_id": realm_id,
                "object_id": object_id,
                "object_type": object_type.value,
                "coordinates": asdict(coordinates),
                "properties": realm_object.properties,
                "timestamp": time.time()
            }
            
            self._queue_request(request)
            
            return object_id
            
        except Exception as e:
            self.logger.error(f"Failed to place object: {e}")
            raise
    
    def cast_glyph_in_realm(self, realm_id: str, glyph_type: Union[GlyphType, str], 
                           location: Union[Coordinates, Dict[str, float]], 
                           caster: str = "daemon", target: Optional[str] = None,
                           duration: Optional[float] = None, 
                           properties: Dict[str, Any] = None) -> str:
        """
        Cast a magical glyph in a realm.
        
        Args:
            realm_id: ID of the target realm
            glyph_type: Type of glyph to cast
            location: Location for the glyph
            caster: Who is casting the glyph
            target: Target of the glyph (optional)
            duration: Duration of the glyph effect (optional)
            properties: Glyph-specific properties
            
        Returns:
            Glyph ID
        """
        try:
            # Validate realm exists
            if realm_id not in self.realms:
                raise ValueError(f"Realm not found: {realm_id}")
            
            # Convert string to enum if needed
            if isinstance(glyph_type, str):
                glyph_type = GlyphType(glyph_type)
            
            # Convert dict to Coordinates if needed
            if isinstance(location, dict):
                location = Coordinates(**location)
            
            # Generate unique glyph ID
            glyph_id = str(uuid.uuid4())
            
            # Create glyph cast
            glyph_cast = GlyphCast(
                glyph_id=glyph_id,
                glyph_type=glyph_type,
                location=location,
                caster=caster,
                target=target,
                duration=duration,
                properties=properties or {}
            )
            
            # Log symbolic action
            self.logger.info(f"Cast {glyph_type.value} glyph in realm {realm_id}")
            
            # Generate engine request
            request = {
                "action": "cast_glyph",
                "realm_id": realm_id,
                "glyph_id": glyph_id,
                "glyph_type": glyph_type.value,
                "location": asdict(location),
                "caster": caster,
                "target": target,
                "duration": duration,
                "properties": glyph_cast.properties,
                "timestamp": time.time()
            }
            
            self._queue_request(request)
            
            return glyph_id
            
        except Exception as e:
            self.logger.error(f"Failed to cast glyph: {e}")
            raise
    
    def load_realm(self, realm_id: str) -> bool:
        """
        Load a realm as the active realm.
        
        Args:
            realm_id: ID of the realm to load
            
        Returns:
            Success status
        """
        try:
            if realm_id not in self.realms:
                raise ValueError(f"Realm not found: {realm_id}")
            
            self.active_realm = realm_id
            
            # Generate engine request
            request = {
                "action": "load_realm",
                "realm_id": realm_id,
                "timestamp": time.time()
            }
            
            self._queue_request(request)
            
            self.logger.info(f"Loaded realm: {realm_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load realm: {e}")
            return False
    
    def modify_realm(self, realm_id: str, properties: Dict[str, Any]) -> bool:
        """
        Modify realm properties.
        
        Args:
            realm_id: ID of the realm to modify
            properties: New properties to apply
            
        Returns:
            Success status
        """
        try:
            if realm_id not in self.realms:
                raise ValueError(f"Realm not found: {realm_id}")
            
            # Update realm properties
            self.realms[realm_id].properties.update(properties)
            self.realms[realm_id].modified_at = time.time()
            
            # Generate engine request
            request = {
                "action": "modify_realm",
                "realm_id": realm_id,
                "properties": properties,
                "timestamp": time.time()
            }
            
            self._queue_request(request)
            
            self.logger.info(f"Modified realm: {realm_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to modify realm: {e}")
            return False
    
    def remove_object(self, realm_id: str, object_id: str) -> bool:
        """
        Remove an object from a realm.
        
        Args:
            realm_id: ID of the realm
            object_id: ID of the object to remove
            
        Returns:
            Success status
        """
        try:
            if realm_id not in self.realms:
                raise ValueError(f"Realm not found: {realm_id}")
            
            # Find and remove object
            realm = self.realms[realm_id]
            for i, obj in enumerate(realm.objects):
                if obj.object_id == object_id:
                    del realm.objects[i]
                    realm.modified_at = time.time()
                    
                    # Generate engine request
                    request = {
                        "action": "remove_object",
                        "realm_id": realm_id,
                        "object_id": object_id,
                        "timestamp": time.time()
                    }
                    
                    self._queue_request(request)
                    
                    self.logger.info(f"Removed object {object_id} from realm {realm_id}")
                    return True
            
            raise ValueError(f"Object not found: {object_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to remove object: {e}")
            return False
    
    def get_realm_info(self, realm_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a realm."""
        if realm_id not in self.realms:
            return None
        
        realm = self.realms[realm_id]
        return {
            "realm_id": realm.realm_id,
            "name": realm.name,
            "archetype": realm.archetype.value,
            "description": realm.description,
            "created_at": realm.created_at,
            "modified_at": realm.modified_at,
            "object_count": len(realm.objects),
            "properties": realm.properties,
            "metadata": realm.metadata
        }
    
    def list_realms(self) -> List[Dict[str, Any]]:
        """List all available realms."""
        return [self.get_realm_info(realm_id) for realm_id in self.realms.keys()]
    
    def _queue_request(self, request: Dict[str, Any]) -> None:
        """Queue a request for external engine processing."""
        self.request_queue.append(request)
        
        # Save request to file for external engines
        if self.config.get("log_requests", True):
            timestamp = int(time.time())
            filename = f"request_{timestamp}_{request['action']}.json"
            filepath = self.output_dir / filename
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(request, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save request: {e}")
    
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending requests for external engines."""
        return self.request_queue.copy()
    
    def clear_request_queue(self) -> None:
        """Clear the request queue."""
        self.request_queue.clear()
    
    def save_realm_state(self, realm_id: str, filepath: str) -> bool:
        """Save realm state to file."""
        try:
            if realm_id not in self.realms:
                raise ValueError(f"Realm not found: {realm_id}")
            
            realm = self.realms[realm_id]
            state = {
                "realm": asdict(realm),
                "saved_at": time.time(),
                "version": "1.0.0"
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Saved realm state: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save realm state: {e}")
            return False
    
    def load_realm_state(self, filepath: str) -> bool:
        """Load realm state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            realm_data = state["realm"]
            realm = Realm(**realm_data)
            
            self.realms[realm.realm_id] = realm
            
            self.logger.info(f"Loaded realm state: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load realm state: {e}")
            return False

# Global bridge instance
storyrealms_bridge = StoryrealmsBridge()

# Convenience functions for scroll integration
def create_realm(name: str, archetype: Union[RealmArchetype, str], 
                description: str = "", properties: Dict[str, Any] = None) -> str:
    """Create a new realm using the global bridge."""
    return storyrealms_bridge.create_realm(name, archetype, description, properties)

def place_object(realm_id: str, object_type: Union[ObjectType, str], 
                coordinates: Union[Coordinates, Dict[str, float]], 
                properties: Dict[str, Any] = None) -> str:
    """Place an object in a realm using the global bridge."""
    return storyrealms_bridge.place_object(realm_id, object_type, coordinates, properties)

def cast_glyph_in_realm(realm_id: str, glyph_type: Union[GlyphType, str], 
                       location: Union[Coordinates, Dict[str, float]], 
                       caster: str = "daemon", target: Optional[str] = None,
                       duration: Optional[float] = None, 
                       properties: Dict[str, Any] = None) -> str:
    """Cast a glyph in a realm using the global bridge."""
    return storyrealms_bridge.cast_glyph_in_realm(realm_id, glyph_type, location, 
                                                 caster, target, duration, properties) 