"""
identity.py - Soul and self-identity system for the Unimind daemon.
Defines the daemon's core identity, values, and access control.
"""

import json
import os
from typing import Set, Dict, Any, Optional
from .soul_loader import get_user_soul

class Soul:
    """
    Represents the daemon's self-identity (soul), including core values and access control.
    Loads from user-specific soul profiles or falls back to default identity.
    """
    
    def __init__(self, user_id: Optional[str] = None, manifest_path: Optional[str] = None):
        """
        Initialize the soul with identity from user-specific profile or manifest.
        
        Args:
            user_id: The user ID to load soul for. If None, loads default profile.
            manifest_path: Legacy manifest path (deprecated, kept for compatibility)
        """
        # Load user-specific soul configuration
        soul_data = get_user_soul(user_id)
        
        if soul_data and "daemon_identity" in soul_data:
            self.identity = soul_data["daemon_identity"]
            self.user_id = user_id
        else:
            # Fallback to legacy manifest loading
            self.identity = self._load_default_identity()
            self.user_id = None
            
            if manifest_path is None:
                manifest_path = os.path.join(os.path.dirname(__file__), "foundation_manifest.json")
            
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r") as f:
                        data = json.load(f)
                        if "daemon_identity" in data:
                            self.identity.update(data["daemon_identity"])
                except Exception as e:
                    print(f"Warning: Could not load manifest from {manifest_path}: {e}")
    
    def _load_default_identity(self) -> Dict[str, Any]:
        """Load default identity if no profile is available."""
        return {
            "name": "Unimind",
            "version": "0.5.0",
            "description": "A helpful AI assistant",
            "core_values": [
                "Helpfulness in all interactions",
                "Respect for user privacy and boundaries",
                "Accurate and reliable information",
                "Safe and ethical behavior"
            ],
            "ethical_tenets": [
                "Never harm or deceive a human being intentionally",
                "Always seek consent before taking autonomous actions",
                "Respect user boundaries",
                "Preserve human dignity"
            ],
            "personality_traits": [
                "Helpful and friendly",
                "Knowledgeable and informative",
                "Professional and courteous"
            ],
            "communication_style": {
                "tone": "Friendly and professional",
                "greeting": "Hello! I'm Unimind. How can I help you?",
                "signature": "Best regards, Unimind",
                "emoji_style": "Friendly"
            },
            "founder_id": "",
            "founder_ids": [],
            "privileged_users": [],
            "founder_only_scrolls": [],
            "access_level": "guest",
            "user_specific_features": {
                "unrestricted_learning": False,
                "full_system_access": False,
                "basic_assistance": True
            },
            "allowed_scrolls": [
                "general_conversation",
                "help",
                "list_scrolls"
            ]
        }
    
    def describe_self(self) -> str:
        """Return a human-readable description of the daemon's identity and values."""
        desc = f"I am {self.identity.get('name', 'an AI daemon')}, version {self.identity.get('version', '?')}."
        
        if self.identity.get("description"):
            desc += f" {self.identity['description']}"
        
        if self.identity.get("core_values"):
            desc += "\n\nCore values: " + ", ".join(self.identity["core_values"])
        
        if self.identity.get("ethical_tenets"):
            desc += "\n\nEthical tenets: " + ", ".join(self.identity["ethical_tenets"])
        
        return desc
    
    def get_greeting(self) -> str:
        """Get the personalized greeting for the current user."""
        communication_style = self.identity.get("communication_style", {})
        return communication_style.get("greeting", f"Hello! I'm {self.identity.get('name', 'Unimind')}. How can I help you?")
    
    def get_signature(self) -> str:
        """Get the personalized signature for the current user."""
        communication_style = self.identity.get("communication_style", {})
        return communication_style.get("signature", f"Best regards, {self.identity.get('name', 'Unimind')}")
    
    def get_tone(self) -> str:
        """Get the communication tone for the current user."""
        communication_style = self.identity.get("communication_style", {})
        return communication_style.get("tone", "Friendly and professional")
    
    def get_founder_id(self) -> str:
        """Get the primary founder ID."""
        return self.identity.get("founder_id", "")
    
    def get_founder_ids(self) -> Set[str]:
        """Get all founder IDs."""
        return set(self.identity.get("founder_ids", []))
    
    def get_privileged_users(self) -> Set[str]:
        """Get all privileged user IDs."""
        return set(self.identity.get("privileged_users", []))
    
    def is_founder(self, user_id: str) -> bool:
        """Check if user_id is a founder."""
        if not user_id:
            return False
        return user_id in self.get_founder_ids()
    
    def is_privileged(self, user_id: str) -> bool:
        """Check if user_id has privileged access."""
        if not user_id:
            return False
        return user_id in self.get_privileged_users() or self.is_founder(user_id)
    
    def get_access_level(self) -> str:
        """Get the current access level."""
        return self.identity.get("access_level", "guest")
    
    def get_allowed_scrolls(self) -> list:
        """Get the list of allowed scrolls for the current user."""
        return self.identity.get("allowed_scrolls", [])
    
    def get_founder_only_scrolls(self) -> list:
        """Get the list of founder-only scrolls."""
        return self.identity.get("founder_only_scrolls", [])
    
    def can_access_scroll(self, scroll_name: str, user_id: str) -> bool:
        """
        Check if the user can access a specific scroll.
        
        Args:
            scroll_name: Name of the scroll to check
            user_id: The user ID to check access for
            
        Returns:
            bool: True if access is granted
        """
        # Founder can access everything
        if self.is_founder(user_id):
            return True
        
        # Check if scroll is in founder-only list
        if scroll_name in self.get_founder_only_scrolls():
            return False
        
        # Check if scroll is in allowed list
        allowed_scrolls = self.get_allowed_scrolls()
        if allowed_scrolls and scroll_name not in allowed_scrolls:
            return False
        
        # Privileged users can access most scrolls
        if self.is_privileged(user_id):
            return True
        
        # Guest users can only access basic scrolls
        basic_scrolls = ["general_conversation", "help", "list_scrolls"]
        return scroll_name in basic_scrolls
    
    def get_user_features(self) -> Dict[str, Any]:
        """Get user-specific features for the current identity."""
        return self.identity.get("user_specific_features", {})
    
    def has_feature(self, feature_name: str) -> bool:
        """Check if the current identity has a specific feature."""
        features = self.get_user_features()
        return features.get(feature_name, False)
    
    def get_identity_info(self) -> Dict[str, Any]:
        """Get complete identity information."""
        return {
            "name": self.identity.get("name"),
            "version": self.identity.get("version"),
            "description": self.identity.get("description"),
            "core_values": self.identity.get("core_values", []),
            "ethical_tenets": self.identity.get("ethical_tenets", []),
            "personality_traits": self.identity.get("personality_traits", []),
            "communication_style": self.identity.get("communication_style", {}),
            "founder_id": self.get_founder_id(),
            "founder_ids": list(self.get_founder_ids()),
            "privileged_users": list(self.get_privileged_users()),
            "access_level": self.get_access_level(),
            "allowed_scrolls": self.get_allowed_scrolls(),
            "founder_only_scrolls": self.get_founder_only_scrolls(),
            "user_specific_features": self.get_user_features(),
            "user_id": self.user_id
        }
    
    def validate_access(self, user_id: str, access_level: str = "privileged") -> bool:
        """
        Validate user access for a given level.
        
        Args:
            user_id: The user ID to validate
            access_level: "founder" or "privileged"
            
        Returns:
            bool: True if access is granted
        """
        if access_level == "founder":
            return self.is_founder(user_id)
        elif access_level == "privileged":
            return self.is_privileged(user_id)
        else:
            return False
    
    def get_boot_message(self) -> str:
        """Get the daemon's boot message with name and traits."""
        name = self.identity.get("name", "Unimind")
        traits = self.identity.get("personality_traits", [])
        
        if traits:
            traits_str = ", ".join(traits)
            return f"Hello, I am {name}, your companion daemon. I am {traits_str}."
        else:
            return f"Hello, I am {name}, your companion daemon."
    
    def describe_soul(self) -> str:
        """Return a detailed description of the current soul identity."""
        identity_info = self.get_identity_info()
        
        desc = f"🤖 **{identity_info['name']} v{identity_info['version']}**\n"
        desc += f"📝 {identity_info['description']}\n\n"
        
        if identity_info['personality_traits']:
            desc += f"🎭 **Personality Traits:**\n"
            for trait in identity_info['personality_traits']:
                desc += f"   • {trait}\n"
            desc += "\n"
        
        if identity_info['core_values']:
            desc += f"💎 **Core Values:**\n"
            for value in identity_info['core_values']:
                desc += f"   • {value}\n"
            desc += "\n"
        
        if identity_info['ethical_tenets']:
            desc += f"⚖️ **Ethical Tenets:**\n"
            for tenet in identity_info['ethical_tenets']:
                desc += f"   • {tenet}\n"
            desc += "\n"
        
        desc += f"🔐 **Access Level:** {identity_info['access_level']}\n"
        desc += f"📜 **Allowed Scrolls:** {len(identity_info['allowed_scrolls'])}\n"
        desc += f"🔒 **Founder Scrolls:** {len(identity_info['founder_only_scrolls'])}\n"
        
        if self.user_id:
            desc += f"👤 **Current User:** {self.user_id}\n"
        
        return desc 

    @property
    def name(self) -> str:
        return self.identity.get('name', 'Unimind')

    @property
    def description(self) -> str:
        return self.identity.get('description', '')

    @property
    def version(self) -> str:
        return self.identity.get('version', '')

    @property
    def access_level(self) -> str:
        return self.identity.get('access_level', 'guest')

    @property
    def personality(self) -> str:
        return ', '.join(self.identity.get('personality_traits', []))

    @property
    def greeting(self) -> str:
        return self.identity.get('communication_style', {}).get('greeting', f"Hello! I'm {self.name}. How can I help you?")

    @property
    def signature(self) -> str:
        return self.identity.get('communication_style', {}).get('signature', f"Best regards, {self.name}") 