#!/usr/bin/env python3
"""
soul_loader.py - Soul loader system for user-specific daemon identities.
Loads different daemon personalities based on user ID from soul profiles.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class SoulLoader:
    """
    Loads user-specific daemon identities from soul profiles.
    Supports different personalities for different users while maintaining core functionality.
    """
    
    def __init__(self, soul_profiles_dir: Optional[str] = None):
        """
        Initialize the soul loader.
        
        Args:
            soul_profiles_dir: Directory containing soul profile JSON files
        """
        if soul_profiles_dir is None:
            soul_profiles_dir = os.path.join(os.path.dirname(__file__), "soul_profiles")
        
        self.soul_profiles_dir = Path(soul_profiles_dir)
        self.logger = logging.getLogger('SoulLoader')
        
        # Ensure the profiles directory exists
        self.soul_profiles_dir.mkdir(exist_ok=True)
        
        # Cache for loaded profiles
        self._profile_cache: Dict[str, Dict[str, Any]] = {}
        
        # Default profile for unknown users
        self._default_profile = None
    
    def get_user_soul(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the soul configuration for a specific user.
        
        Args:
            user_id: The user ID to load soul for. If None, loads default profile.
            
        Returns:
            Dictionary containing the daemon identity configuration
        """
        if not user_id:
            return self._get_default_soul()
        
        # Check cache first
        if user_id in self._profile_cache:
            self.logger.info(f"Loading cached soul profile for user: {user_id}")
            return self._profile_cache[user_id]
        
        # Try to load user-specific profile
        profile_path = self.soul_profiles_dir / f"{user_id}.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    profile_data = json.load(f)
                    self._profile_cache[user_id] = profile_data
                    self.logger.info(f"Loaded soul profile for user: {user_id}")
                    return profile_data
            except Exception as e:
                self.logger.error(f"Error loading soul profile for {user_id}: {e}")
                return self._get_default_soul()
        else:
            self.logger.info(f"No soul profile found for user: {user_id}, using default")
            return self._get_default_soul()
    
    def _get_default_soul(self) -> Dict[str, Any]:
        """Get the default soul configuration for unknown users."""
        if self._default_profile is None:
            # Try to load guest_user.json as default
            guest_profile_path = self.soul_profiles_dir / "guest_user.json"
            
            if guest_profile_path.exists():
                try:
                    with open(guest_profile_path, 'r') as f:
                        self._default_profile = json.load(f)
                        self.logger.info("Loaded default soul profile (guest_user.json)")
                except Exception as e:
                    self.logger.error(f"Error loading default soul profile: {e}")
                    self._default_profile = self._create_fallback_soul()
            else:
                self._default_profile = self._create_fallback_soul()
        
        return self._default_profile
    
    def _create_fallback_soul(self) -> Dict[str, Any]:
        """Create a fallback soul configuration if no profiles are available."""
        return {
            "daemon_identity": {
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
        }
    
    def list_available_profiles(self) -> list:
        """List all available soul profiles."""
        profiles = []
        for profile_file in self.soul_profiles_dir.glob("*.json"):
            profiles.append(profile_file.stem)
        return profiles
    
    def create_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Create a new soul profile for a user.
        
        Args:
            user_id: The user ID
            profile_data: The soul profile data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile_path = self.soul_profiles_dir / f"{user_id}.json"
            
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            # Update cache
            self._profile_cache[user_id] = profile_data
            
            self.logger.info(f"Created soul profile for user: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating soul profile for {user_id}: {e}")
            return False
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Update an existing soul profile for a user.
        
        Args:
            user_id: The user ID
            profile_data: The updated soul profile data
            
        Returns:
            True if successful, False otherwise
        """
        return self.create_user_profile(user_id, profile_data)
    
    def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete a soul profile for a user.
        
        Args:
            user_id: The user ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile_path = self.soul_profiles_dir / f"{user_id}.json"
            
            if profile_path.exists():
                profile_path.unlink()
                
                # Remove from cache
                if user_id in self._profile_cache:
                    del self._profile_cache[user_id]
                
                self.logger.info(f"Deleted soul profile for user: {user_id}")
                return True
            else:
                self.logger.warning(f"No soul profile found for user: {user_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting soul profile for {user_id}: {e}")
            return False
    
    def get_profile_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a user's soul profile.
        
        Args:
            user_id: The user ID
            
        Returns:
            Profile information or None if not found
        """
        profile_data = self.get_user_soul(user_id)
        
        if profile_data and "daemon_identity" in profile_data:
            identity = profile_data["daemon_identity"]
            return {
                "user_id": user_id,
                "daemon_name": identity.get("name", "Unknown"),
                "version": identity.get("version", "Unknown"),
                "description": identity.get("description", ""),
                "access_level": identity.get("access_level", "guest"),
                "personality_traits": identity.get("personality_traits", []),
                "communication_style": identity.get("communication_style", {}),
                "allowed_scrolls": identity.get("allowed_scrolls", []),
                "founder_only_scrolls": identity.get("founder_only_scrolls", [])
            }
        
        return None

    def load_default_soul(self) -> Dict[str, Any]:
        """Public method to get the default soul profile (for development and fallback)."""
        return self._get_default_soul()
    
    def load_soul(self, user_id: Optional[str] = None) -> 'Soul':
        """Load the soul for a user and return a Soul object."""
        from unimind.soul.identity import Soul  # Local import to avoid circular import
        soul_data = self.get_user_soul(user_id)
        return Soul(user_id=user_id)  # Soul class will internally use get_user_soul

# Global soul loader instance
soul_loader = SoulLoader()

def get_user_soul(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get the soul configuration for a specific user using the global loader."""
    return soul_loader.get_user_soul(user_id)

def create_user_profile(user_id: str, profile_data: Dict[str, Any]) -> bool:
    """Create a new soul profile for a user using the global loader."""
    return soul_loader.create_user_profile(user_id, profile_data)

def list_available_profiles() -> list:
    """List all available soul profiles using the global loader."""
    return soul_loader.list_available_profiles() 