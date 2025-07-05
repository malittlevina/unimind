"""
realm_gui.py – GUI interface for 3D realm-building system.
Provides a graphical interface for creating and managing 3D realms.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import threading
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from unimind.bridge.storyrealms_bridge import (
        storyrealms_bridge, RealmArchetype, ObjectType, 
        GlyphType, Coordinates
    )
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    print("Warning: storyrealms_bridge not available")

class RealmGUI:
    """GUI interface for 3D realm-building system."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Unimind 3D Realm Builder")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize bridge
        self.bridge = storyrealms_bridge if BRIDGE_AVAILABLE else None
        self.current_realm_id = None
        self.realms_list = []
        
        # Create GUI elements
        self.create_widgets()
        self.setup_styles()
        self.refresh_realms()
        
        # Start request monitoring
        self.start_request_monitor()
    
    def setup_styles(self):
        """Setup custom styles for the GUI."""
        style = ttk.Style()
        
        # Configure dark theme
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Arial', 16, 'bold'))
        
        style.configure('Header.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Arial', 12, 'bold'))
        
        style.configure('Info.TLabel', 
                       background='#2b2b2b', 
                       foreground='#cccccc', 
                       font=('Arial', 10))
        
        style.configure('Success.TLabel', 
                       background='#2b2b2b', 
                       foreground='#4CAF50', 
                       font=('Arial', 10))
        
        style.configure('Error.TLabel', 
                       background='#2b2b2b', 
                       foreground='#f44336', 
                       font=('Arial', 10))
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🌍 Unimind 3D Realm Builder", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_realm_tab()
        self.create_objects_tab()
        self.create_glyphs_tab()
        self.create_management_tab()
        self.create_requests_tab()
    
    def create_realm_tab(self):
        """Create the realm creation tab."""
        realm_frame = ttk.Frame(self.notebook)
        self.notebook.add(realm_frame, text="🏗️ Create Realms")
        
        # Left panel - Realm creation
        left_frame = ttk.Frame(realm_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Realm creation section
        create_frame = ttk.LabelFrame(left_frame, text="Create New Realm", padding=10)
        create_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Name input
        ttk.Label(create_frame, text="Realm Name:", style='Header.TLabel').pack(anchor=tk.W)
        self.realm_name_var = tk.StringVar(value="My Realm")
        name_entry = ttk.Entry(create_frame, textvariable=self.realm_name_var, width=40)
        name_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Archetype selection
        ttk.Label(create_frame, text="Archetype:", style='Header.TLabel').pack(anchor=tk.W)
        self.archetype_var = tk.StringVar(value="forest_glade")
        archetype_combo = ttk.Combobox(create_frame, textvariable=self.archetype_var, 
                                      values=[a.value for a in RealmArchetype], state="readonly")
        archetype_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Description input
        ttk.Label(create_frame, text="Description:", style='Header.TLabel').pack(anchor=tk.W)
        self.realm_desc_text = scrolledtext.ScrolledText(create_frame, height=4, width=40)
        self.realm_desc_text.pack(fill=tk.X, pady=(0, 10))
        
        # Properties frame
        props_frame = ttk.LabelFrame(create_frame, text="Properties", padding=10)
        props_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Weather
        ttk.Label(props_frame, text="Weather:", style='Info.TLabel').pack(anchor=tk.W)
        self.weather_var = tk.StringVar(value="clear")
        weather_combo = ttk.Combobox(props_frame, textvariable=self.weather_var,
                                   values=["clear", "stormy", "mystical", "calm"], state="readonly")
        weather_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Time of day
        ttk.Label(props_frame, text="Time of Day:", style='Info.TLabel').pack(anchor=tk.W)
        self.time_var = tk.StringVar(value="noon")
        time_combo = ttk.Combobox(props_frame, textvariable=self.time_var,
                                values=["dawn", "noon", "dusk", "night"], state="readonly")
        time_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Atmosphere
        ttk.Label(props_frame, text="Atmosphere:", style='Info.TLabel').pack(anchor=tk.W)
        self.atmosphere_var = tk.StringVar(value="peaceful")
        atmosphere_combo = ttk.Combobox(props_frame, textvariable=self.atmosphere_var,
                                      values=["peaceful", "tense", "magical", "eerie"], state="readonly")
        atmosphere_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Create button
        create_btn = ttk.Button(create_frame, text="Create Realm", command=self.create_realm)
        create_btn.pack(pady=10)
        
        # Right panel - Realm list
        right_frame = ttk.Frame(realm_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Realms list
        realms_frame = ttk.LabelFrame(right_frame, text="Available Realms", padding=10)
        realms_frame.pack(fill=tk.BOTH, expand=True)
        
        # Realms listbox
        self.realms_listbox = tk.Listbox(realms_frame, bg='#3c3c3c', fg='#ffffff', 
                                        selectbackground='#4CAF50', height=15)
        self.realms_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.realms_listbox.bind('<<ListboxSelect>>', self.on_realm_select)
        
        # Realm info
        self.realm_info_text = scrolledtext.ScrolledText(realms_frame, height=8, width=40,
                                                        bg='#3c3c3c', fg='#ffffff')
        self.realm_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame
        btn_frame = ttk.Frame(realms_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Refresh", command=self.refresh_realms).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Load", command=self.load_selected_realm).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Delete", command=self.delete_selected_realm).pack(side=tk.LEFT)
    
    def create_objects_tab(self):
        """Create the objects tab."""
        objects_frame = ttk.Frame(self.notebook)
        self.notebook.add(objects_frame, text="🌳 Place Objects")
        
        # Left panel - Object placement
        left_frame = ttk.Frame(objects_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Object placement section
        place_frame = ttk.LabelFrame(left_frame, text="Place Object", padding=10)
        place_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Object type selection
        ttk.Label(place_frame, text="Object Type:", style='Header.TLabel').pack(anchor=tk.W)
        self.object_type_var = tk.StringVar(value="tree")
        object_combo = ttk.Combobox(place_frame, textvariable=self.object_type_var,
                                   values=[o.value for o in ObjectType], state="readonly")
        object_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Coordinates frame
        coords_frame = ttk.LabelFrame(place_frame, text="Coordinates", padding=10)
        coords_frame.pack(fill=tk.X, pady=(0, 10))
        
        # X coordinate
        ttk.Label(coords_frame, text="X:", style='Info.TLabel').pack(anchor=tk.W)
        self.x_var = tk.DoubleVar(value=0.0)
        x_entry = ttk.Entry(coords_frame, textvariable=self.x_var)
        x_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Y coordinate
        ttk.Label(coords_frame, text="Y:", style='Info.TLabel').pack(anchor=tk.W)
        self.y_var = tk.DoubleVar(value=0.0)
        y_entry = ttk.Entry(coords_frame, textvariable=self.y_var)
        y_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Z coordinate
        ttk.Label(coords_frame, text="Z:", style='Info.TLabel').pack(anchor=tk.W)
        self.z_var = tk.DoubleVar(value=0.0)
        z_entry = ttk.Entry(coords_frame, textvariable=self.z_var)
        z_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Properties frame
        obj_props_frame = ttk.LabelFrame(place_frame, text="Object Properties", padding=10)
        obj_props_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Scale
        ttk.Label(obj_props_frame, text="Scale:", style='Info.TLabel').pack(anchor=tk.W)
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_entry = ttk.Entry(obj_props_frame, textvariable=self.scale_var)
        scale_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Glow color
        ttk.Label(obj_props_frame, text="Glow Color:", style='Info.TLabel').pack(anchor=tk.W)
        self.glow_var = tk.StringVar(value="none")
        glow_combo = ttk.Combobox(obj_props_frame, textvariable=self.glow_var,
                                 values=["none", "blue", "green", "red", "yellow", "purple"], state="readonly")
        glow_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Place button
        place_btn = ttk.Button(place_frame, text="Place Object", command=self.place_object)
        place_btn.pack(pady=10)
        
        # Right panel - Objects list
        right_frame = ttk.Frame(objects_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Objects list
        objects_list_frame = ttk.LabelFrame(right_frame, text="Objects in Current Realm", padding=10)
        objects_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Objects listbox
        self.objects_listbox = tk.Listbox(objects_list_frame, bg='#3c3c3c', fg='#ffffff',
                                         selectbackground='#4CAF50', height=15)
        self.objects_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Object info
        self.object_info_text = scrolledtext.ScrolledText(objects_list_frame, height=8, width=40,
                                                         bg='#3c3c3c', fg='#ffffff')
        self.object_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Remove button
        ttk.Button(objects_list_frame, text="Remove Object", command=self.remove_object).pack(pady=(10, 0))
    
    def create_glyphs_tab(self):
        """Create the glyphs tab."""
        glyphs_frame = ttk.Frame(self.notebook)
        self.notebook.add(glyphs_frame, text="✨ Cast Glyphs")
        
        # Left panel - Glyph casting
        left_frame = ttk.Frame(glyphs_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Glyph casting section
        cast_frame = ttk.LabelFrame(left_frame, text="Cast Glyph", padding=10)
        cast_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Glyph type selection
        ttk.Label(cast_frame, text="Glyph Type:", style='Header.TLabel').pack(anchor=tk.W)
        self.glyph_type_var = tk.StringVar(value="illumination")
        glyph_combo = ttk.Combobox(cast_frame, textvariable=self.glyph_type_var,
                                  values=[g.value for g in GlyphType], state="readonly")
        glyph_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Location frame
        loc_frame = ttk.LabelFrame(cast_frame, text="Location", padding=10)
        loc_frame.pack(fill=tk.X, pady=(0, 10))
        
        # X coordinate
        ttk.Label(loc_frame, text="X:", style='Info.TLabel').pack(anchor=tk.W)
        self.glyph_x_var = tk.DoubleVar(value=0.0)
        glyph_x_entry = ttk.Entry(loc_frame, textvariable=self.glyph_x_var)
        glyph_x_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Y coordinate
        ttk.Label(loc_frame, text="Y:", style='Info.TLabel').pack(anchor=tk.W)
        self.glyph_y_var = tk.DoubleVar(value=0.0)
        glyph_y_entry = ttk.Entry(loc_frame, textvariable=self.glyph_y_var)
        glyph_y_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Z coordinate
        ttk.Label(loc_frame, text="Z:", style='Info.TLabel').pack(anchor=tk.W)
        self.glyph_z_var = tk.DoubleVar(value=0.0)
        glyph_z_entry = ttk.Entry(loc_frame, textvariable=self.glyph_z_var)
        glyph_z_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Glyph properties frame
        glyph_props_frame = ttk.LabelFrame(cast_frame, text="Glyph Properties", padding=10)
        glyph_props_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Intensity
        ttk.Label(glyph_props_frame, text="Intensity:", style='Info.TLabel').pack(anchor=tk.W)
        self.intensity_var = tk.StringVar(value="moderate")
        intensity_combo = ttk.Combobox(glyph_props_frame, textvariable=self.intensity_var,
                                      values=["weak", "moderate", "strong", "extreme"], state="readonly")
        intensity_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Duration
        ttk.Label(glyph_props_frame, text="Duration (seconds):", style='Info.TLabel').pack(anchor=tk.W)
        self.duration_var = tk.DoubleVar(value=3600.0)
        duration_entry = ttk.Entry(glyph_props_frame, textvariable=self.duration_var)
        duration_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Cast button
        cast_btn = ttk.Button(cast_frame, text="Cast Glyph", command=self.cast_glyph)
        cast_btn.pack(pady=10)
        
        # Right panel - Glyphs list
        right_frame = ttk.Frame(glyphs_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Glyphs list
        glyphs_list_frame = ttk.LabelFrame(right_frame, text="Active Glyphs", padding=10)
        glyphs_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Glyphs listbox
        self.glyphs_listbox = tk.Listbox(glyphs_list_frame, bg='#3c3c3c', fg='#ffffff',
                                        selectbackground='#4CAF50', height=15)
        self.glyphs_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Glyph info
        self.glyph_info_text = scrolledtext.ScrolledText(glyphs_list_frame, height=8, width=40,
                                                        bg='#3c3c3c', fg='#ffffff')
        self.glyph_info_text.pack(fill=tk.BOTH, expand=True)
    
    def create_management_tab(self):
        """Create the management tab."""
        mgmt_frame = ttk.Frame(self.notebook)
        self.notebook.add(mgmt_frame, text="⚙️ Management")
        
        # Left panel - Realm management
        left_frame = ttk.Frame(mgmt_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Realm management section
        mgmt_section = ttk.LabelFrame(left_frame, text="Realm Management", padding=10)
        mgmt_section.pack(fill=tk.X, pady=(0, 10))
        
        # Current realm info
        ttk.Label(mgmt_section, text="Current Realm:", style='Header.TLabel').pack(anchor=tk.W)
        self.current_realm_label = ttk.Label(mgmt_section, text="None selected", style='Info.TLabel')
        self.current_realm_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Buttons
        btn_frame = ttk.Frame(mgmt_section)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame, text="Save Realm State", command=self.save_realm_state).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Load Realm State", command=self.load_realm_state).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT)
        
        # Properties modification
        props_section = ttk.LabelFrame(left_frame, text="Modify Realm Properties", padding=10)
        props_section.pack(fill=tk.X, pady=(0, 10))
        
        # Weather modification
        ttk.Label(props_section, text="Weather:", style='Info.TLabel').pack(anchor=tk.W)
        self.mod_weather_var = tk.StringVar(value="clear")
        mod_weather_combo = ttk.Combobox(props_section, textvariable=self.mod_weather_var,
                                        values=["clear", "stormy", "mystical", "calm"], state="readonly")
        mod_weather_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Time modification
        ttk.Label(props_section, text="Time of Day:", style='Info.TLabel').pack(anchor=tk.W)
        self.mod_time_var = tk.StringVar(value="noon")
        mod_time_combo = ttk.Combobox(props_section, textvariable=self.mod_time_var,
                                     values=["dawn", "noon", "dusk", "night"], state="readonly")
        mod_time_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Apply changes button
        ttk.Button(props_section, text="Apply Changes", command=self.apply_realm_changes).pack(pady=(10, 0))
        
        # Right panel - System info
        right_frame = ttk.Frame(mgmt_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # System info
        info_section = ttk.LabelFrame(right_frame, text="System Information", padding=10)
        info_section.pack(fill=tk.BOTH, expand=True)
        
        self.system_info_text = scrolledtext.ScrolledText(info_section, height=20, width=40,
                                                         bg='#3c3c3c', fg='#ffffff')
        self.system_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        ttk.Button(info_section, text="Refresh Info", command=self.refresh_system_info).pack(pady=(10, 0))
    
    def create_requests_tab(self):
        """Create the requests tab."""
        requests_frame = ttk.Frame(self.notebook)
        self.notebook.add(requests_frame, text="📤 Engine Requests")
        
        # Requests display
        requests_display_frame = ttk.LabelFrame(requests_frame, text="Generated Requests", padding=10)
        requests_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Requests text area
        self.requests_text = scrolledtext.ScrolledText(requests_display_frame, height=25, width=80,
                                                      bg='#3c3c3c', fg='#ffffff')
        self.requests_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Buttons frame
        btn_frame = ttk.Frame(requests_display_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Refresh Requests", command=self.refresh_requests).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear Requests", command=self.clear_requests).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Export Requests", command=self.export_requests).pack(side=tk.LEFT)
    
    def create_realm(self):
        """Create a new realm."""
        if not self.bridge:
            messagebox.showerror("Error", "Bridge not available")
            return
        
        try:
            name = self.realm_name_var.get()
            archetype = self.archetype_var.get()
            description = self.realm_desc_text.get("1.0", tk.END).strip()
            
            properties = {
                "weather": self.weather_var.get(),
                "time_of_day": self.time_var.get(),
                "atmosphere": self.atmosphere_var.get()
            }
            
            realm_id = self.bridge.create_realm(name, archetype, description, properties)
            
            messagebox.showinfo("Success", f"Realm '{name}' created successfully!\nID: {realm_id[:8]}...")
            self.refresh_realms()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create realm: {str(e)}")
    
    def place_object(self):
        """Place an object in the current realm."""
        if not self.bridge or not self.current_realm_id:
            messagebox.showerror("Error", "No realm selected")
            return
        
        try:
            object_type = self.object_type_var.get()
            coordinates = Coordinates(
                x=self.x_var.get(),
                y=self.y_var.get(),
                z=self.z_var.get(),
                scale_x=self.scale_var.get(),
                scale_y=self.scale_var.get(),
                scale_z=self.scale_var.get()
            )
            
            properties = {}
            if self.glow_var.get() != "none":
                properties["glow"] = self.glow_var.get()
            
            object_id = self.bridge.place_object(self.current_realm_id, object_type, coordinates, properties)
            
            messagebox.showinfo("Success", f"Object placed successfully!\nID: {object_id[:8]}...")
            self.refresh_objects()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to place object: {str(e)}")
    
    def cast_glyph(self):
        """Cast a glyph in the current realm."""
        if not self.bridge or not self.current_realm_id:
            messagebox.showerror("Error", "No realm selected")
            return
        
        try:
            glyph_type = self.glyph_type_var.get()
            location = Coordinates(
                x=self.glyph_x_var.get(),
                y=self.glyph_y_var.get(),
                z=self.glyph_z_var.get()
            )
            
            properties = {
                "intensity": self.intensity_var.get(),
                "duration": self.duration_var.get()
            }
            
            glyph_id = self.bridge.cast_glyph_in_realm(
                self.current_realm_id, glyph_type, location, "gui_user", 
                duration=self.duration_var.get(), properties=properties
            )
            
            messagebox.showinfo("Success", f"Glyph cast successfully!\nID: {glyph_id[:8]}...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to cast glyph: {str(e)}")
    
    def refresh_realms(self):
        """Refresh the realms list."""
        if not self.bridge:
            return
        
        try:
            self.realms_list = self.bridge.list_realms()
            self.realms_listbox.delete(0, tk.END)
            
            for realm in self.realms_list:
                if realm:
                    display_text = f"{realm['name']} ({realm['archetype']})"
                    self.realms_listbox.insert(tk.END, display_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh realms: {str(e)}")
    
    def on_realm_select(self, event):
        """Handle realm selection."""
        selection = self.realms_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.realms_list) and self.realms_list[index]:
                realm = self.realms_list[index]
                self.current_realm_id = realm['realm_id']
                self.current_realm_label.config(text=f"{realm['name']} ({realm['archetype']})")
                self.display_realm_info(realm)
                self.refresh_objects()
    
    def display_realm_info(self, realm):
        """Display realm information."""
        info = f"""Realm Information:
Name: {realm['name']}
Archetype: {realm['archetype']}
Description: {realm.get('description', 'No description')}
Objects: {realm['object_count']}
Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(realm['created_at']))}
Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(realm['modified_at']))}

Properties:
"""
        for key, value in realm['properties'].items():
            info += f"  {key}: {value}\n"
        
        self.realm_info_text.delete("1.0", tk.END)
        self.realm_info_text.insert("1.0", info)
    
    def refresh_objects(self):
        """Refresh the objects list."""
        if not self.bridge or not self.current_realm_id:
            return
        
        try:
            realm_info = self.bridge.get_realm_info(self.current_realm_id)
            if realm_info:
                self.objects_listbox.delete(0, tk.END)
                # Note: This would need to be enhanced to show actual objects
                # For now, just show the count
                self.objects_listbox.insert(tk.END, f"Objects: {realm_info['object_count']}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh objects: {str(e)}")
    
    def load_selected_realm(self):
        """Load the selected realm."""
        if not self.bridge or not self.current_realm_id:
            messagebox.showerror("Error", "No realm selected")
            return
        
        try:
            success = self.bridge.load_realm(self.current_realm_id)
            if success:
                messagebox.showinfo("Success", "Realm loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load realm")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load realm: {str(e)}")
    
    def delete_selected_realm(self):
        """Delete the selected realm."""
        if not self.bridge or not self.current_realm_id:
            messagebox.showerror("Error", "No realm selected")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this realm?"):
            try:
                # Note: This would need to be implemented in the bridge
                messagebox.showinfo("Info", "Realm deletion not yet implemented")
                self.refresh_realms()
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete realm: {str(e)}")
    
    def remove_object(self):
        """Remove the selected object."""
        if not self.bridge or not self.current_realm_id:
            messagebox.showerror("Error", "No realm selected")
            return
        
        selection = self.objects_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No object selected")
            return
        
        # Note: This would need object IDs to be implemented
        messagebox.showinfo("Info", "Object removal not yet implemented")
    
    def save_realm_state(self):
        """Save the current realm state."""
        if not self.bridge or not self.current_realm_id:
            messagebox.showerror("Error", "No realm selected")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                success = self.bridge.save_realm_state(self.current_realm_id, filepath)
                if success:
                    messagebox.showinfo("Success", f"Realm state saved to {filepath}")
                else:
                    messagebox.showerror("Error", "Failed to save realm state")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save realm state: {str(e)}")
    
    def load_realm_state(self):
        """Load a realm state from file."""
        if not self.bridge:
            messagebox.showerror("Error", "Bridge not available")
            return
        
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                success = self.bridge.load_realm_state(filepath)
                if success:
                    messagebox.showinfo("Success", f"Realm state loaded from {filepath}")
                    self.refresh_realms()
                else:
                    messagebox.showerror("Error", "Failed to load realm state")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load realm state: {str(e)}")
    
    def clear_all(self):
        """Clear all realms and objects."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all realms?"):
            try:
                self.bridge.clear_request_queue()
                self.refresh_realms()
                messagebox.showinfo("Success", "All realms cleared")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear realms: {str(e)}")
    
    def apply_realm_changes(self):
        """Apply changes to the current realm."""
        if not self.bridge or not self.current_realm_id:
            messagebox.showerror("Error", "No realm selected")
            return
        
        try:
            properties = {
                "weather": self.mod_weather_var.get(),
                "time_of_day": self.mod_time_var.get()
            }
            
            success = self.bridge.modify_realm(self.current_realm_id, properties)
            if success:
                messagebox.showinfo("Success", "Realm properties updated")
                self.refresh_realms()
            else:
                messagebox.showerror("Error", "Failed to update realm properties")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update realm: {str(e)}")
    
    def refresh_system_info(self):
        """Refresh system information."""
        if not self.bridge:
            return
        
        try:
            info = f"""System Information:
Bridge Status: {'Connected' if self.bridge else 'Disconnected'}
Engine Type: {self.bridge.engine_type if self.bridge else 'Unknown'}
Active Realm: {self.current_realm_id[:8] + '...' if self.current_realm_id else 'None'}
Total Realms: {len(self.bridge.realms) if self.bridge else 0}
Pending Requests: {len(self.bridge.get_pending_requests()) if self.bridge else 0}

Configuration:
"""
            if self.bridge:
                for key, value in self.bridge.config.items():
                    if isinstance(value, dict):
                        info += f"  {key}:\n"
                        for k, v in value.items():
                            info += f"    {k}: {v}\n"
                    else:
                        info += f"  {key}: {value}\n"
            
            self.system_info_text.delete("1.0", tk.END)
            self.system_info_text.insert("1.0", info)
        
        except Exception as e:
            self.system_info_text.delete("1.0", tk.END)
            self.system_info_text.insert("1.0", f"Error loading system info: {str(e)}")
    
    def refresh_requests(self):
        """Refresh the requests display."""
        if not self.bridge:
            return
        
        try:
            requests = self.bridge.get_pending_requests()
            
            if requests:
                display_text = "Generated Requests:\n" + "=" * 50 + "\n\n"
                for i, request in enumerate(requests, 1):
                    display_text += f"Request {i}:\n"
                    display_text += json.dumps(request, indent=2)
                    display_text += "\n" + "-" * 30 + "\n\n"
            else:
                display_text = "No pending requests"
            
            self.requests_text.delete("1.0", tk.END)
            self.requests_text.insert("1.0", display_text)
        
        except Exception as e:
            self.requests_text.delete("1.0", tk.END)
            self.requests_text.insert("1.0", f"Error loading requests: {str(e)}")
    
    def clear_requests(self):
        """Clear all pending requests."""
        if not self.bridge:
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all requests?"):
            try:
                self.bridge.clear_request_queue()
                self.refresh_requests()
                messagebox.showinfo("Success", "All requests cleared")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear requests: {str(e)}")
    
    def export_requests(self):
        """Export requests to a file."""
        if not self.bridge:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                requests = self.bridge.get_pending_requests()
                with open(filepath, 'w') as f:
                    json.dump(requests, f, indent=2)
                messagebox.showinfo("Success", f"Requests exported to {filepath}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export requests: {str(e)}")
    
    def start_request_monitor(self):
        """Start monitoring for new requests."""
        def monitor():
            while True:
                try:
                    self.refresh_requests()
                    time.sleep(5)  # Refresh every 5 seconds
                except:
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = RealmGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 