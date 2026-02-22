# ui/gui/theme.py
import dearpygui.dearpygui as dpg
from pathlib import Path
from typing import Optional, Tuple, Union

# =============================================================================
# COLOR CONSTANTS
# =============================================================================

class Colors:
    """Spaudible color palette (RGB tuples)."""
    
    # Background gradient points (from your PNG)
    BG_UPPER_RIGHT = (1, 24, 9)      # #011809 - Deep forest black
    BG_MIDDLE = (2, 38, 43)          # #02262b - Dark teal
    BG_BOTTOM_LEFT = (2, 13, 41)     # #020d29 - Deep ocean navy
    
    # Primary UI - Dark Green family
    PRIMARY_DARK = (27, 70, 42)      # #1b462a - Main button color
    PRIMARY_HOVER = (40, 97, 61)     # #28613d - Hover state (lighter)
    PRIMARY_ACTIVE = (20, 51, 32)    # #143320 - Active/pressed (darker)
    PRIMARY_SHADOW = (15, 35, 20)    # Shadow layer for 3D effect
    
    # Desaturated grays for inputs/menus
    BG_ELEMENT = (45, 55, 50)        # Desaturated gray-green
    BG_ELEMENT_HOVER = (55, 65, 60)  # Lighter on hover
    BG_ELEMENT_ACTIVE = (65, 75, 70) # Active state
    BG_MENU = (35, 45, 40)          # Menu bar background
    
    # Text colors
    TEXT_PRIMARY = (255, 255, 255)   # Pure white for primary text
    TEXT_SECONDARY = (200, 210, 205) # Slightly dimmed
    TEXT_MUTED = (140, 150, 145)     # Disabled/placeholder text
    
    # Accent colors (for future similarity scoring)
    ACCENT_CYAN = (0, 200, 180)      # Bright cyan-aqua
    ACCENT_GOLD = (220, 190, 100)    # Muted gold
    ACCENT_AMBER = (255, 200, 80)    # Warm amber
    
    # 3D Border effects
    BORDER_HIGHLIGHT = (60, 100, 75) # Top/left edge (lighter)
    BORDER_SHADOW = (15, 35, 25)     # Bottom/right edge (darker)
    BORDER_MID = (35, 75, 55)       # Neutral border


# =============================================================================
# BACKGROUND MANAGER
# =============================================================================

class BackgroundManager:
    """Handles the tri-gradient background image stretching."""
    
    def __init__(self):
        self.texture_tag = "bg_texture"
        self.image_tag = "bg_image"
        self._is_loaded = False
        
    def load(self) -> bool:
        """Load background PNG from data/gui/background.png"""
        try:
            bg_path = Path(__file__).parent.parent.parent / "data" / "gui" / "background.png"
            
            if not bg_path.exists():
                print(f"[Theme] Background not found at {bg_path}, using solid color")
                return False
                
            width, height, channels, data = dpg.load_image(str(bg_path))
            
            with dpg.texture_registry():
                dpg.add_static_texture(
                    width, height, data, 
                    tag=self.texture_tag,
                    label="Background"
                )
            
            self._is_loaded = True
            print(f"[Theme] Loaded background: {width}x{height}")
            return True
            
        except Exception as e:
            print(f"[Theme] Error loading background: {e}")
            return False
    
    def create_background(self, parent: str):
        """Create the background image element filling the window."""
        if not self._is_loaded:
            return
            
        vp_width = dpg.get_viewport_width()
        vp_height = dpg.get_viewport_height()
        
        dpg.add_image(
            self.texture_tag,
            tag=self.image_tag,
            parent=parent,
            pos=(0, 0),
            width=vp_width,
            height=vp_height
        )
    
    def update_size(self):
        """Update background size to match viewport (call on resize)."""
        if dpg.does_item_exist(self.image_tag):
            vp_width = dpg.get_viewport_width()
            vp_height = dpg.get_viewport_height()
            dpg.configure_item(
                self.image_tag,
                width=vp_width,
                height=vp_height
            )


# =============================================================================
# MAIN THEME CLASS
# =============================================================================

class SpaudibleTheme:
    """Orchestrates all theme initialization and widget styling."""
    
    def __init__(self):
        self.bg_manager = BackgroundManager()
        self.theme_tag = "spaudible_main_theme"
        self.shadow_theme_tag = "spaudible_shadow_theme"
        
    def initialize(self):
        """Load assets and apply theme."""
        self.bg_manager.load()
        self._create_main_theme()
        self._create_shadow_theme()
        dpg.bind_theme(self.theme_tag)
        
    def _create_main_theme(self):
        """Create the comprehensive DPG theme."""
        with dpg.theme(tag=self.theme_tag) as theme:
            with dpg.theme_component(dpg.mvAll):
                
                # --- Window & Container Backgrounds ---
                # Use middle color as fallback
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, Colors.BG_MIDDLE)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (2, 30, 35, 0))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (*Colors.BG_BOTTOM_LEFT, 250))
                dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, (0, 0, 0, 100))
                dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg, Colors.BG_MIDDLE)
                
                # --- Text Styling ---
                dpg.add_theme_color(dpg.mvThemeCol_Text, Colors.TEXT_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, Colors.TEXT_MUTED)
                dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, (*Colors.PRIMARY_DARK, 180))
                
                # --- Buttons (Semi-3D Dark Green) ---
                dpg.add_theme_color(dpg.mvThemeCol_Button, Colors.PRIMARY_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, Colors.PRIMARY_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, Colors.PRIMARY_ACTIVE)
                # Border creates the 3D edge effect
                dpg.add_theme_color(dpg.mvThemeCol_Border, Colors.BORDER_MID)
                dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, Colors.BORDER_SHADOW)
                
                # --- Input Fields & Frames ---
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, Colors.BG_ELEMENT)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, Colors.BG_ELEMENT_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, Colors.BG_ELEMENT_ACTIVE)
                
                # --- Menu & Selection Headers ---
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, Colors.BG_MENU)
                dpg.add_theme_color(dpg.mvThemeCol_Header, Colors.PRIMARY_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, Colors.PRIMARY_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, Colors.PRIMARY_ACTIVE)
                
                # --- Tabs (for collapsible settings) ---
                dpg.add_theme_color(dpg.mvThemeCol_Tab, Colors.BG_ELEMENT)
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, Colors.PRIMARY_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, Colors.PRIMARY_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (30, 40, 35))
                dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (25, 55, 40))
                
                # --- Sliders & Scrollbars ---
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, Colors.PRIMARY_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, Colors.ACCENT_CYAN)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (20, 30, 28))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, Colors.PRIMARY_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, Colors.PRIMARY_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, Colors.ACCENT_CYAN)
                
                # --- Checkboxes & Radio Buttons ---
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, Colors.ACCENT_CYAN)
                
                # --- Progress Bar & Plot ---
                dpg.add_theme_color(dpg.mvThemeCol_PlotLines, Colors.ACCENT_CYAN)
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, Colors.ACCENT_CYAN)
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, Colors.ACCENT_GOLD)
                
                # --- Separators ---
                dpg.add_theme_color(dpg.mvThemeCol_Separator, (40, 70, 60))
                dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, (60, 100, 85))
                dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, Colors.ACCENT_CYAN)
                
                # --- Resizing & Drag ---
                dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, (40, 60, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, Colors.PRIMARY_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, Colors.ACCENT_CYAN)
                
                # --- Rounding ---
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 6)
                
                # --- Borders (The 3D effect) ---
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 3)  # Border for depth
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)  # Clean window edges
                dpg.add_theme_style(dpg.mvStyleVar_PopupBorderSize, 1)
                
                # --- Spacing & Padding ---
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5)  # Centered
                
    def _create_shadow_theme(self):
        """Theme for shadow layers behind buttons (3D effect)."""
        with dpg.theme(tag=self.shadow_theme_tag) as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, Colors.PRIMARY_SHADOW)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, Colors.PRIMARY_SHADOW)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, Colors.PRIMARY_SHADOW)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
    
    def create_background(self, parent: str):
        """Place background in window."""
        self.bg_manager.create_background(parent)
        
    def update_background(self):
        """Call this in your render loop or resize handler."""
        self.bg_manager.update_size()


# =============================================================================
# WIDGET UTILITIES
# =============================================================================

def add_3d_button(label: str, callback=None, parent=None, width: int = 150, height: int = 34, **kwargs) -> int:
    """
    Create a button with semi-3D depth effect using offset shadow layer.
    Returns the main button ID (the clickable one).
    """
    with dpg.group(horizontal=False, parent=parent):
        # Shadow layer (offset 2px down/right, darker)
        shadow = dpg.add_button(
            label="",
            width=width,
            height=height,
            pos=(2, 2),
            enabled=False
        )
        dpg.bind_item_theme(shadow, "spaudible_shadow_theme")
        
        # Main button (offset to top-left, overlapping shadow)
        btn = dpg.add_button(
            label=label,
            width=width,
            height=height,
            callback=callback,
            **kwargs
        )
        # Theme is automatically applied via global theme
    
    return btn

def add_styled_slider(label: str, default_value: float = 1.0, 
                     min_value: float = 0.0, max_value: float = 10.0,
                     parent=None, **kwargs) -> int:
    """Create a slider with Spaudible styling."""
    slider = dpg.add_slider_float(
        label=label,
        default_value=default_value,
        min_value=min_value,
        max_value=max_value,
        parent=parent,
        width=200,
        **kwargs
    )
    return slider

def add_collapsible_section(label: str, parent=None, default_open: bool = True):
    """
    Helper to create a tree node (collapsible section) with proper styling.
    Returns the tree node tag.
    """
    return dpg.add_tree_node(
        label=label,
        parent=parent,
        default_open=default_open,
        bullet=False,
        span_full_width=True
    )


# =============================================================================
# INITIALIZATION
# =============================================================================

_theme_instance: Optional[SpaudibleTheme] = None

def initialize_theme() -> SpaudibleTheme:
    """Initialize and return the global theme instance."""
    global _theme_instance
    _theme_instance = SpaudibleTheme()
    _theme_instance.initialize()
    return _theme_instance

def get_theme() -> Optional[SpaudibleTheme]:
    """Get the current theme instance (initialize first)."""
    return _theme_instance
