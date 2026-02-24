# ui/gui/theme.py
import numpy as np
import dearpygui.dearpygui as dpg
from pathlib import Path
from typing import Optional, Tuple, Union, List

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
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, Colors.BG_MIDDLE)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (2, 30, 35, 0))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (*Colors.BG_BOTTOM_LEFT, 250))
                dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, (0, 0, 0, 100))
                dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg, Colors.BG_MIDDLE)
                
                # --- Text Styling ---
                dpg.add_theme_color(dpg.mvThemeCol_Text, Colors.TEXT_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, Colors.TEXT_MUTED)
                dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, (*Colors.PRIMARY_DARK, 180))
                
                # --- Buttons ---
                dpg.add_theme_color(dpg.mvThemeCol_Button, Colors.PRIMARY_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, Colors.PRIMARY_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, Colors.PRIMARY_ACTIVE)
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
                
                # --- Tabs ---
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
                
                # --- Checkboxes ---
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
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 12)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 6)
                
                # --- Borders ---
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 3)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_PopupBorderSize, 1)
                
                # --- Spacing & Padding ---
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5)
                
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
# CUSTOM BUTTONS
# =============================================================================

class GradientButtonFactory:
    """Creates photorealistic embossed buttons with S-curve gradients and proper 3D shading."""
    
    def __init__(self):
        self._texture_cache = {}
        self._font_path = Path(__file__).parent.parent.parent / "data" / "fonts" / "OpenSans-Regular.ttf"

    def _generate_embossed_gradient(self, width: int, height: int, top_color: Tuple[int, int, int], 
                                    bottom_color: Tuple[int, int, int], label: str = "", 
                                    corner_radius: int = 8) -> List[float]:
        """Generate vertical gradient with plastic emboss luminance distribution."""
        width = int(width)
        height = int(height)
        corner_radius = int(corner_radius)
        
        # Create coordinate grid
        y_normalized = np.linspace(1.0, 0.0, height)[:, np.newaxis]
        
        # Piecewise plastic emboss curve based on your luminance specification:
        # Top 20%: Aggressive drop from highlights to mid-tones
        # Middle 60%: Gentle graduation in the mid-tones
        # Bottom 20%: Aggressive drop from mid-tones to shadows
        s_curve = np.zeros_like(y_normalized)
        
        for i, y in enumerate(y_normalized[:, 0]):
            t = 1.0 - y  # Convert to 0=top, 1=bottom for easier logic
            
            if t < 0.2:
                # Top region: Quadratic ease-in (steep start)
                # Maps t=[0,0.2] to factor=[1.0,0.32]
                u = t / 0.2
                factor = 1.0 - 0.68 * (u ** 2)
            elif t < 0.8:
                # Middle region: Linear (gentle slope)
                # Maps t=[0.2,0.8] to factor=[0.32,0.12]
                u = (t - 0.2) / 0.6
                factor = 0.32 - 0.20 * u
            else:
                # Bottom region: Quadratic ease-out (steep end)
                # Maps t=[0.8,1.0] to factor=[0.12,0.0]
                u = (t - 0.8) / 0.2
                factor = 0.12 * ((1.0 - u) ** 2)
                
            s_curve[i, 0] = factor
        
        # Initialize RGBA
        gradient = np.zeros((height, width, 4), dtype=np.float32)
        
        # Interpolate RGB with the plastic curve
        for i in range(3):
            top_val = top_color[i] / 255.0
            bottom_val = bottom_color[i] / 255.0
            channel = bottom_val + (top_val - bottom_val) * s_curve
            gradient[:, :, i] = np.tile(channel, (1, width))
        
        gradient[:, :, 3] = 1.0
        
        # Apply rounded corners via alpha mask
        if corner_radius > 0:
            try:
                from PIL import Image, ImageDraw
                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle((0, 0, width-1, height-1), radius=corner_radius, fill=255)
                mask_arr = np.array(mask).astype(np.float32) / 255.0
                gradient[:, :, 3] *= mask_arr
            except ImportError:
                pass
        
        # Render text if PIL available
        if label and self._font_path.exists():
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.fromarray((gradient * 255).astype(np.uint8), 'RGBA')
                draw = ImageDraw.Draw(img)
                
                font_size = max(12, int(height * 0.55))
                try:
                    font = ImageFont.truetype(str(self._font_path), font_size)
                except:
                    font = ImageFont.load_default()
                    
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                x = (width - text_w) // 2
                y = (height - text_h) // 2 - 1
                
                draw.text((x+1, y+1), label, font=font, fill=(0, 0, 0, 160))
                draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
                
                gradient = np.array(img).astype(np.float32) / 255.0
            except Exception:
                pass
        
        return gradient.flatten().tolist()

    def _get_or_create_texture(self, tag: str, width: int, height: int, 
                              top_color: Tuple[int, int, int], 
                              bottom_color: Tuple[int, int, int], 
                              label: str = "", corner_radius: int = 8) -> str:
        """Cache textures to avoid regeneration."""
        width = int(width)
        height = int(height)
        corner_radius = int(corner_radius)
        
        cache_key = f"{tag}_{width}_{height}_{label}_{corner_radius}"
        if cache_key in self._texture_cache:
            return self._texture_cache[cache_key]
        
        data = self._generate_embossed_gradient(width, height, top_color, bottom_color, label, corner_radius)
        
        with dpg.texture_registry():
            dpg.add_static_texture(width, height, data, tag=tag)
        
        self._texture_cache[cache_key] = tag
        return tag

    def create_button(self, label: str, callback=None, parent=None, width: int = 150, 
                     height: int = 40, tag: str = None, corner_radius: int = 8) -> int:
        """
        Create an embossed 3D button.
        """
        width = int(width)
        height = int(height)
        corner_radius = int(corner_radius)
        
        # Color definitions for embossed effect
        base = Colors.PRIMARY_DARK
        
        normal_top = tuple(min(255, int(c * 1.4)) for c in base)
        normal_bottom = tuple(max(0, int(c * 0.6)) for c in base)
        hover_top = tuple(min(255, int(c * 1.6)) for c in base)
        hover_bottom = tuple(max(0, int(c * 0.8)) for c in base)
        active_top = tuple(max(0, int(c * 0.7)) for c in base)
        active_bottom = tuple(min(255, int(c * 1.2)) for c in base)
        
        # Generate unique texture tags
        import hashlib
        hash_base = hashlib.md5(f"{label}_{width}_{height}".encode()).hexdigest()[:8]
        tex_normal = f"btn_norm_{hash_base}"
        tex_hover = f"btn_hov_{hash_base}"
        tex_active = f"btn_act_{hash_base}"
        
        # Create textures (these go into texture registry, not parent)
        self._get_or_create_texture(tex_normal, width, height, normal_top, normal_bottom, label, corner_radius)
        self._get_or_create_texture(tex_hover, width, height, hover_top, hover_bottom, label, corner_radius)
        self._get_or_create_texture(tex_active, width, height, active_top, active_bottom, label, corner_radius)
        
        # Create shadow texture
        shadow_tag = f"shadow_{width}_{height}_{corner_radius}"
        if shadow_tag not in self._texture_cache:
            shadow_data = [20/255.0, 25/255.0, 22/255.0, 0.6] * (width * height)
            
            try:
                from PIL import Image, ImageDraw
                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle((0, 0, width-1, height-1), radius=corner_radius, fill=153)
                mask_arr = np.array(mask).astype(np.float32) / 255.0
                
                for i in range(height):
                    for j in range(width):
                        idx = (i * width + j) * 4 + 3
                        if idx < len(shadow_data):
                            shadow_data[idx] = float(mask_arr[i, j]) * 0.6
            except:
                pass
            
            with dpg.texture_registry():
                dpg.add_static_texture(width, height, shadow_data, tag=shadow_tag)
            self._texture_cache[shadow_tag] = shadow_tag
        
        # Build kwargs for parent - only include if not None
        # REMOVED pos=(3, 3) - this was causing all buttons to stack at same position!
        image_kwargs = {
            'width': width,
            'height': height
        }
        if parent is not None:
            image_kwargs['parent'] = parent
        
        # Add shadow image (NOT to texture registry) - flows naturally in layout
        dpg.add_image(shadow_tag, **image_kwargs)
        
        # Create wrapper callback that handles hover/active state
        def wrapper_callback(sender, app_data, user_data):
            # On click/release, reset to appropriate state
            data = dpg.get_item_user_data(sender)
            if data:
                if dpg.is_item_hovered(sender):
                    dpg.configure_item(sender, texture_tag=data['textures']['hover'])
                else:
                    dpg.configure_item(sender, texture_tag=data['textures']['normal'])
            # Call the actual callback (pass sender only if it expects an argument)
            if callback:
                try:
                    callback(sender)
                except TypeError:
                    # If callback doesn't take any arguments, call without args
                    callback()
        
        # Build kwargs for image button
        btn_kwargs = {
            'texture_tag': tex_normal,
            'width': width,
            'height': height,
            'callback': wrapper_callback,
            'frame_padding': 0,
            'background_color': (0, 0, 0, 0),
            'tint_color': (255, 255, 255, 255)
        }
        if parent is not None:
            btn_kwargs['parent'] = parent
        
        # Add main button image
        btn = dpg.add_image_button(**btn_kwargs)
        
        if tag:
            dpg.configure_item(btn, tag=tag)
        
        # Store state data
        dpg.set_item_user_data(btn, {
            'textures': {
                'normal': tex_normal,
                'hover': tex_hover,
                'active': tex_active
            },
            'original_pos': (0, 0)
        })
        
        return btn

    def _on_hover(self, sender):
        """Handle hover state."""
        try:
            user_data = dpg.get_item_user_data(sender)
            if user_data and dpg.is_item_hovered(sender):
                dpg.configure_item(sender, texture_tag=user_data['textures']['hover'])
            else:
                user_data = dpg.get_item_user_data(sender)
                if user_data:
                    dpg.configure_item(sender, texture_tag=user_data['textures']['normal'])
        except:
            pass

    def _on_active(self, sender):
        """Handle pressed state."""
        try:
            user_data = dpg.get_item_user_data(sender)
            if user_data:
                dpg.configure_item(sender, texture_tag=user_data['textures']['active'])
        except:
            pass

    def _on_deactivate(self, sender):
        """Handle release state."""
        try:
            user_data = dpg.get_item_user_data(sender)
            if user_data:
                if dpg.is_item_hovered(sender):
                    dpg.configure_item(sender, texture_tag=user_data['textures']['hover'])
                else:
                    dpg.configure_item(sender, texture_tag=user_data['textures']['normal'])
        except:
            pass


# Global factory
_gradient_factory = GradientButtonFactory()

def add_gradient_button(label: str, callback=None, parent=None, width: int = 150, 
                       height: int = 40, tag: str = None, **kwargs) -> int:
    """
    Create a gradient button with embossed 3D appearance.
    
    Args:
        label: Button text
        callback: Click callback
        parent: Parent container (tag string or int). If None, uses current DPG stack.
        width: Button width in pixels
        height: Button height in pixels
        tag: Optional unique tag for the button
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        Integer tag of the created button
    """
    return _gradient_factory.create_button(
        label=label,
        callback=callback,
        parent=parent,
        width=int(width),
        height=int(height),
        tag=tag
    )

def add_styled_slider(label: str, default_value: float = 1.0, min_value: float = 0.0, 
                     max_value: float = 10.0, parent=None, **kwargs) -> int:
    """Create a slider with Spaudible styling."""
    return dpg.add_slider_float(
        label=label,
        default_value=default_value,
        min_value=min_value,
        max_value=max_value,
        parent=parent,
        width=200,
        **kwargs
    )

def add_collapsible_section(label: str, parent=None, default_open: bool = True):
    """Helper to create a tree node (collapsible section) with proper styling."""
    return dpg.add_tree_node(
        label=label,
        parent=parent,
        default_open=default_open,
        bullet=False,
        span_full_width=True
    )

# =============================================================================
# WIDGET UTILITIES
# =============================================================================

def add_3d_button(label: str, callback=None, parent=None, width: int = 150, height: int = 34, **kwargs) -> int:
    """Create a button with semi-3D depth effect using offset shadow layer."""
    with dpg.group(horizontal=False, parent=parent):
        shadow = dpg.add_button(
            label="",
            width=width,
            height=height,
            pos=(2, 2),
            enabled=False
        )
        dpg.bind_item_theme(shadow, "spaudible_shadow_theme")
        
        btn = dpg.add_button(
            label=label,
            width=width,
            height=height,
            callback=callback,
            **kwargs
        )
    
    return btn

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
