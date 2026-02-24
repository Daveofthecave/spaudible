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
    """Creates embossed buttons with S-curve gradients and proper 3D shading."""
    
    def __init__(self):
        self._texture_cache = {}
        self._font_path = Path(__file__).parent.parent.parent / "data" / "fonts" / "OpenSans-Regular.ttf"
        self._debug_mode = False  # Set to True for debugging hover/click events
        self._button_states = {}
        
        # Preset sizes for convenience
        self.SIZE_SMALL = (80, 24)
        self.SIZE_MEDIUM = (150, 24)
        self.SIZE_LARGE = (200, 32)
        self.SIZE_XLARGE = (300, 40)

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

    def update_all_buttons(self):
        """Update all tracked buttons each frame based on their current state.
        
        This should be called once per frame in the main render loop.
        """
        try:
            for btn_id, state in list(self._button_states.items()):
                if not dpg.does_item_exist(btn_id):
                    # Button was deleted, remove from tracking
                    del self._button_states[btn_id]
                    continue
                
                label = state.get('label', 'Unknown')
                is_hovered = dpg.is_item_hovered(btn_id)
                is_active = dpg.is_item_active(btn_id)
                
                # Determine desired state
                if is_active:
                    desired_state = 'active'
                elif is_hovered:
                    desired_state = 'hover'
                else:
                    desired_state = 'normal'
                
                # Only update if state changed
                if state['current_state'] != desired_state:
                    old_state = state['current_state']
                    state['current_state'] = desired_state
                    
                    if self._debug_mode:
                        print(f"[UPDATE] Button '{label}' (id={btn_id}): {old_state} -> {desired_state}")
                    
                    if desired_state == 'active':
                        dpg.configure_item(btn_id, texture_tag=state['textures']['active'])
                        dpg.configure_item(btn_id, pos=(2, 2))
                    elif desired_state == 'hover':
                        dpg.configure_item(btn_id, texture_tag=state['textures']['hover'])
                        dpg.configure_item(btn_id, pos=state['original_pos'])
                    else:  # normal
                        dpg.configure_item(btn_id, texture_tag=state['textures']['normal'])
                        dpg.configure_item(btn_id, pos=state['original_pos'])
                        
        except Exception as e:
            if self._debug_mode:
                print(f"[UPDATE] Error: {e}")

    def create_button(self, label: str, callback=None, parent=None, width: int = 150, height: int = 40, tag: str = None, corner_radius: int = 8) -> int:
        """Create an embossed 3D button with proper layering, hover effects, and transparent background."""
        width = int(width)
        height = int(height)
        corner_radius = int(corner_radius)
        
        if self._debug_mode:
            print(f"\n=== Creating Button: {label} ===")
            print(f"Dimensions: {width}x{height}")
        
        # Color definitions for embossed effect
        base = Colors.PRIMARY_DARK
        normal_top = tuple(min(255, int(c * 1.4)) for c in base)
        normal_bottom = tuple(max(0, int(c * 0.6)) for c in base)
        hover_top = tuple(min(255, int(c * 1.6)) for c in base)
        hover_bottom = tuple(max(0, int(c * 0.8)) for c in base)
        active_top = tuple(max(0, int(c * 0.7)) for c in base)
        active_bottom = tuple(min(255, int(c * 1.2)) for c in base)
        
        # Generate unique tags
        import hashlib
        hash_base = hashlib.md5(f"{label}_{width}_{height}_{tag or ''}".encode()).hexdigest()[:8]
        tex_normal = f"btn_norm_{hash_base}"
        tex_hover = f"btn_hov_{hash_base}"
        tex_active = f"btn_act_{hash_base}"
        shadow_tag = f"btn_shad_{hash_base}"
        
        if self._debug_mode:
            print(f"Texture tags: normal={tex_normal}")
        
        # Create textures
        self._get_or_create_texture(tex_normal, width, height, normal_top, normal_bottom, label, corner_radius)
        self._get_or_create_texture(tex_hover, width, height, hover_top, hover_bottom, label, corner_radius)
        self._get_or_create_texture(tex_active, width, height, active_top, active_bottom, label, corner_radius)
        
        # Create shadow texture
        if shadow_tag not in self._texture_cache:
            shadow_data = [20/255.0, 25/255.0, 22/255.0, 0.5] * (width * height)
            try:
                from PIL import Image, ImageDraw
                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle((0, 0, width-1, height-1), radius=corner_radius, fill=128)
                mask_arr = np.array(mask).astype(np.float32) / 255.0
                for i in range(height):
                    for j in range(width):
                        idx = (i * width + j) * 4 + 3
                        if idx < len(shadow_data):
                            shadow_data[idx] = float(mask_arr[i, j]) * 0.5
            except:
                pass
            with dpg.texture_registry():
                dpg.add_static_texture(width, height, shadow_data, tag=shadow_tag)
            self._texture_cache[shadow_tag] = shadow_tag
        
        # Create container as child_window to establish local coordinate system for proper positioning
        container_kwargs = {
            'width': width + 3,  # Extra space for shadow offset
            'height': height + 3,
            'border': False,
            'no_scrollbar': True,
            'no_scroll_with_mouse': True,
            'autosize_x': False,
            'autosize_y': False,
        }
        if parent is not None:
            container_kwargs['parent'] = parent
        
        container = dpg.add_child_window(**container_kwargs)
        
        if self._debug_mode:
            print(f"Container created: {container}")
        
        # Add shadow first (behind), positioned at offset (3, 3) relative to container
        dpg.add_image(shadow_tag, width=width, height=height, pos=(3, 3), parent=container)
        
        # Create the actual button (image_button) at (0, 0), overlaying the shadow
        btn = dpg.add_image_button(
            texture_tag=tex_normal,
            width=width,
            height=height,
            pos=(0, 0),
            parent=container,
            callback=callback,
            frame_padding=0,
            background_color=(0, 0, 0, 0),
            tint_color=(255, 255, 255, 255)
        )
        
        if tag:
            dpg.configure_item(btn, tag=tag)
        
        if self._debug_mode:
            print(f"Button created: {btn}")
        
        # Store texture references and original position
        self._button_states[btn] = {
            'textures': {
                'normal': tex_normal,
                'hover': tex_hover,
                'active': tex_active
            },
            'original_pos': (0, 0),
            'label': label,
            'current_state': 'normal'
        }
        
        # Create transparent theme for this button to eliminate green background
        with dpg.theme() as transparent_btn_theme:
            with dpg.theme_component(dpg.mvImageButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 0, 0, 0))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
        dpg.bind_item_theme(btn, transparent_btn_theme)
        
        if self._debug_mode:
            print(f"=== Button creation complete ===\n")
        
        return btn

    def create_small(self, label: str, callback=None, parent=None, tag: str = None) -> int:
        """Create a small button (80x24px)."""
        return self.create_button(label, callback, parent, self.SIZE_SMALL[0], self.SIZE_SMALL[1], tag)

    def create_medium(self, label: str, callback=None, parent=None, tag: str = None) -> int:
        """Create a medium button (150x24px)."""
        return self.create_button(label, callback, parent, self.SIZE_MEDIUM[0], self.SIZE_MEDIUM[1], tag)

    def create_large(self, label: str, callback=None, parent=None, tag: str = None) -> int:
        """Create a large button (200x32px)."""
        return self.create_button(label, callback, parent, self.SIZE_LARGE[0], self.SIZE_LARGE[1], tag)

    def create_xlarge(self, label: str, callback=None, parent=None, tag: str = None) -> int:
        """Create an extra-large button (300x40px)."""
        return self.create_button(label, callback, parent, self.SIZE_XLARGE[0], self.SIZE_XLARGE[1], tag)

    def cleanup(self):
        """Clear all tracked buttons. Call when rebuilding UI to prevent memory leaks."""
        self._button_states.clear()


# Global factory
_gradient_factory = GradientButtonFactory()

def add_gradient_button(label: str, callback=None, parent=None, width: int = 150, 
                       height: int = 40, tag: str = None, **kwargs) -> int:
    """
    Create a gradient button with embossed 3D appearance.
    
    Args:
        label: Button text
        callback: Click callback function
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

def add_small_button(label: str, callback=None, parent=None, tag: str = None) -> int:
    """Create a small button (80x24px) with embossed 3D appearance."""
    return _gradient_factory.create_small(label, callback, parent, tag)

def add_medium_button(label: str, callback=None, parent=None, tag: str = None) -> int:
    """Create a medium button (150x24px) with embossed 3D appearance."""
    return _gradient_factory.create_medium(label, callback, parent, tag)

def add_large_button(label: str, callback=None, parent=None, tag: str = None) -> int:
    """Create a large button (200x32px) with embossed 3D appearance."""
    return _gradient_factory.create_large(label, callback, parent, tag)

def add_xlarge_button(label: str, callback=None, parent=None, tag: str = None) -> int:
    """Create an extra-large button (300x40px) with embossed 3D appearance."""
    return _gradient_factory.create_xlarge(label, callback, parent, tag)


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
