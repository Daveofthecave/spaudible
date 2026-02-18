# ui/cli/console_utils.py
import os
import sys
from pathlib import Path
from wcwidth import wcswidth

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Print a clean header with title."""
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)

def print_menu(options):
    """Print numbered menu options."""
    print()
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print()

def get_choice(max_choice):
    """Get validated user choice."""
    while True:
        try:
            choice = input(f"  Enter choice [1-{max_choice}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= max_choice:
                return int(choice)
            print(f"    Please enter a number between 1 and {max_choice}.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)

def pad(s: str, width: int = 2) -> str:
    """Normalize emoji to fixed display width by right-padding with spaces.
    Detects variation selectors (6+ byte sequences) that cause width ambiguity
    across different terminals.
    
    Example:
        f"{icon('🔎')}Find Similar Songs"  # Text always starts at col 2
        f"  {icon('⚠️')}Warning"          # Text always starts at col 4
    """
    
    encoded = s.encode('utf-8')
    
    # If it contains U+FE0F (Variation Selector-16), it's a text symbol 
    # masquerading as emoji. These render as width 2 on modern terminals 
    # but width 1 on older ones. We pad conservatively.
    if b'\xef\xb8\x8f' in encoded:  # U+FE0F in UTF-8
        # 6+ bytes: Symbol + VS16 (e.g., ⚙️, ⬅️, ⚖️, ℹ️)
        # Add 2 spaces: ensures alignment whether terminal treats as width 1 or 2
        return s + '  '
    
    # 4 bytes: Native emoji (e.g., 🔎, 🐌, 🧮, 📊) - consistently width 2
    if len(encoded) == 4:
        return s + ' '
    
    # Fallback to wcwidth
    w = wcswidth(s)
    if w is None or w < 0:
        w = len(s)

    return s + (' ' if w >= 2 else '  ')

def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms                  "
    elif seconds < 60:
        return f"{seconds:.1f}s                  "
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.0f}s                  "
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m                  "
