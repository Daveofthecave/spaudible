# ui/cli/console_utils.py
import os
import sys
from pathlib import Path

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Print a clean header with title."""
    width = 60
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

def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
