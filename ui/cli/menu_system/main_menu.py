# ui/cli/menu_system/main_menu.py
from ui.cli.console_utils import print_header, print_menu, get_choice
from .main_menu_handlers import handle_core_search, handle_settings

def screen_main_menu():
    """Main menu controller - routes to specialized handlers."""
    print_header("🎵 Spaudible - Song Discovery Tool")
    
    print()
    print("  Spaudible discovers music similar to any Spotify track")
    print("  you provide. Input a song name, track URL, track ID,")
    print("  ISRC, or audio file to receive a playlist of acoustically ")
    print("  similar songs from a collection of 256 million tracks.\n")

    options = [
        "🔎 Find Similar Songs",
        "⚙️  Settings",
        "🚪 Exit"
    ]
    
    print_menu(options)
    choice = get_choice(len(options))
    
    # Route to appropriate handler
    handlers = {
        1: handle_core_search,
        2: handle_settings,
        3: lambda: "exit"
    }
    
    handler = handlers.get(choice)
    if handler:
        result = handler()
        return result if result else "main_menu"
    
    return "main_menu"
