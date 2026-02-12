# ui/cli/menu_system/preprocessing_prompt.py
from ui.cli.console_utils import print_header, print_menu, get_choice

def screen_preprocessing_prompt():
    """Screen 2: Ask user to start preprocessing."""
    print_header("Spaudible - Setup Required")

    print("\n  ✅ All required databases found!\n")

    print("  ⚠️ IMPORTANT: One-time setup required\n")

    print("  Spotify's music metadata databases hold over")
    print("  256 million cataloged tracks containing detailed")
    print("  attributes like genre, tempo, and acousticness.\n")

    print("  To make song searches fast and accurate, Spaudible")
    print("  needs to extract attributes from these databases and")
    print("  convert them into an optimized numerical search index.\n")

    print("  This process takes about 1-3 hours on an NVMe SSD,")
    print("  and requires about 34 GB of additional disk space.")
    print("  Your computer can be used normally during this time.\n")
    
    print("  Would you like to start the setup process now?")
    
    options = [
        "Yes, start setup",
        "No, quit program"
    ]
    
    print_menu(options)
    choice = get_choice(len(options))
    
    if choice == 1:
        print("\n     Starting setup process...\n")
        return "start_preprocessing"
    else:
        print("\n  Setup cancelled.")
        print("  Run Spaudible again when ready to set up.")
        return "exit"
