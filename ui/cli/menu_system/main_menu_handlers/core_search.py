# ui/cli/menu_system/main_menu_handlers/core_search.py
import os
import sys
import time
import math
from pathlib import Path
from typing import Optional, Tuple, Any
from .input_router import route_input
from .text_search import (
    interactive_text_search, simple_text_search_fallback
)
from .utils import (
    check_preprocessed_files, get_metadata_db_path, format_track_display,
    get_similarity_color, save_playlist
)
from ui.cli.console_utils import (
    print_header, format_elapsed_time, clear_screen, print_menu, get_choice
)
from core.vectorization.canonical_track_resolver import (
    build_canonical_vector, get_resolver
)
from core.similarity_engine.orchestrator import SearchOrchestrator
from config import PathConfig, REGION_FILTER_STRENGTH, EXPECTED_VECTORS
from core.utilities.config_manager import config_manager

def _print_search_prompt():
    print("\n🔍 What would you like to find similar songs for?\n")
    print("   Enter any of the following:\n")

    print("   • Song, artist, or album (eg. Muse Knights of Cydonia)")    
    print("   • Spotify track URL (https://open.spotify.com/track/...)")
    print("   • Spotify track ID (eg. 0eGsygTp906u18L0Oimnem)")
    print("   • ISRC code (eg. GBARL9300135)")
    # print("   • Audio file (coming soon)")
    # print("   • Spotify playlist URL (coming soon)")

def handle_core_search() -> str:
    """Main search interface."""
    print_header("Find Similar Songs")
    
    _print_search_prompt()
    
    while True:
        user_input = input("\n   Enter input (or type 'back' to return): ").strip()
        
        if user_input.lower() == 'back':
            return "main_menu"
        
        if not user_input:
            print("  ❌ Please enter something.")
            continue
        
        # Route the input
        input_type, processed_data = route_input(user_input)
        print(f"\n   Detected input type: {input_type}")
        
        try:
            if input_type in ["spotify_track_url", "track_id"]:
                result = _search_by_track_id(processed_data)
            elif input_type == "isrc_code":
                result = _handle_isrc_code(processed_data)
            elif input_type == "audio_file":
                result = _handle_audio_file(processed_data)
            elif input_type == "text_query":
                result = _handle_text_search(processed_data)
            elif input_type == "spotify_playlist_url":
                result = _handle_spotify_playlist(processed_data)
            else:
                result = _handle_unknown_input(user_input)
            
            if result == "main_menu":
                return "main_menu"
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print("  Please try again.")
            continue


def _search_by_track_id(
    track_id: str,
    top_k: Optional[int] = None,
    search_mode: str = "sequential",
    with_metadata: bool = True,
    deduplicate: Optional[bool] = None
) -> str:
    """
    Core search function for track IDs.
    """
    if top_k is None:
        top_k = config_manager.get_top_k()

    # print(f"   Finding songs similar to track: {track_id}")
    print(f"   Using algorithm: {config_manager.get_algorithm_name()}")     
    
    # Check preprocessed files
    files_exist, error_msg = check_preprocessed_files()
    if not files_exist:
        print(f"  ❌ {error_msg}")
        input("\n   Press Enter to return...")
        return "main_menu"
    
    try:
        # Build vector for the track
        start_time = time.time()
        vector, track_data = build_canonical_vector(track_id)

        '''
        # Debug: Validate query vector
        print(f"  🔍 Query vector range: [{min(v for v in vector if v != -1):.3f}, {max(v for v in vector if v != -1):.3f}]")
        print(f"  🔍 Query NaNs: {sum(1 for v in vector if math.isnan(v))}")
        print(f"  🔍 Query valid dims: {sum(1 for v in vector if v != -1)}/32")
        '''
        
        if vector is None or all(v == -1.0 for v in vector):
            print("  ❌ Could not build vector for this track.")
            print("  Make sure the track exists in the database.")
            input("\n   Press Enter to return...")
            return "main_menu"
        
        vector_time = time.time() - start_time
        print(f"   Track converted to vector in {format_elapsed_time(vector_time)}")
        
        # Show track info
        if track_data:
            track_name = track_data.get('name', 'Unknown Track')
            artist_names = track_data.get('artist_names', [])
            artist_display = ', '.join(artist_names) if artist_names else 'Unknown Artist'
            release_date = track_data.get('release_date', '')
            year = release_date[:4] if release_date and len(release_date) >= 4 else ''
            year_str = f" ({year})" if year else ''
            
            print(f"   Searching {EXPECTED_VECTORS:,} vectors for songs similar to:\n")
            print(f"   🎵  {track_name} - {artist_display}{year_str}")

        force_cpu = config_manager.get_force_cpu()
        
        # Initialize search orchestrator
        metadata_db = get_metadata_db_path()
        orchestrator = SearchOrchestrator(
            vectors_path=str(PathConfig.get_vector_file()),
            index_path=str(PathConfig.get_index_file()),
            metadata_db=get_metadata_db_path(),
            chunk_size=100_000_000,
            use_gpu=True,
            force_cpu=force_cpu
        )

        search_mode = "sequential"   
        
        # Run search - pass query_track_id for region filtering
        search_start = time.time()
        results = orchestrator.search(
            vector,
            search_mode=search_mode,
            top_k=top_k,
            with_metadata=with_metadata,
            deduplicate=deduplicate,
            query_track_id=track_id,
            region_strength=config_manager.get_region_strength()
        )
        
        search_time = time.time() - search_start
        orchestrator.close()
        
        # Display results
        print_header("Search Results")
        if not results:
            print("\n  ❌ No similar tracks found.")
        else:
            print(f"\n  ✅ Found {len(results)} similar tracks in {format_elapsed_time(search_time).strip()}:")
            print("  " + "─" * (65 - 2))
            
            for i, result in enumerate(results, 1):
                if len(result) == 3:
                    result_track_id, similarity, metadata = result
                    color = get_similarity_color(similarity)
                    track_name = metadata.get('track_name', 'Unknown')
                    artist_name = metadata.get('artist_name', 'Unknown')
                    artist_name = artist_name.replace(',', ', ')
                    year = metadata.get('album_release_year', '')
                    year_str = f" ({year})" if year else ''
                    print(f"  {i:2d}. {color} {similarity:.4f} - {track_name} - {artist_name}{year_str}")
                else:
                    result_track_id, similarity = result
                    color = get_similarity_color(similarity)
                    print(f"  {i:2d}. {color} {similarity:.4f} - {result_track_id}")

            # Post-search options            
            options = ["Search another track", "Save results as playlist", "Return to main menu"]
            print_menu(options)
            choice = get_choice(len(options))

            if choice == 1:
                # Clear screen and return to search screen
                clear_screen()
                print_header("Find Similar Songs")
                _print_search_prompt()
                return "core_search"
            elif choice == 2:
                playlist_name = input("  Enter playlist name: ").strip() or "Similar Songs"
                filename = save_playlist(results, playlist_name)
                print(f"  ✅ Playlist saved to: {filename}")
                input("\n  Press Enter to continue...")
                # Stay in results screen after saving
                return "core_search"
            elif choice == 3:
                return "main_menu"
            else:
                print("  ❌ Invalid choice, returning to main menu.")
                return "main_menu"
        
        return "main_menu"
        
    except Exception as e:
        print(f"  ❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        input("\n   Press Enter to return...")
        return "main_menu"

def _handle_isrc_code(isrc: str) -> str:
    """
    Handle ISRC code input by resolving to track ID first.
    Requires main database to be present.
    """
    # print(f"📄 Resolving ISRC: {isrc}")
    
    # Check if databases exist
    main_db_path = PathConfig.get_main_db()
    if not main_db_path.exists():
        print("\n  ❌ This feature requires Spotify database files.")
        print(f"     Missing: {main_db_path}")
        print("\n  ISRC search needs the database to map ISRC codes to track IDs.")
        print("  Please ensure data/databases/ contains the Spotify databases.")

        print("\n  Options:")
        options = ["Use Spotify Track ID instead", "Return to search"]
        print_menu(options)
        choice = get_choice(len(options))
        if choice == 1:
            track_id_input = input(" Enter track ID: ").strip()
            input_type, processed_data = route_input(track_id_input)
            if input_type in ["spotify_track_url", "track_id"]:
                return _search_by_track_id(processed_data)
            else:
                print("\n ❌ Invalid track ID format.")
                print(" Please enter a valid 22-character Spotify track ID or URL.")
                input("\n Press Enter to continue...")
                return "core_search"
    
    # Resolve ISRC to track ID
    resolver = get_resolver()
    track_id = resolver.resolve_isrc(isrc)
    
    if not track_id:
        print(f"\n❌ No track found with ISRC: {isrc}")
        print("   Make sure the ISRC is correct and the track exists in the database.")
        input("\n   Press Enter to return...")
        return "core_search"
    
    # Show resolved track info
    metadata = resolver.get_track_info(track_id)
    if metadata:
        track_name = metadata.get('track_name', 'Unknown')
        artist_name = metadata.get('artists', ['Unknown'])[0] if metadata.get('artists') else 'Unknown'
        # print(f"\n  Found: {track_name} - {artist_name}")
    
    # Proceed with normal similarity search
    return _search_by_track_id(track_id)

# Placeholder implementations for other handlers
def _handle_audio_file(file_path: str) -> str:
    print(f"   Analyzing audio file: {file_path}")
    print("   Audio file analysis coming soon!")
    print("\n   This feature will require Spotify databases for metadata lookup.")
    input("\n   Press Enter to return to search...")
    return "core_search"

def _handle_text_search(query: str) -> str:
    """
    Handle text-based search queries (eg. "Keane Perfect Symmetry").
    Uses interactive CLI with arrow-key navigation.
    """
    print_header("Text Search")
    # print(f"\n  Searching for: '{query}'")
    print()
    
    # Check preprocessed files
    files_exist, error_msg = check_preprocessed_files()
    if not files_exist:
        print(f"\n  ❌ {error_msg}")
        input("\n  Press Enter to return...")
        return "core_search"
    
    # Use interactive search
    track_id = interactive_text_search(query)
    
    if track_id:
        # User selected a track, now find similar songs
        return _search_by_track_id(track_id)
    else:
        # User cancelled
        return "core_search"

def _handle_spotify_playlist(playlist_id: str) -> str:
    print(f"   Analyzing playlist: {playlist_id}")
    print("   Playlist analysis coming soon!")
    print("\n   This feature will require Spotify API access.")
    input("\n   Press Enter to return to search...")
    return "core_search"

def _handle_unknown_input(user_input: str) -> str:
    print(f"  🤔 I didn't understand: '{user_input}'")
    print("\n  Please try one of these formats:")
    print("  • Spotify URL: https://open.spotify.com/track/0eGsygTp906u18L0Oimnem")
    print("  • Track ID: 003vvx7Niy0yvhvHt4a68B")
    print("  • ISRC code: USIR20400274")
    input("\n   Press Enter to return to search...")
    return "core_search"
