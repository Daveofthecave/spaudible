# ui/cli/menu_system/main_menu_handlers/core_search.py
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Any
from ui.cli.console_utils import print_header, format_elapsed_time, clear_screen
from .input_router import route_input
from .utils import (
    check_preprocessed_files,
    get_metadata_db_path,
    format_track_display,
    get_similarity_color,
    save_playlist
)
from core.similarity_engine.orchestrator import SearchOrchestrator
from core.vectorization.canonical_track_resolver import build_canonical_vector
from config import PathConfig, REGION_FILTER_STRENGTH
from core.utilities.config_manager import config_manager

def handle_core_search() -> str:
    """Main search interface."""
    print_header("Find Similar Songs")
    
    print("\n🔍 What would you like to find similar songs for?\n")
    print("   Enter any of the following:\n")

    print("   • Spotify track URL (https://open.spotify.com/track/...)")
    print("   • 22-character Spotify track ID (eg. 0eGsygTp906u18L0Oimnem)")
    # print("   • Audio file (coming soon)")
    # print("   • Search query (artist, song, album) (coming soon)")
    # print("   • Spotify playlist URL (coming soon)")
    # print("   • ISRC code (coming soon)")
    
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
            elif input_type == "audio_file":
                result = _handle_audio_file(processed_data)
            elif input_type == "text_query":
                result = _handle_text_search(processed_data)
            elif input_type == "spotify_playlist_url":
                result = _handle_spotify_playlist(processed_data)
            elif input_type == "isrc_code":
                result = _handle_isrc_code(processed_data)
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
    top_k: int = 25,
    search_mode: str = "sequential",
    with_metadata: bool = True,
    deduplicate: Optional[bool] = None
) -> str:
    """
    Core search function for track IDs.
    """
    print(f"   Finding songs similar to track: {track_id}")
    print(f"   Using algorithm: {config_manager.get_algorithm_name()}")     
    
    # Check preprocessed files
    files_exist, error_msg = check_preprocessed_files()
    if not files_exist:
        print(f"  ❌ {error_msg}")
        input("\n  Press Enter to return...")
        return "main_menu"
    
    try:
        # Build vector for the track
        start_time = time.time()
        vector, track_data = build_canonical_vector(track_id)
        
        if vector is None or all(v == -1.0 for v in vector):
            print("  ❌ Could not build vector for this track.")
            print("  Make sure the track exists in the database.")
            input("\n  Press Enter to return...")
            return "main_menu"
        
        vector_time = time.time() - start_time
        print(f"   Track converted to vector in {format_elapsed_time(vector_time)}")
        
        # Show track info
        if track_data:
            track_name = track_data.get('name', 'Unknown Track')
            artist_names = track_data.get('artist_names', [])
            artist_display = ', '.join(artist_names) if artist_names else 'Unknown Artist'
            print(f"   Searching for songs similar to:\n")
            print(f"   🎵   {track_name} - {artist_display}\n")

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
            print(f"\n  ✅ Found {len(results)} similar tracks in {format_elapsed_time(search_time)}:")
            print("  " + "─" * (65 - 2))
            
            for i, result in enumerate(results, 1):
                if len(result) == 3:
                    result_track_id, similarity, metadata = result
                    color = get_similarity_color(similarity)
                    track_name = metadata.get('track_name', 'Unknown')
                    artist_name = metadata.get('artist_name', 'Unknown')
                    print(f"  {i:2d}. {color} {similarity:.4f} - {track_name} - {artist_name}")
                else:
                    result_track_id, similarity = result
                    color = get_similarity_color(similarity)
                    print(f"  {i:2d}. {color} {similarity:.4f} - {result_track_id}")

            # Post-search options
            print("\n  📋 Options:")
            print("  1. Search another track")
            print("  2. Save results as playlist")
            print("  3. Return to main menu")
            
            choice = input("\n  Choice (1-3): ").strip()
            
            if choice == '1':
                # Clear screen and return to search screen
                clear_screen()
                print_header("Find Similar Songs")
                print("\n🔍 What would you like to find similar songs for?\n")
                print("   Enter any of the following:\n")
                print("   • Spotify track URL (https://open.spotify.com/track/...)")
                print("   • 22-character Spotify track ID (eg. 0eGsygTp906u18L0Oimnem)")
                return "core_search"
            elif choice == '2':
                playlist_name = input("  Enter playlist name: ").strip() or "Similar Songs"
                filename = save_playlist(results, playlist_name)
                print(f"  ✅ Playlist saved to: {filename}")
                input("\n  Press Enter to continue...")
                # Stay in results screen after saving
                return "core_search"
            elif choice == '3':
                return "main_menu"
            else:
                print("  ❌ Invalid choice, returning to main menu.")
                return "main_menu"
        
        return "main_menu"
        
    except Exception as e:
        print(f"  ❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        input("\n  Press Enter to return...")
        return "main_menu"

# Placeholder implementations for other handlers
def _handle_audio_file(file_path: str) -> str:
    print(f"  🔊 Analyzing audio file: {file_path}")
    print("  🚧 Audio file analysis coming soon!")
    input("\n  Press Enter to return...")
    return "main_menu"

def _handle_text_search(query: str) -> str:
    print(f"  🔤 Searching for: '{query}'")
    print("  🚧 Text search coming soon!")
    input("\n  Press Enter to return...")
    return "main_menu"

def _handle_spotify_playlist(playlist_id: str) -> str:
    print(f"  📁 Analyzing playlist: {playlist_id}")
    print("  🚧 Playlist analysis coming soon!")
    input("\n  Press Enter to return...")
    return "main_menu"

def _handle_isrc_code(isrc: str) -> str:
    print(f"  📄 Searching by ISRC: {isrc}")
    print("  🚧 ISRC search coming soon!")
    input("\n  Press Enter to return...")
    return "main_menu"

def _handle_unknown_input(user_input: str) -> str:
    print(f"  🤔 I didn't understand: '{user_input}'")
    print("\n  Please try one of these formats:")
    print("  • Spotify URL: https://open.spotify.com/track/0eGsygTp906u18L0Oimnem")
    print("  • Track ID: 003vvx7Niy0yvhvHt4a68B")
    print("  • Audio file: /path/to/song.mp3")
    print("  • Search: 'Mr. Brightside The Killers'")
    return "main_menu"
