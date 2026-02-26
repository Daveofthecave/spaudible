# ui/gui/results_view.py
import dearpygui.dearpygui as dpg
from typing import List, Tuple

class ResultsView:
    def __init__(self, dpi_scale: float = 1.0):
        self.dpi_scale = float(dpi_scale)
        self.tag = "results_panel"
        self.container_tag = "results_container"
        self.placeholder_tag = "results_placeholder"
        self.results = []

    def _s(self, value) -> int:
        return int(float(value) * self.dpi_scale)

    def build(self):
        """Build - NO parent parameter."""
        dpg.add_text("Results", color=(100, 200, 255))
        dpg.add_separator()
        dpg.add_text(
            "Enter a search query above to find similar songs.",
            tag=self.placeholder_tag,
            color=(150, 150, 150),
        )
        with dpg.child_window(
            tag=self.container_tag,
            autosize_x=True,
            autosize_y=True,
            border=False,
            show=False,
        ):
            pass

    def clear_results(self):
        """Clear and show placeholder."""
        self.results = []
        if dpg.does_item_exist(self.container_tag):
            dpg.configure_item(self.container_tag, show=False)
        if dpg.does_item_exist(self.placeholder_tag):
            dpg.configure_item(self.placeholder_tag, show=True)

    def update_results(self, results: List[Tuple]):
        """Display results."""
        self.results = results
        if dpg.does_item_exist(self.placeholder_tag):
            dpg.configure_item(self.placeholder_tag, show=False)
        
        # Delete and recreate container to clear it
        if dpg.does_item_exist(self.container_tag):
            dpg.delete_item(self.container_tag)
        
        with dpg.child_window(
            tag=self.container_tag,
            autosize_x=True,
            autosize_y=True,
            border=False,
        ):
            if not results:
                dpg.add_text("No results found.")
                return
            
            dpg.add_text(f"Found {len(results)} similar songs:")
            dpg.add_separator()
            
            for i, result in enumerate(results, 1):
                if len(result) == 3:
                    _, similarity, metadata = result
                    name = metadata.get('track_name', 'Unknown')
                    artist = metadata.get('artist_name', '')
                    text = f"{i}. {similarity:.4f} - {name}"
                    if artist:
                        text += f" - {artist}"
                else:
                    text = f"{i}. Result"
                dpg.add_text(text)

    def show_loading(self, message: str = "Searching..."):
        if dpg.does_item_exist(self.placeholder_tag):
            dpg.set_value(self.placeholder_tag, message)
            dpg.configure_item(self.placeholder_tag, color=(150, 255, 150))
            dpg.configure_item(self.placeholder_tag, show=True)

    def hide_loading(self):
        if dpg.does_item_exist(self.placeholder_tag):
            dpg.configure_item(self.placeholder_tag, color=(150, 150, 150))
