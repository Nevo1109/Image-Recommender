import os
import subprocess
import json
from dash import html, Input, Output, State, callback_context, no_update, ALL, MATCH
from config import PROGRESS_COLOR, OPERATION_LABELS
from utils import (
    open_native_file_dialog,
    file_to_base64,
    write_progress_file,
    read_progress_file,
    update_display_from_selected_images,
    get_random_image_from_db,
)


def register_callbacks(app, selected_images):
    """Register all Dash callbacks for the application."""

    @app.callback(
        [
            Output("images-row", "children", allow_duplicate=True),
            Output("images-count-store", "data", allow_duplicate=True),
        ],
        [
            Input("browse-images-btn", "n_clicks"),
            Input("clear-all-btn", "n_clicks"),
            Input("random-image-btn", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def handle_image_actions(browse_clicks, clear_clicks, random_clicks):
        """Handle image-related actions: browse, clear, and random selection."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Clear all selected images
        if trigger_id == "clear-all-btn" and clear_clicks:
            selected_images.clear()
            return [], 0

        # Browse and add images from file system
        if trigger_id == "browse-images-btn" and browse_clicks:
            try:
                selected_files = open_native_file_dialog()
                if not selected_files:
                    return no_update, no_update

                # Check available slots (max 3 images)
                available_slots = 3 - len(selected_images)
                if available_slots <= 0:
                    return (
                        update_display_from_selected_images(selected_images)[0],
                        len(selected_images),
                    )

                # Process selected files
                for filepath in selected_files[:available_slots]:
                    try:
                        filename = os.path.basename(filepath)
                        original_name = filename
                        counter = 1
                        
                        # Handle duplicate filenames
                        while filename in selected_images:
                            name_parts = original_name.split(".")
                            if len(name_parts) > 1:
                                filename = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                            else:
                                filename = f"{original_name}_{counter}"
                            counter += 1

                        selected_images[filename] = file_to_base64(filepath)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        continue

                display_data = update_display_from_selected_images(selected_images)
                return display_data[0], display_data[1]

            except Exception as e:
                print(f"❌ Error: {str(e)}")
                return no_update, no_update

        # Add random image from database
        if trigger_id == "random-image-btn" and random_clicks:
            try:
                # Check available slots
                available_slots = 3 - len(selected_images)
                if available_slots <= 0:
                    print("⚠️ Maximum number of images (3) already selected")
                    return (
                        update_display_from_selected_images(selected_images)[0],
                        len(selected_images),
                    )

                # Get random image from database
                random_image_path = get_random_image_from_db()
                if not random_image_path:
                    print("❌ Could not get random image from database")
                    return no_update, no_update

                # Convert to Base64
                base64_data = file_to_base64(random_image_path)
                if not base64_data:
                    print(f"❌ Could not convert image to base64: {random_image_path}")
                    return no_update, no_update

                # Generate unique filename
                filename = os.path.basename(random_image_path)
                original_name = filename
                counter = 1
                while filename in selected_images:
                    name_parts = original_name.split(".")
                    if len(name_parts) > 1:
                        filename = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                    else:
                        filename = f"{original_name}_{counter}"
                    counter += 1

                # Add image to selection
                selected_images[filename] = base64_data
                print(f"✅ Added random image: {filename}")

                display_data = update_display_from_selected_images(selected_images)
                return display_data[0], display_data[1]

            except Exception as e:
                print(f"❌ Error adding random image: {str(e)}")
                return no_update, no_update

        return no_update, no_update

    @app.callback(
        [
            Output({"type": "progress-store", "index": MATCH}, "data"),
            Output({"type": "progress-poller", "index": MATCH}, "disabled"),
        ],
        [
            Input({"type": "action-btn", "index": MATCH}, "n_clicks"),
            Input({"type": "progress-poller", "index": MATCH}, "n_intervals"),
        ],
        State({"type": "progress-store", "index": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def handle_progress_operations(n_clicks, n_intervals, current_store):
        """Handle progress tracking for long-running operations."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        try:
            trigger = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
            op_id = trigger["index"]
        except Exception:
            return no_update, no_update

        # Start operation
        if "action-btn" in ctx.triggered[0]["prop_id"]:
            write_progress_file(op_id, 0, running=True, message="started from UI")
            return (
                {"value": 0, "running": True, "operation": op_id, "message": "started"},
                False,
            )

        # Poll progress
        data = read_progress_file(op_id)
        disabled = not data.get("running", False) or data.get("value", 0) >= 100
        return data, disabled

    @app.callback(
        [
            Output({"type": "progress-fill", "index": MATCH}, "style"),
            Output({"type": "progress-container", "index": MATCH}, "style"),
            Output({"type": "action-btn", "index": MATCH}, "children"),
        ],
        Input({"type": "progress-store", "index": MATCH}, "data"),
    )
    def render_operation_progress(store):
        """Render progress bar and button states for operations."""
        value = store.get("value", 0) if store else 0
        operation = store.get("operation", "unknown") if store else "unknown"

        # Progress fill styling
        fill_style = {
            "height": "100%",
            "width": f"{value}%",
            "transition": "width 0.1s linear",
            "backgroundColor": PROGRESS_COLOR,
            "position": "absolute",
            "top": 0,
            "left": 0,
            "zIndex": 1,
        }
        
        # Container styling
        container_style = (
            {"display": "none"}
            if value >= 100
            else {
                "position": "absolute",
                "top": 0,
                "left": 0,
                "width": "100%",
                "height": "100%",
                "backgroundColor": "rgba(255,255,255,0.18)",
                "borderRadius": "6px",
                "overflow": "hidden",
                "zIndex": 1,
            }
        )

        # Button content based on progress
        if value == 0:
            btn_content = OPERATION_LABELS.get(operation, "▶️ Start")
        elif value < 100:
            btn_content = f"🔄 {value}%"
        else:
            btn_content = f"✅ {operation.title()}"

        return fill_style, container_style, btn_content

    @app.callback(
        Output("operation-results", "children"),
        [Input({"type": "progress-store", "index": ALL}, "data")],
        prevent_initial_call=True,
    )
    def handle_completed_operations(all_stores):
        """Display completed operations summary."""
        completed = [
            store.get("operation")
            for store in (all_stores or [])
            if store and store.get("value") == 100
        ]
        
        if completed:
            return html.Div(
                [
                    html.H6("✅ Completed:", className="text-success mb-1"),
                    html.P(
                        f"{', '.join([op.title() for op in completed])}",
                        className="text-white-50 small mb-2",
                    ),
                ]
            )
        return no_update

    @app.callback(
        Output("folder-btn", "n_clicks"),
        Input("folder-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_folder(n_clicks):
        """Open data folder in Windows Explorer."""
        if n_clicks:
            subprocess.Popen(r'explorer "E:\data\recommender_data"')
        return 0

    @app.callback(
        [Output("images-row", "children"), Output("images-count-store", "data")],
        [Input("refresh-trigger", "children")],
        prevent_initial_call=True,
    )
    def update_images_and_count(refresh_trigger):
        """Update image display and count after refresh trigger."""
        return update_display_from_selected_images(selected_images)

    @app.callback(
        [Output("placeholder", "children"), Output("placeholder", "style")],
        Input("images-count-store", "data"),
    )
    def update_placeholder_from_store(image_count):
        """Update placeholder text based on image count."""
        if image_count > 0:
            return "", {"display": "none"}
        else:
            return (
                "Click + to add images or 🎲 for random (max 3)",
                {
                    "height": "100%",
                    "minHeight": "150px",
                    "display": "flex",
                },
            )

    @app.callback(
        [
            Output("modal", "is_open"),
            Output("modal-image", "src"),
            Output("modal-title", "children"),
            Output("refresh-trigger", "children"),
        ],
        [
            Input({"type": "img-click", "index": ALL}, "n_clicks"),
            Input("close-btn", "n_clicks"),
            Input("remove-btn", "n_clicks"),
        ],
        [State("modal", "is_open"), State("modal-title", "children")],
        prevent_initial_call=True,
    )
    def toggle_modal(img_clicks, close_clicks, remove_clicks, is_open, current_title):
        """Handle modal opening/closing and image removal."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update

        trigger = ctx.triggered[0]["prop_id"]
        trigger_value = ctx.triggered[0]["value"]

        # Open modal with clicked image
        if "img-click" in trigger and trigger_value and trigger_value > 0:
            import re

            match = re.search(r'"index":"([^"]+)"', trigger)
            if match:
                filename = match.group(1)
                if filename in selected_images:
                    return True, selected_images[filename], filename, no_update

        # Remove image from selection
        if (
            "remove-btn" in trigger
            and remove_clicks
            and remove_clicks > 0
            and current_title
            and current_title in selected_images
        ):
            selected_images.pop(current_title, None)
            return False, no_update, no_update, f"refresh-{remove_clicks}"

        # Close modal
        if "close-btn" in trigger and close_clicks and close_clicks > 0:
            return False, no_update, no_update, no_update

        return no_update, no_update, no_update, no_update