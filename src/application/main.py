import dash
import dash_bootstrap_components as dbc
import dash_split_pane
from dash import html, dcc
from config import BODY_HEIGHT, COLORS
from styles import get_index_string
from components import *
from callbacks import register_callbacks

# Application setup
selected_images = {}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])
app.title = "Image Recommender"
app.index_string = get_index_string()


def build_settings_controls():
    """Creates all settings controls with new structure."""
    return [
        create_embeddings_section(),
        create_histograms_section(),
        create_kmeans_section(),
        create_hashing_section(),
    ]


# Main layout definition
app.layout = html.Div(
    [
        # Data stores for client-side state management
        dcc.Store(id="image-store", data={}),
        dcc.Store(id="images-count-store", data=0),
        html.Div(id="refresh-trigger", style={"display": "none"}),
        
        # Modal and header components
        create_modal(),
        create_header(),
        
        # Main content area with split pane
        html.Div(
            [
                dash_split_pane.DashSplitPane(
                    children=[
                        # Left panel - Image selection and results
                        html.Div(
                            [
                                create_image_card(),
                                dbc.Button(
                                    "Search",
                                    id={"type": "action-btn", "index": "search"},
                                    size="lg",
                                    color="outline-light",
                                    style={
                                        "width": "100px",
                                        "margin": "10px auto",
                                        "display": "block",
                                    },
                                ),
                                html.Div(id="operation-results"),
                                create_results_card(),
                            ],
                            style={
                                "height": BODY_HEIGHT,
                                "display": "flex",
                                "flexDirection": "column",
                                "background": COLORS["background"],
                                "minHeight": 0,
                                "padding": "10px 20px 20px 20px",
                            },
                        ),
                        # Right panel - Settings and controls
                        html.Div(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H3(
                                                "⚙️ Settings",
                                                className="text-center m-0 text-white",
                                            ),
                                            className="bg-secondary",
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Accordion(
                                                    build_settings_controls(),
                                                    start_collapsed=True,
                                                    className="accordion-sketchy",
                                                )
                                            ],
                                            className="bg-secondary p-2",
                                        ),
                                    ],
                                    className="h-100 bg-secondary",
                                    style={
                                        "border": "none",
                                        "borderTopLeftRadius": "8px",
                                    },
                                )
                            ],
                            style={"height": BODY_HEIGHT, "padding": 0},
                        ),
                    ],
                    id="main-splitter",
                    split="vertical",
                    size="63%",
                    minSize=1000,
                    maxSize=-420,
                    style={"height": BODY_HEIGHT},
                )
            ],
            style={"height": BODY_HEIGHT, "background": COLORS["background"]},
        ),
    ],
    style={"height": "100vh", "display": "flex", "flexDirection": "column"},
)

# Register all application callbacks
register_callbacks(app, selected_images)

if __name__ == "__main__":
    app.run(debug=True)