import dash_bootstrap_components as dbc
from dash import html, dcc


def create_progress_component(op_id: str, label: str, color: str):
    """Creates progress component with button"""
    return html.Div(
        [
            dcc.Store(
                id={"type": "progress-store", "index": op_id},
                data={"value": 0, "running": False, "operation": op_id},
            ),
            dcc.Interval(
                id={"type": "progress-poller", "index": op_id},
                interval=500,
                disabled=True,
            ),
            html.Div(
                [
                    html.Div(
                        html.Div(
                            id={"type": "progress-fill", "index": op_id},
                            style={"height": "100%", "width": "0%"},
                        ),
                        id={"type": "progress-container", "index": op_id},
                        style={
                            "position": "absolute",
                            "top": 0,
                            "left": 0,
                            "width": "100%",
                            "height": "100%",
                            "borderRadius": "6px",
                            "overflow": "hidden",
                        },
                    ),
                    dbc.Button(
                        label,
                        id={"type": "action-btn", "index": op_id},
                        size="lg",
                        color="secondary",
                        outline=True,
                        className="w-100 custom-transparent-btn",
                        style={
                            "zIndex": "15",
                            "position": "relative",
                            "pointerEvents": "auto",
                            "background": "transparent",
                        },
                    ),
                ],
                className="progress-btn-container mb-2",
                style={
                    "position": "relative",
                    "height": "48px",
                    "width": "100%",
                    "zIndex": "10",
                },
            ),
        ]
    )


def create_slider_row(label, slider_id, config):
    """Creates slider row"""
    return dbc.Row(
        [
            dbc.Col(
                html.Span(
                    label,
                    className="px-4 py-1 rounded-5 text-bg-light text-center d-inline-block",
                ),
                width=3,
                className="d-flex align-items-center",
            ),
            dbc.Col(
                dcc.Slider(
                    config["min"],
                    config["max"],
                    config["step"],
                    config.get("marks"),
                    config["value"],
                    id=slider_id,
                    tooltip=config.get("tooltip"),
                    className="px-0",
                ),
                width=9,
                className="px-2",
            ),
        ],
        className="g-1 mb-2 align-items-center",
    )


def create_dropdown_row(label, dropdown_id, options, value):
    """Creates dropdown row"""
    return dbc.Row(
        [
            dbc.Col(
                html.Span(
                    label,
                    className="px-4 py-1 rounded-5 text-bg-light text-center d-inline-block",
                ),
                width=3,
                className="d-flex align-items-center",
            ),
            dbc.Col(
                dcc.Dropdown(
                    options,
                    value=value,
                    id=dropdown_id,
                    searchable=False,
                    maxHeight=9999,
                ),
                width=True,
            ),
        ],
        className="g-1 mb-2",
    )


def create_switch_row(label, switch_id):
    """Creates switch row with label on the left, switch on the right"""
    return html.Div(
        dbc.Switch(
            id=switch_id,
            label=html.Span(
                label, className="px-4 py-1 rounded-5 text-bg-light text-center"
            ),
            className="switch-container",
            label_class_name="order-first",
        ),
        className="d-inline-block",
    )


def create_embeddings_section():
    """Creates embeddings section with model selection outside"""
    return dbc.AccordionItem(
        [
            # Model Selection outside the Create accordion
            html.Div(
                [
                    create_dropdown_row(
                        "Model",
                        "embedding-model",
                        [
                            {"label": "ViT-B/16", "value": "vitb16"},
                            {"label": "CLIP ViT-B/16", "value": "clip_vitb16"},
                            {"label": "ResNet50", "value": "resnet50"},
                        ],
                        None,
                    ),
                ],
                className="mb-3",
            ),
            # Create New Accordion
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            create_slider_row(
                                "Num Worker",
                                "num-worker-slider",
                                {
                                    "min": 1,
                                    "max": 16,
                                    "step": 1,
                                    "value": 8,
                                    "marks": {
                                        str(i): str(i) if i % 2 == 0 else ""
                                        for i in range(1, 17)
                                    },
                                    "tooltip": {"placement": "top"},
                                },
                            ),
                            create_slider_row(
                                "Batch Size",
                                "batch-size-slider",
                                {
                                    "min": 4,
                                    "max": 13,
                                    "step": 1,
                                    "value": 7,
                                    "marks": {str(i): str(2**i) for i in range(4, 14)},
                                },
                            ),
                            html.Div(
                                [
                                    create_switch_row(
                                        "Persistent Worker", "persistent-worker"
                                    ),
                                    create_switch_row("Pin Memory", "pin-memory"),
                                ],
                                className="d-flex justify-content-center gap-4 mb-2",
                            ),
                            create_slider_row(
                                "Prefetch Factor",
                                "prefetch-factor-slider",
                                {
                                    "min": 0,
                                    "max": 16,
                                    "step": 1,
                                    "value": 2,
                                    "marks": {
                                        str(i): str(i) if i % 2 == 0 else ""
                                        for i in range(0, 17)
                                    },
                                    "tooltip": {"placement": "top"},
                                },
                            ),
                            html.Hr(
                                style={
                                    "margin": "16px 0 12px 0",
                                    "borderColor": "#6c757d",
                                }
                            ),
                            create_progress_component(
                                "embeddings", "Start", "secondary"
                            ),
                        ],
                        title=html.H6(
                            "Create New",
                            className="mb-0",
                            style={"fontSize": "0.95rem"},
                        ),
                    )
                ],
                start_collapsed=True,
                className="accordion-clean",
            ),
        ],
        title=html.H4("Embeddings", className="mb-0"),
    )


def create_histograms_section():
    """Creates histograms section with colorspace outside"""
    return dbc.AccordionItem(
        [
            # Configuration outside the Create accordion
            html.Div(
                [
                    create_dropdown_row(
                        "Colorspace",
                        "hist-colorspace",
                        [
                            {"label": "LAB", "value": "lab"},
                            {"label": "HSV", "value": "hsv"},
                            {"label": "RGB", "value": "rgb"},
                        ],
                        "lab",
                    ),
                    create_slider_row(
                        "Bins",
                        "hist-bins-slider",
                        {"min": 8, "max": 256, "step": 8, "value": 32},
                    ),
                    html.Div(
                        [create_switch_row("Normalize", "hist-normalize")],
                        className="d-flex justify-content-center mb-2",
                    ),
                ],
                className="mb-3",
            ),
            # Create New Accordion
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            create_slider_row(
                                "Num Worker",
                                "hist-num-worker-slider",
                                {
                                    "min": 1,
                                    "max": 16,
                                    "step": 1,
                                    "value": 6,
                                    "marks": {
                                        str(i): str(i) if i % 2 == 0 else ""
                                        for i in range(1, 17)
                                    },
                                    "tooltip": {"placement": "top"},
                                },
                            ),
                            html.Hr(
                                style={
                                    "margin": "16px 0 12px 0",
                                    "borderColor": "#6c757d",
                                }
                            ),
                            create_progress_component(
                                "histograms", "Start", "secondary"
                            ),
                        ],
                        title=html.H6(
                            "Create New",
                            className="mb-0",
                            style={"fontSize": "0.95rem"},
                        ),
                    )
                ],
                start_collapsed=True,
                className="accordion-clean",
            ),
        ],
        title=html.H4("Histograms", className="mb-0"),
    )


def create_kmeans_section():
    """Creates KMeans section with clustering parameters outside"""
    return dbc.AccordionItem(
        [
            # Configuration outside the Create accordion
            html.Div(
                [
                    create_dropdown_row(
                        "Init Method",
                        "kmeans-init",
                        [
                            {"label": "k-means++", "value": "k-means++"},
                            {"label": "random", "value": "random"},
                        ],
                        None,
                    ),
                    create_slider_row(
                        "Clusters",
                        "kmeans-clusters-slider",
                        {
                            "min": 2,
                            "max": 50,
                            "step": 1,
                            "value": 5,
                            "marks": {
                                str(i): str(i) for i in range(2, 51) if i % 5 == 0
                            },
                            "tooltip": {"placement": "top"},
                        },
                    ),
                    create_slider_row(
                        "Max Iterations",
                        "kmeans-maxiter-slider",
                        {
                            "min": 50,
                            "max": 1000,
                            "step": 50,
                            "value": 300,
                            "marks": {
                                str(i): str(i) if i % 100 == 0 else ""
                                for i in range(50, 1001, 50)
                            },
                            "tooltip": {"placement": "top"},
                        },
                    ),
                ],
                className="mb-3",
            ),
            # Create New Accordion
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            create_slider_row(
                                "Num Worker",
                                "kmeans-nworker-slider",
                                {
                                    "min": 1,
                                    "max": 16,
                                    "step": 1,
                                    "value": 8,
                                    "marks": {
                                        str(i): str(i) if i % 2 == 0 else ""
                                        for i in range(1, 17)
                                    },
                                    "tooltip": {"placement": "top"},
                                },
                            ),
                            html.Hr(
                                style={
                                    "margin": "16px 0 12px 0",
                                    "borderColor": "#6c757d",
                                }
                            ),
                            create_progress_component(
                                "clustering", "Start", "secondary"
                            ),
                        ],
                        title=html.H6(
                            "Create New",
                            className="mb-0",
                            style={"fontSize": "0.95rem"},
                        ),
                    )
                ],
                start_collapsed=True,
                className="accordion-clean",
            ),
        ],
        title=html.H4("KMeans", className="mb-0"),
    )


def create_hashing_section():
    """Creates hashing section with hash parameters outside"""
    return dbc.AccordionItem(
        [
            # Configuration outside the Create accordion
            html.Div(
                [
                    create_dropdown_row(
                        "Hash Method",
                        "hash-method",
                        [
                            {"label": "Perceptual Hash (pHash)", "value": "phash"},
                            {"label": "Average Hash (aHash)", "value": "ahash"},
                            {"label": "Difference Hash (dHash)", "value": "dhash"},
                            {"label": "Wavelet Hash (wHash)", "value": "whash"},
                            {"label": "Color Hash (cHash)", "value": "chash"},
                            {"label": "Block Hash (bHash)", "value": "bhash"},
                        ],
                        None,
                    ),
                    create_slider_row(
                        "Hash Size",
                        "hash-size-slider",
                        {
                            "min": 3,
                            "max": 8,
                            "step": 1,
                            "value": 5,
                            "marks": {str(i): str(2**i) for i in range(3, 9)},
                        },
                    ),
                ],
                className="mb-3",
            ),
            # Create New Accordion
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            create_slider_row(
                                "Num Worker",
                                "hash-num-worker-slider",
                                {
                                    "min": 1,
                                    "max": 16,
                                    "step": 1,
                                    "value": 8,
                                    "marks": {
                                        str(i): str(i) if i % 2 == 0 else ""
                                        for i in range(1, 17)
                                    },
                                    "tooltip": {"placement": "top"},
                                },
                            ),
                            html.Hr(
                                style={
                                    "margin": "16px 0 12px 0",
                                    "borderColor": "#6c757d",
                                }
                            ),
                            create_progress_component("hashing", "Start", "secondary"),
                        ],
                        title=html.H6(
                            "Create New",
                            className="mb-0",
                            style={"fontSize": "0.95rem"},
                        ),
                    )
                ],
                start_collapsed=True,
                className="accordion-clean",
            ),
        ],
        title=html.H4("Hashing", className="mb-0"),
    )


def create_modal():
    """Creates modal for image display"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
            dbc.ModalBody(
                [
                    html.Img(
                        id="modal-image",
                        style={
                            "maxWidth": "100%",
                            "maxHeight": "80vh",
                            "objectFit": "contain",
                            "display": "block",
                            "margin": "0 auto",
                        },
                    ),
                    dbc.ButtonGroup(
                        [
                            dbc.Button("🗑️ Remove", id="remove-btn", color="danger"),
                            dbc.Button("✕ Close", id="close-btn", color="secondary"),
                        ],
                        className="d-flex justify-content-center mt-3",
                    ),
                ]
            ),
        ],
        id="modal",
        size="lg",
        is_open=False,
    )


def create_header():
    """Creates header"""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "📊 Graph",
                            id="graph-btn",
                            color="light",
                            outline=True,
                            size="lg",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.H1(
                            "Image Recommender",
                            className="text-center m-0",
                            style={"color": "#c181fd"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "📁 Folder",
                            id="folder-btn",
                            color="light",
                            outline=True,
                            size="lg",
                        ),
                        width=3,
                        className="text-end",
                    ),
                ],
                align="center",
                className="h-100 px-3",
            )
        ],
        style={"height": "80px", "background": "#212529"},
    )


def create_image_card():
    """Creates image selection card with improved button arrangement"""
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div(
                        [
                            html.H4(
                                "Select Images",
                                className="text-center m-0 p-2",
                                style={"color": "#dab2ff"},
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "➕",
                                        id="browse-images-btn",
                                        className="image-action-btn btn-add",
                                        size="sm",
                                        title="Add images from file browser",
                                    ),
                                    dbc.Button(
                                        "🎲",
                                        id="random-image-btn",
                                        className="image-action-btn btn-random",
                                        size="sm",
                                        title="Add random image from database",
                                    ),
                                    dbc.Button(
                                        "🗑️",
                                        id="clear-all-btn",
                                        className="image-action-btn btn-delete",
                                        size="sm",
                                        title="Clear all images",
                                    ),
                                ],
                                className="position-absolute top-0 end-0 mt-1 me-1",
                                style={"display": "flex", "gap": "4px"},
                            ),
                        ],
                        className="position-relative",
                    )
                ],
                className="bg-secondary",
            ),
            dbc.CardBody(
                [
                    html.Div(
                        id="images-row",
                        className="w-100",
                        style={
                            "flex": "1 1 auto",
                            "minHeight": 0,
                            "overflowY": "auto",
                            "padding": "0",
                            "margin": "0",
                        },
                    ),
                    html.Div(
                        "Click ➕ to add images or 🎲 for random (max 3)",
                        id="placeholder",
                        className="text-white d-flex justify-content-center align-items-center",
                        style={"height": "100%", "minHeight": "150px"},
                    ),
                ],
                className="bg-secondary d-flex flex-column p-0",
                style={"overflowY": "auto", "flex": "1 1 auto", "minHeight": 0},
            ),
        ],
        className="h-100 mb-0",
    )


def create_results_card():
    """Creates results card"""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(
                    "Results",
                    className="text-center m-0",
                    style={"color": "#dab2ff"},
                ),
                className="bg-secondary",
            ),
            dbc.CardBody(
                html.Div(
                    "Upload images to see results",
                    id="results",
                    className="d-flex justify-content-center align-items-center text-white py-5",
                    style={"height": "100%"},
                ),
                className="bg-secondary d-flex flex-column",
                style={"overflowY": "auto", "flex": "1 1 auto", "minHeight": 0},
            ),
        ],
        className="h-100",
    )
