import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash_deck
import pydeck as pdk
import pandas as pd
import numpy as np
import os

CSV_FILE = "E:/data/recommender_data/umap_vitb16_points.csv"

def load_and_process_data():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} not found")
    df = pd.read_csv(CSV_FILE)
    for col in ['umap_x', 'umap_y', 'umap_z', 'ids']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    label_col = next((c for c in df.columns if any(k in c.lower() for k in ["label", "cluster", "class"])), None)
    points, colors = [], {}
    if label_col:
        uniq = df[label_col].unique()
        np.random.seed(42)
        colors = {l: np.random.randint(50, 255, 3).tolist() for i, l in enumerate(uniq)}

    for _, r in df.iterrows():
        try:
            points.append({
                "position": [float(r.umap_x), float(r.umap_y), float(r.umap_z)],
                "color": colors.get(r[label_col], [100, 150, 255]) if label_col else [100, 150, 255],
                "id": int(r.ids),
                "label": str(r[label_col]) if label_col else "N/A"
            })
        except: continue
    return df, points, label_col

df, points, label_col = load_and_process_data()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "UMAP 3D Visualizer"

center = [df.umap_x.mean(), df.umap_y.mean(), df.umap_z.mean()]
ranges = [np.ptp(df["umap_x"]), np.ptp(df["umap_y"]), np.ptp(df["umap_z"])]
max_range = max(ranges)

default_view_state = dict(
    longitude=center[0], latitude=center[1],
    zoom=max(0, 8 - np.log2(max_range + 1)),
    pitch=45, bearing=0,
    target=center, rotationX=15, rotationOrbit=0
)

app.layout = html.Div(style={"height": "100vh", "display": "flex"}, children=[
    html.Div(style={"width": "350px", "padding": "1rem", "background": "#2b2b2b", "overflowY": "auto"}, children=[
        html.H4("UMAP 3D Visualizer", style={"color": "white"}),
        html.P(f"📊 Points: {len(points):,}", style={"color": "lightblue"}),
        html.P(f"🏷️ Labels: {label_col or 'None'}", style={"color": "lightgreen"}),
        html.P(f"📐 Range: {max_range:.2f}", style={"color": "orange"}),
        dbc.Button("🔄 Reset View", id="reset-btn", color="primary", style={"width": "100%", "margin": "5px 0"}),
        dbc.Button("🎯 Center on Data", id="center-btn", color="secondary", style={"width": "100%"}),
        html.Label("Point Size:", style={"color": "white"}),
        dcc.Slider(id="point-size-slider", min=1, max=10, value=3, step=1,
                   marks={i: str(i) for i in range(1, 11)}, tooltip={"placement": "bottom", "always_visible": True}),
        html.Div(id="click-info", style={"color": "yellow", "padding": "10px", "background": "#333", "borderRadius": "5px"},
                 children="Click a point to see details")
    ]),
    html.Div(style={"flex": 1, "position": "relative"}, children=[
        dash_deck.DeckGL(
            id="deck-gl",
            tooltip={"html": "<b>ID:</b> {id}<br/><b>Label:</b> {label}<br/>{position}",
                     "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"}},
            style={"position": "absolute", "top": 0, "bottom": 0, "left": 0, "right": 0}
        ),
        dcc.Store(id="view-state", data=default_view_state),
        dcc.Store(id="point-size", data=1)
    ])
])

def create_deck_config(view_state, size):
    view = pdk.ViewState(
        target=view_state.get("target", center),
        zoom=view_state.get("zoom", default_view_state["zoom"]),
        rotation_x=view_state.get("rotationX", 30),
        rotation_orbit=view_state.get("rotationOrbit", 0),
    )

    layer = pdk.Layer(
        "PointCloudLayer",
        data=points,
        get_position="position",
        get_color="color",
        point_size=size,
        pickable=True,
        auto_highlight=True
    )

    return pdk.Deck(
        initial_view_state=view,
        views=[pdk.View("OrbitView", controller=True)],
        layers=[layer],
        parameters={"clearColor": [0.1, 0.1, 0.1, 1]}
    ).to_json()


@app.callback(Output("deck-gl", "data"), [Input("view-state", "data"), Input("point-size", "data")])
def update_deck(view_state, size): return create_deck_config(view_state, size)

@app.callback(Output("point-size", "data"), Input("point-size-slider", "value"))
def update_point_size(size): return size

@app.callback(
    Output("view-state", "data"),
    [Input("reset-btn", "n_clicks"),
     Input("center-btn", "n_clicks"),
     Input("deck-gl", "viewState"),
     Input("deck-gl", "clickInfo")],
    State("view-state", "data"),
    prevent_initial_call=True
)
def update_view_state(r, c, v, click_info, cur):
    t = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if t == "reset-btn":
        return default_view_state

    if t == "center-btn":
        return {**cur, "target": center, "zoom": default_view_state["zoom"]}

    if t == "deck-gl" and v:  # normale Bewegung
        return {
            "target": v.get("target", cur["target"]),
            "zoom": v.get("zoom", cur["zoom"]),
            "rotationX": v.get("rotationX", cur.get("rotationX", 30)),
            "rotationOrbit": v.get("rotationOrbit", cur.get("rotationOrbit", 0))
        }

    # Klick/Doppelklick auf Punkt → Kamera dort hin
    if t == "deck-gl" and click_info and click_info.get("object"):
        pos = click_info["object"]["position"]
        return {
            "target": pos,
            "zoom": cur.get("zoom", default_view_state["zoom"]) + 2,
            "rotationX": 30,
            "rotationOrbit": 0
        }

    raise dash.exceptions.PreventUpdate


@app.callback(Output("click-info", "children"), Input("deck-gl", "clickInfo"), prevent_initial_call=True)
def show_click_info(info):
    if not info or not info.get("object"): return "Click a point to see details"
    o, p = info["object"], info["object"]["position"]
    return [html.P("🔍 Point Details:", style={"fontWeight": "bold", "color": "white"}),
            html.P(f"ID: {o.get('id')}", style={"color": "lightblue"}),
            html.P(f"Label: {o.get('label')}", style={"color": "lightgreen"}),
            html.P(f"X: {p[0]:.3f}, Y: {p[1]:.3f}, Z: {p[2]:.3f}", style={"color": "lightgray", "fontSize": "0.8rem"})]

if __name__ == "__main__":
    app.run()
