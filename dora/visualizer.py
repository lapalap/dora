import plotly.express as px
import numpy as np

from dash import Dash, dcc, html, Input, Output, no_update


class OutlierVisualizer:
    def __init__(self, embeddings, outlier_neuron_idx, neuron_idx):
        self.embeddings = embeddings
        self.outlier_neuron_idx = outlier_neuron_idx
        self.neuron_idx = neuron_idx

        self.color_codes = np.array(["normal" for i in range(len(embeddings))])
        self.color_codes[outlier_neuron_idx] = "outlier"

    def show_plotly(self):
        fig = px.scatter(
            x=self.embeddings[:, 0],
            y=self.embeddings[:, 1],
            color=self.color_codes,
        )

        fig.show()

    def visualize(self):
        fig = px.scatter(
            x=self.embeddings[:, 0],
            y=self.embeddings[:, 1],
            color=self.color_codes,
        )

        app = Dash(__name__)

        @app.callback(
            Output("graph-tooltip", "show"),
            Output("graph-tooltip", "bbox"),
            Output("graph-tooltip", "children"),
            Input("graph-basic-2", "hoverData"),
        )
        def display_hover(hoverData):
            if hoverData is None:
                return False, no_update, no_update

            # demo only shows the first point, but other points may also be available
            pt = hoverData["points"][0]
            bbox = pt["bbox"]
            num = pt["pointNumber"]

            img_src = ".dora/sAMS/model.avgpool/36.jpg"
            desc = "neuron desc"
            if len(desc) > 300:
                desc = desc[:100] + "..."

            children = [
                html.Div(
                    [
                        html.Img(src=img_src, style={"width": "100%"}),
                        html.P(f"{desc}"),
                    ],
                    style={"width": "200px", "white-space": "normal"},
                )
            ]

            return True, bbox, children

        app.layout = html.Div(
            [
                dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
                dcc.Tooltip(id="graph-tooltip"),
            ]
        )
