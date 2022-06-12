import plotly.graph_objects as go

import numpy as np

from dash import Dash, dcc, html, Input, Output, no_update


class OutlierVisualizer:
    def __init__(
        self,
        embeddings,
        outlier_neuron_idx,
        neuron_idx,
        experiment_name,
        storage_dir,
    ):
        self.embeddings = embeddings
        self.outlier_neuron_idx = outlier_neuron_idx
        self.neuron_idx = neuron_idx
        self.storage_dir = storage_dir
        self.experiment_name = experiment_name

        self.indices_of_outlier_neuron_indices = np.sort(
            np.array(self.neuron_idx)
        ).searchsorted(self.outlier_neuron_idx)

        mask = np.ones(len(embeddings), np.bool)
        mask[self.indices_of_outlier_neuron_indices] = 0
        self.indices_of_normal_neuron_indices = mask

        self.color_codes = np.array(["normal" for i in range(len(embeddings))])
        self.color_codes[self.indices_of_outlier_neuron_indices] = "outlier"

    def render_plotly(self):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.embeddings[self.indices_of_normal_neuron_indices, 0],
                y=self.embeddings[self.indices_of_normal_neuron_indices, 1],
                mode="markers",
                name="Normal",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.embeddings[self.indices_of_outlier_neuron_indices, 0],
                y=self.embeddings[self.indices_of_outlier_neuron_indices, 1],
                mode="markers",
                name="Outlier",
            )
        )

        fig.update_layout(
            title=f"DORA Experiment: {self.experiment_name}",
            xaxis_title="X",
            yaxis_title="Y",
            legend_title="Neuron type",
            font=dict(size=25),
        )

        return fig

    def visualize(self, notebook_mode=False):
        fig = self.render_plotly()
        fig.update_traces(hoverinfo="none", hovertemplate=None)

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

            from PIL import Image

            filename = (
                self.storage_dir
                + "/sAMS/"
                + self.experiment_name
                + f"/{self.neuron_idx[num]}.jpg"
            )
            img_src = Image.open(filename)
            desc = f"Neuron idx: {self.neuron_idx[num]}"

            children = [
                html.Div(
                    [
                        html.Img(src=img_src, style={"width": "100%"}),
                        html.P(f"{desc}"),
                    ],
                    # style={"width": "200px", "white-space": "normal"},
                )
            ]

            return True, bbox, children

        app.layout = html.Div(
            [
                dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
                dcc.Tooltip(id="graph-tooltip"),
            ],
            style={"height": "100vh", "padding": 10},
        )

        if notebook_mode == False:
            app.run_server(debug=True)
        else:
            app.run_server(debug=True, mode="inline")
