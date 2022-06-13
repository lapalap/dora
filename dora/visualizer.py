import numpy as np
from PIL import Image
import plotly.graph_objects as go

from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output, no_update


class SAMSCollection:
    def __init__(self, filenames):
        self.filenames = filenames

    def __getitem__(self, idx):
        return Image.open(self.filenames[idx])


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

        mask = np.ones(len(embeddings), bool)
        mask[self.indices_of_outlier_neuron_indices] = 0
        ## mask is a bool array
        self.indices_of_normal_neuron_indices = np.where(mask == True)[0]

        self.color_codes = np.array(["normal" for i in range(len(embeddings))])
        self.color_codes[self.indices_of_outlier_neuron_indices] = "outlier"

        self.filenames = [
            (self.storage_dir + "/sAMS/" + self.experiment_name + f"/{idx}.jpg")
            for idx in self.neuron_idx
        ]

        self.neurons = SAMSCollection(filenames=self.filenames)

        self.curve_number_mapping = {0: "normal", 1: "outlier"}

    def get_outlier_neurons(self):
        data = []
        count = 0
        for idx in self.indices_of_outlier_neuron_indices:
            data.append(
                {
                    "neuron_idx": self.neuron_idx[idx],
                    "image": Image.open(self.filenames[idx]),
                }
            )
            count += 1
        return data

    def get_normal_neurons(self):
        data = []
        count = 0
        for idx in self.indices_of_normal_neuron_indices:
            data.append(
                {
                    "neuron_idx": self.neuron_idx[idx],
                    "image": Image.open(self.filenames[idx]),
                }
            )
            count += 1
        return data

    def render_plotly(self):
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=self.embeddings[self.indices_of_normal_neuron_indices, 0],
                    y=self.embeddings[self.indices_of_normal_neuron_indices, 1],
                    mode="markers",
                    name=self.curve_number_mapping[0],
                ),
                go.Scatter(
                    x=self.embeddings[self.indices_of_outlier_neuron_indices, 0],
                    y=self.embeddings[self.indices_of_outlier_neuron_indices, 1],
                    mode="markers",
                    name=self.curve_number_mapping[0],
                ),
            ]
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

        if notebook_mode is False:
            app = Dash(__name__)
        else:
            app = JupyterDash(__name__)

        fig.update_traces(hoverinfo="none", hovertemplate=None)

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
            num = pt["pointIndex"]
            curve_number = pt["curveNumber"]

            point_type = self.curve_number_mapping[curve_number]

            if point_type == "normal":
                idx = np.array(self.neuron_idx)[self.indices_of_normal_neuron_indices][
                    num
                ]
            else:
                idx = self.outlier_neuron_idx[num]

            img_src = Image.open(self.filenames[self.neuron_idx.index(idx)])
            desc = f"Neuron idx: {idx}"

            children = [
                html.Div(
                    [
                        html.Img(src=img_src, style={"width": "100%"}),
                        html.P(f"{desc}"),
                        html.P(f"type: {point_type}"),
                    ],
                    style={"width": "200px", "white-space": "normal"},
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
