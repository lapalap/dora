import umap
import plotly.express as px
from pyod.models.pca import PCA

OD_METHODS = {
    "PCA": PCA
    ## add more later
}


class OutlierDetector:
    def __init__(self, name="PCA"):

        # self.od =
        self.reducer = umap.UMAP()
        self.embeddings = None
        self.activations = None

    def run(self, activations):
        self.activations = activations
        self.embeddings = self.reducer.fit_transform(activations)

        outliers = self.detect_outliers(activations=activations)
        print("DONE")

    def detect_outliers(self, activations):
        raise NotImplementedError

    def save(self, filename):
        assert (
            self.embeddings is not None
        ), "Embeddings are still None, maybe you did not run UMAP yet"
        raise NotImplementedError

    def show(self):
        assert (
            self.embeddings is not None
        ), "Embeddings are still None, maybe you did not run UMAP yet"

        fig = px.scatter(
            x=self.embeddings[:, 0], y=self.embeddings[:, 1], title="DORA Results"
        )
        fig.show()
