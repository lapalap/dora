import umap


class UmapVisualizer:
    def __init__(self):

        self.reducer = umap.UMAP()
        self.embeddings = None

    def run(self, activations):
        num_items = activations.shape[0]
        self.embeddings = self.reducer.fit_transform(activations)

    def save(self, filename):
        assert (
            self.embeddings is not None
        ), "Embeddings are still done, maybe you did not run UMAP yet"
        raise NotImplementedError

    def show(self):
        assert (
            self.embeddings is not None
        ), "Embeddings are still done, maybe you did not run UMAP yet"
