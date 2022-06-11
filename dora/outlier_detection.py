import umap
import numpy as np
from pyod.models.pca import PCA

OD_METHODS = {
    "PCA": PCA
    ## add more later
}


class OutlierDetector:
    def __init__(self, name="PCA", outliers_fraction=0.05, random_state=1):

        self.reducer = umap.UMAP()
        assert (
            name in OD_METHODS
        ), f"Expected Outlier detection method to be one of: {OD_METHODS} but got {name}"

        self.outlier_detector = OD_METHODS[name](
            contamination=outliers_fraction, random_state=random_state
        )
        self.embeddings = None
        self.activations = None

    def run(self, activations):
        self.activations = activations
        self.embeddings = self.reducer.fit_transform(activations)

        outliers = self.detect_outliers(activations=activations)
        return outliers

    def detect_outliers(self, activations):
        classifier = self.outlier_detector.fit(activations)
        ## returns a one hot vetcor
        outliers = classifier.predict(activations)

        ## convert to indices from one hot and take first element from tuple
        ## se: https://stackoverflow.com/questions/50646102/what-is-the-purpose-of-numpy-where-returning-a-tuple
        return np.where(outliers == 1)[0]
