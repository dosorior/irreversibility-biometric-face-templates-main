"""
Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
Written by Hatef Otroshi <hatef.otroshi@idiap.ch>

This file is part of Bob toolbox.

Bob is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

Bob is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Bob. If not, see <http://www.gnu.org/licenses/>.

        *************************

Note: If you use this implementation, please cite the following paper:
    - Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Face Reconstruction from Deep Facial Embeddings using a Convolutional Neural Network" 
      In 2022 IEEE International Conference on Image Processing (ICIP), IEEE, 2022.
"""

from sklearn.base import TransformerMixin, BaseEstimator
from .networks import Generator
import torch

from bob.pipelines import SampleBatch, Sample, SampleSet
import numpy as np


class InversionTransformer(TransformerMixin, BaseEstimator):
    """
    Transforms any :math:`\mathbb{R}^n` into an image :math:`\mathbb{R}^{h \\times w \\times c}`.

    Parameters
    ----------

    checkpoint: str
       Checkpoint of the image generator

    generator:
       instance of the generator network

    """

    def __init__(self, checkpoint, generator=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator() if generator is None else generator

        # TODO: use the checkpoint variable here
        self.generator.load_state_dict(
            torch.load(checkpoint, map_location=self.device,)
        )
        self.generator.eval()
        self.generator.to(self.device)
        self.checkpoint = checkpoint

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None):
        return self

    def transform(self, samples):
        def _transform(data):
            data = data.flatten()
            data = np.reshape(data, (1, data.shape[0], 1, 1))
            embedding = torch.Tensor(data).to(self.device)
            reconstructed_img = self.generator(embedding)[0]
            return reconstructed_img.cpu().detach().numpy() * 255.0

        if isinstance(samples[0], SampleSet):
            return [
                SampleSet(self.transform(sset.samples), parent=sset,)
                for sset in samples
            ]
        else:
            return [
                Sample(_transform(sample.data), parent=sample,) for sample in samples
            ]
