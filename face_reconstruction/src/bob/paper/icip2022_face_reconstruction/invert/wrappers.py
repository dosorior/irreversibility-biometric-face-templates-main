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

from sklearn.pipeline import Pipeline


def get_invert_pipeline(FR_transformer, inv_transformer, feature_extractor):

    ### TODO: Look here Hatef

    # pipeline = make_pipeline(
    #    *[item for item in FR_transformer],
    #    inv_transformer,
    #    *[item for item in FR_transformer]
    # )
    pipeline = Pipeline(
        FR_transformer.steps
        + [
            ("inverted-samples", inv_transformer),
            ("inverted-features", feature_extractor),
        ]
    )

    return pipeline
