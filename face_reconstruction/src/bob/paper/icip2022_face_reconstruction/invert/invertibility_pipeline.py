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


import logging
from bob.bio.base.pipelines.vanilla_biometrics.score_writers import (
    FourColumnsScoreWriter,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import BioAlgorithm

logger = logging.getLogger(__name__)
import tempfile

from bob.bio.base.pipelines.vanilla_biometrics.pipelines import VanillaBiometricsPipeline, check_valid_pipeline

class InvertBiometricsPipeline(VanillaBiometricsPipeline):
    """
    Invert Biometrics Pipeline

    This is the backbone of most biometric recognition systems.
    It implements three subpipelines and they are the following:

     - :py:class:`VanillaBiometrics.train_background_model`: Initializes or trains your transformer.
        It will run :py:meth:`sklearn.base.BaseEstimator.fit`

     - :py:class:`VanillaBiometrics.create_biometric_reference`: Creates biometric references
        It will run :py:meth:`sklearn.base.BaseEstimator.transform` followed by a sequence of
        :py:meth:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.BioAlgorithm.enroll`

     - :py:class:`VanillaBiometrics.compute_scores`: Computes scores
        It will run :py:meth:`sklearn.base.BaseEstimator.transform` followed by a sequence of
        :py:meth:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.BioAlgorithm.score`


    Example
    -------
       >>> from bob.pipelines.transformers import Linearize
       >>> from sklearn.pipeline import make_pipeline
       >>> from bob.bio.base.pipelines.vanilla_biometrics import Distance, InvertBiometricsPipeline
       >>> estimator_1 = Linearize()
       >>> transformer = make_pipeline(estimator_1)
       >>> biometric_algoritm = Distance()
       >>> pipeline = InvertBiometricsPipeline(transformer, biometric_algoritm)
       >>> pipeline(samples_for_training_back_ground_model, samplesets_for_enroll, samplesets_for_scoring)  # doctest: +SKIP


    To run this pipeline using Dask, used the function :py:func:`dask_vanilla_biometrics`.

    Example
    -------
      >>> from bob.bio.base.pipelines.vanilla_biometrics import dask_vanilla_biometrics
      >>> pipeline = InvertBiometricsPipeline(transformer, biometric_algoritm)
      >>> pipeline = dask_vanilla_biometrics(pipeline)
      >>> pipeline(samples_for_training_back_ground_model, samplesets_for_enroll, samplesets_for_scoring).compute()  # doctest: +SKIP


    Parameters
    ----------

      transformer: :py:class`sklearn.pipeline.Pipeline` or a `sklearn.base.BaseEstimator`
        Transformer that will preprocess your data

      biometric_algorithm: :py:class:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.BioAlgorithm`
        Biometrics algorithm object that implements the methods `enroll` and `score` methods

      score_writer: :any:`bob.bio.base.pipelines.vanilla_biometrics.ScoreWriter`
          Format to write scores. Default to :any:`bob.bio.base.pipelines.vanilla_biometrics.FourColumnsScoreWriter`

    """

    def __init__(
        self,
        transformer,
        inversionAttack_transformer,
        biometric_algorithm,
        score_writer=None,
    ):
        self.transformer = transformer
        self.inversionAttack_transformer = inversionAttack_transformer
        self.biometric_algorithm = biometric_algorithm
        self.score_writer = score_writer
        if self.score_writer is None:
            tempdir = tempfile.TemporaryDirectory()
            self.score_writer = FourColumnsScoreWriter(tempdir.name)

        check_valid_pipeline(self)

    def __call__(
        self,
        background_model_samples,
        biometric_reference_samples,
        probe_samples,
        invert_references_samples,
        allow_scoring_with_all_biometric_references=True,
    ):
        logger.info(
            f" >> Vanilla Biometrics: Training background model with pipeline {self.transformer}"
        )

        # Training background model (fit will return even if samples is ``None``,
        # in which case we suppose the algorithm is not trainable in any way)
        self.transformer = self.train_background_model(background_model_samples)

        logger.info(
            f" >> Creating biometric references with the biometric algorithm {self.biometric_algorithm}"
        )

        # Create biometric samples
        biometric_references = self.create_biometric_reference(
            biometric_reference_samples
        )

        logger.info(
            f" >> Computing scores with the biometric algorithm {self.biometric_algorithm}"
        )

        # Scores all probes
        scores_probes, _ = self.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

        # Scores all inverted references
        scores_inversionAttack, _ = self.compute_scores_invertedReferences(
            invert_references_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

        return scores_probes, scores_inversionAttack

    def compute_scores_invertedReferences(
        self,
        invert_references_samples,
        biometric_references,
        allow_scoring_with_all_biometric_references=True,
    ):
        # probes is a list of SampleSets
        invert_reference_features = self.inversionAttack_transformer.transform(
            invert_references_samples
        )
        scores = self.biometric_algorithm.score_samples(
            invert_reference_features,
            biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        # scores is a list of Samples
        return scores, invert_reference_features