import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import numpy as np
import imp
import os
import insightface


# rom bob.extension.download import get_file

class PyTorchModel(TransformerMixin, BaseEstimator):
    """
    Base Transformer using pytorch models


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)

    preprocessor:
        A function that will transform the data right before forward. The default transformation is `X/255`

    """

    def __init__(
            self,
            checkpoint_path=None,
            config=None,
            preprocessor=lambda x: (x - 127.5) / 128.0,
            device='cpu',
            **kwargs
    ):

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor_ = preprocessor
        self.device = device

    def preprocessor(self, X):
        X = self.preprocessor_(X)
        if X.size(2) != 112:
            X = torch.nn.functional.interpolate(X, mode='bilinear', size=(112, 112), align_corners=False)
        return X

    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        **Parameters:**

        image : 2D :py:class:`numpy.ndarray` (floats)
        The image to extract the features from.

        **Returns:**

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
        The list of features extracted from the image.
        """
        if self.model is None:
            self._load_model()

            self.model.eval()

            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False

        # X = check_array(X, allow_nd=True)
        # X = torch.Tensor(X)
        X = self.preprocessor(X)

        return self.model(X)  # .detach().numpy()

    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def to(self, device):
        self.device = device

        if self.model != None:
            self.model.to(self.device)


def _get_iresnet_file():  # This function is unused
    return "../../models/models/iresnet100-73e07ba7.pth"


class IResnet100(PyTorchModel):
    """
    ArcFace model (RESNET 100) from Insightface ported to pytorch
    """

    def __init__(self,
                 preprocessor=lambda x: (x - 127.5) / 128.0,
                 device='cpu'
                 ):
        self.device = device
        filename = "../recreate_icip2022_face_reconstruction/models/"

        path = os.path.dirname(filename)
        config = os.path.join(path,
                              "iresnet.py")  # https://github.com/nizhib/pytorch-insightface/blob/main/insightface/iresnet.py
        checkpoint_path = os.path.join(path, "iresnet100-73e07ba7.pth")

        super(IResnet100, self).__init__(
            checkpoint_path, config, device=device
        )

    def _load_model(self):  # This function is unused
        model = imp.load_source("module", self.config).iresnet100(self.checkpoint_path)
        self.model = model


def get_FaceRecognition_transformer(device='cpu'):
    FaceRecognition_transformer = IResnet100(device=device)
    # FaceRecognition_transformer = insightface.iresnet100(pretrained=True)
    return FaceRecognition_transformer


class ID_Loss:
    def __init__(self, device='cpu'):
        self.FaceRecognition_transformer = get_FaceRecognition_transformer(device=device)

    def __call__(self, img1, img2):
        embedding1 = self.FaceRecognition_transformer.transform(img1 * 255.0)  # Note: input img should be in (0,1)
        embedding2 = self.FaceRecognition_transformer.transform(img2 * 255.0)  # Note: input img should be in (0,1)
        return torch.nn.MSELoss()(embedding1, embedding2)
