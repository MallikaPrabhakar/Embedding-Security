# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
import os
import torch
import numpy as np
from bob.bio.base.pipelines.vanilla_biometrics import Distance
from bob.bio.base.pipelines.vanilla_biometrics import VanillaBiometricsPipeline
from bob.bio.face.utils import dnn_default_cropping
from bob.bio.face.utils import embedding_transformer
# from bob.bio.face.utils import cropped_positions_arcface
from bob.bio.facexzoo.utils import cropped_positions_arcface
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from bob.bio.face.annotator import BobIpMTCNN

# from bob.learn.pytorch.architectures.facexzoo import FaceXZooModelFactory
from bob.bio.facexzoo.backbones import FaceXZooModelFactory

# from bob.bio.face.embeddings.pytorch import PyTorchModel
################# PyTorchModel from bob.bio.face.embeddings.pytorch
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
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
        preprocessor=lambda x: x / 255,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor = preprocessor
        self.memory_demanding = memory_demanding
        self.device = device

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
        import torch

        if self.model is None:
            self._load_model()
        X = check_array(X, allow_nd=True)
        X = torch.Tensor(X)
        with torch.no_grad():
            X = self.preprocessor(X)

        def _transform(X):
            with torch.no_grad():
                return self.model(X.to(self.device)).cpu().detach().numpy()

        if self.memory_demanding:
            features = np.array([_transform(x[None, ...]) for x in X])

            # If we ndim is > than 3. We should stack them all
            # The enroll_features can come from a source where there are `N` samples containing
            # nxd samples
            if features.ndim >= 3:
                features = np.vstack(features)

            return features

        else:
            return _transform(X)

    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def place_model_on_device(self, device=None):
        import torch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        if self.model is not None:
            self.model.to(device)
################# PyTorchModel


# from bob.bio.face.embeddings.pytorch import FaceXZooModel
class FaceXZooModel(PyTorchModel):
    """
    FaceXZoo models
    """

    def __init__(
        self,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        memory_demanding=False,
        device=None,
        arch="MobileFaceNet",
        head='MV-Softmax',
        **kwargs,
    ):

        self.arch = arch
        self.head = head
        _model = FaceXZooModelFactory(self.arch, self.head)
        filename = _model.get_facexzoo_file()
        checkpoint_name = _model.get_checkpoint_name()
        config = None
        path = os.path.dirname(filename)
        checkpoint_path = filename#os.path.join(path, self.arch + ".pt")

        super(FaceXZooModel, self).__init__(
            checkpoint_path,
            config,
            memory_demanding=memory_demanding,
            preprocessor=preprocessor,
            device=device,
            **kwargs,
        )

    def _load_model(self):

        _model = FaceXZooModelFactory(self.arch, self.head)
        self.model = _model.get_model()

        model_dict = self.model.state_dict()

        pretrained_dict = torch.load(
            self.checkpoint_path, map_location=torch.device("cpu")
        )["state_dict"]

        pretrained_dict_keys = pretrained_dict.keys()
        model_dict_keys = model_dict.keys()

        new_pretrained_dict = {}
        for k in model_dict:
            new_pretrained_dict[k] = pretrained_dict["backbone." + k]
        model_dict.update(new_pretrained_dict)
        self.model.load_state_dict(model_dict)

        self.model.eval()
        self.place_model_on_device()
        
    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None):
        return self


        
# from bob.bio.face.embeddings.pytorch import iresnet_template
# iresnet_template is replaced with pipeline_template
def pipeline_template(embedding, annotation_type, fixed_positions=None):
    # DEFINE CROPPING
    cropped_image_size = (112, 112)
    if annotation_type == "eyes-center" or annotation_type == "bounding-box":
        # Hard coding eye positions for backward consistency
        # cropped_positions = {
        cropped_positions = cropped_positions_arcface()
        if annotation_type == "bounding-box":
            # This will allow us to use `BoundingBoxAnnotatorCrop`
            cropped_positions.update(
                {"topleft": (0, 0), "bottomright": cropped_image_size}
            )

    else:
        cropped_positions = dnn_default_cropping(cropped_image_size, annotation_type)

    annotator = BobIpMTCNN(min_size=40, factor=0.709, thresholds=(0.1, 0.2, 0.2))
    transformer = embedding_transformer(
        cropped_image_size=cropped_image_size,
        embedding=embedding,
        cropped_positions=cropped_positions,
        fixed_positions=fixed_positions,
        color_channel="rgb",
        annotator=annotator,
    )

    algorithm = Distance()

    return VanillaBiometricsPipeline(transformer, algorithm)



def MobileFaceNet(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the MobileFaceNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`MobileFaceNet` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(
            arch="MobileFaceNet", memory_demanding=memory_demanding
        ),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



def ResNet50_ir(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the ResNet50_ir pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`ResNet50_ir` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="ResNet50_ir", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def ResNet152_irse(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the ResNet152_irse pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`ResNet152_irse` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="ResNet152_irse", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



def HRNet(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the HRNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`HRNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="HRNet", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def EfficientNet_B0(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the EfficientNet_B0 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`EfficientNet_B0` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="EfficientNet_B0", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



def TF_NAS_A(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the TF_NAS_A pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`TF_NAS_A` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="TF_NAS_A", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )




def LightCNN29(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the LightCNN29 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`LightCNN29` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="LightCNN29", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )




def GhostNet(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the GhostNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`GhostNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.


    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="GhostNet", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def AttentionNet56(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the AttentionNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`AttentionNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """
    return pipeline_template(
        embedding=FaceXZooModel(arch="AttentionNet56", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )

    

def AttentionNet92(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the AttentionNet pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`AttentionNet` to extract the features


    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """
    return pipeline_template(
        embedding=FaceXZooModel(arch="AttentionNet92", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def ResNeSt50(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the ResNeSt50 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`ResNeSt50` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="ResNeSt50", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )




def ReXNet_1(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the ReXNet_1 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`ReXNet_1` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="ReXNet_1", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



def RepVGG_A0(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the RepVGG_A0 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`RepVGG_A0` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="RepVGG_A0", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



def RepVGG_B0(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the RepVGG_B0 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`RepVGG_B0` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="RepVGG_B0", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



def RepVGG_B1(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the RepVGG_B1 pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`RepVGG_B1` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(arch="RepVGG_B1", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def SwinTransformer_preprocessor(X):
    preprocessor_=lambda x: (x - 127.5) / 128.0

    X = preprocessor_(X)
    if X.size(2) != 224:
        X = torch.nn.functional.interpolate(X, mode='bilinear', size=(224, 224), align_corners=False)
    
    return X

def SwinTransformer_S(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the SwinTransformer_S pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`RepVGG_B1` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(preprocessor=SwinTransformer_preprocessor,
            arch="SwinTransformer_S", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )


def SwinTransformer_T(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the SwinTransformer_T pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`RepVGG_B1` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(preprocessor=SwinTransformer_preprocessor,
            arch="SwinTransformer_T", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



############# Heads


def AM_Softmax(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the AM_Softmax pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`AM_Softmax` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="AM-Softmax", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



def AdaM_Softmax(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the AdaM_Softmax pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`AdaM_Softmax` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="AdaM-Softmax", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )




def AdaCos(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the AdaCos pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`AdaCos` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="AdaCos", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )





def ArcFace(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the ArcFace pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`ArcFace` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="ArcFace", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )





def MV_Softmax(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the MV_Softmax pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`MV_Softmax` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="MV-Softmax", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )





def CurricularFace(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the CurricularFace pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`CurricularFace` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="CurricularFace", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )





def CircleLoss(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the CircleLoss pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`CircleLoss` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="CircleLoss", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )



def NPCFace(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the NPCFace pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`NPCFace` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="NPCFace", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )




def MagFace(annotation_type, fixed_positions=None, memory_demanding=False):
    """
    Get the MagFace pipeline which will crop the face :math:`112 \\times 112` and
    use the :py:class:`MagFace` to extract the features

    .. warning::

       If you are at Idiap, please use the option `-l sge-gpu` while running the `vanilla-biometrics` pipeline.



    Parameters
    ----------

      annotation_type: str
         Type of the annotations (e.g. `eyes-center')

      fixed_positions: dict
         Set it if in your face images are registered to a fixed position in the image

      memory_demanding: bool

    """

    return pipeline_template(
        embedding=FaceXZooModel(head="MagFace", memory_demanding=memory_demanding),
        annotation_type=annotation_type,
        fixed_positions=fixed_positions,
    )