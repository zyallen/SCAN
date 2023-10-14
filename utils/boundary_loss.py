import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from utils.boundary_units import simplex, probs2one_hot, one_hot,one_hot2hd_dist

class BoundaryLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss

