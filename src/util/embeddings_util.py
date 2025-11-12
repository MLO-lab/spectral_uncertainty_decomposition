import logging, os
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import gc
from torch import Tensor


logger = logging.getLogger(__name__)

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        pass

class AllMpnet(BaseEmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2",
                                    cache_folder= os.environ['HF_CACHE'],
                                    device = torch.device("cuda"))
    def encode(self, texts: List[str], encoding_arguments: dict) -> np.ndarray:
        return self.model.encode(texts, **encoding_arguments)

    
def embedding_factory(config: dict) -> BaseEmbeddingModel:
    model_name = config["model_name"]
    if model_name == "all-mpnet-base-v2":
        return AllMpnet()
    else:
        raise NotImplementedError

