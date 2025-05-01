import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset,SubsetRandomSampler
from tbsim.configs.base import TrainConfig

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories,DiffusionNoise

from trajdata.custom_func.get_lane_info import get_lane_info
from trajdata.custom_func.get_actions import get_actions_inverse_dynamics

NEIGHBOR_DISTANCE = 30
def default_value():
    return NEIGHBOR_DISTANCE


class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, train_config: TrainConfig):
        super(UnifiedDataModule, self).__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None

    @property
    def modality_shapes(self):
        # TODO: better way to figure out channel size?
        return dict(
            image=(3 + self._data_config.history_num_frames + 1,  # semantic map + num_history + current
                   self._data_config.raster_size,
                   self._data_config.raster_size),
            static=(3,self._data_config.raster_size,self._data_config.raster_size),
            dynamic=(self._data_config.history_num_frames + 1,self._data_config.raster_size,self._data_config.raster_size)

        )

    def setup(self, stage = None):
        raise NotImplementedError("UnifiedDataModule is not implemented")

