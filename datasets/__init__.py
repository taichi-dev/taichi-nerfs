from .colmap import ColmapDataset
from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .ngp import NGPDataset

dataset_dict = {
    'nerf': NeRFDataset,
    'nsvf': NSVFDataset,
    'colmap': ColmapDataset,
    'ngp': NGPDataset,
}
