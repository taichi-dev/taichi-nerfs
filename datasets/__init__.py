from .colmap import ColmapDataset
from .nerf import NeRFDataset
from .nsvf import NSVFDataset

dataset_dict = {
    'nerf': NeRFDataset,
    'nsvf': NSVFDataset,
    'colmap': ColmapDataset,
}
