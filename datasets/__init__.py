from .toys import ToysDataset
from .beauty import BeautyDataset
from .clothing import ClothingDataset
from .sports import SportsDataset

DATASETS = {
    ToysDataset.code(): ToysDataset,
    BeautyDataset.code(): BeautyDataset,
    ClothingDataset.code(): ClothingDataset,
    SportsDataset.code(): SportsDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
