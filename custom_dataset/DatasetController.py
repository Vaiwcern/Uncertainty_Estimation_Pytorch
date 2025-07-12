import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader, DistributedSampler
from custom_dataset.CustomDataset import *


class DatasetController:
    @staticmethod
    def _create_loader(dataset, batch_size, num_workers, distributed=False, rank=0, world_size=1, shuffle=True):
        if distributed:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
            shuffle = False  # Must be False when using sampler
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

    @staticmethod
    def get_roadtracer_train_wrapper(dataset_path, batch_size, add_channel, num_workers=4,
                                     distributed=False, rank=0, world_size=1):
        print(f"📂 Loading RoadTracer train dataset from {dataset_path}")
        dataset = RTDataset(dataset_dir=dataset_path, train=True, add_channel=add_channel, normalize=True)
        return DatasetController._create_loader(dataset, batch_size, num_workers, distributed, rank, world_size)

    @staticmethod
    def get_roadtracer_test_wrapper(dataset_path, batch_size, add_channel, num_workers=4,
                                    distributed=False, rank=0, world_size=1):
        print(f"📂 Loading RoadTracer test dataset from {dataset_path}")
        dataset = RTDataset(dataset_dir=dataset_path, train=False, add_channel=add_channel, normalize=True)
        return DatasetController._create_loader(dataset, batch_size, num_workers, distributed, rank, world_size, shuffle=False)

    @staticmethod
    def get_massachusetts_train_wrapper(dataset_path, batch_size, add_channel, num_workers=4,
                                        distributed=False, rank=0, world_size=1):
        print(f"📂 Loading Massachusetts train dataset from {dataset_path}")
        dataset = MassachusettsDataset(dataset_dir=dataset_path, split='train', add_channel=add_channel, normalize=True)
        return DatasetController._create_loader(dataset, batch_size, num_workers, distributed, rank, world_size)

    @staticmethod
    def get_massachusetts_test_wrapper(dataset_path, batch_size, add_channel, num_workers=4,
                                       distributed=False, rank=0, world_size=1):
        print(f"📂 Loading Massachusetts test dataset from {dataset_path}")
        dataset = MassachusettsDataset(dataset_dir=dataset_path, split='test', add_channel=add_channel, normalize=True)
        return DatasetController._create_loader(dataset, batch_size, num_workers, distributed, rank, world_size, shuffle=False)

    @staticmethod
    def get_drive_train_wrapper(dataset_path, batch_size, add_channel, num_workers=4,
                                distributed=False, rank=0, world_size=1):
        print(f"📂 Loading Drive train dataset from {dataset_path}")
        dataset = DRIVEDataset(dataset_dir=dataset_path, train=True, add_channel=add_channel, normalize=True)
        return DatasetController._create_loader(dataset, batch_size, num_workers, distributed, rank, world_size)

    @staticmethod
    def get_drive_test_wrapper(dataset_path, batch_size, add_channel, num_workers=4,
                               distributed=False, rank=0, world_size=1):
        print(f"📂 Loading Drive test dataset from {dataset_path}")
        dataset = DRIVEDataset(dataset_dir=dataset_path, train=False, add_channel=add_channel, normalize=True)
        return DatasetController._create_loader(dataset, batch_size, num_workers, distributed, rank, world_size, shuffle=False)

    @staticmethod
    def get_cell_nuclei_train_wrapper(dataset_path, batch_size, add_channel, num_workers=4,
                                      distributed=False, rank=0, world_size=1):
        print(f"📂 Loading Cell Nuclei train dataset from {dataset_path}")
        dataset = CellNucleiDataset(dataset_dir=dataset_path, train=True, add_channel=add_channel, normalize=True)
        return DatasetController._create_loader(dataset, batch_size, num_workers, distributed, rank, world_size)

    @staticmethod
    def get_cell_nuclei_test_wrapper(dataset_path, batch_size, add_channel, num_workers=4,
                                     distributed=False, rank=0, world_size=1):
        print(f"📂 Loading Cell Nuclei test dataset from {dataset_path}")
        dataset = CellNucleiDataset(dataset_dir=dataset_path, train=False, add_channel=add_channel, normalize=True)
        return DatasetController._create_loader(dataset, batch_size, num_workers, distributed, rank, world_size, shuffle=False)
