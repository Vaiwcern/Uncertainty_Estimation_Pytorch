# dataset_factory.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_dataset.DatasetController import DatasetController

class DatasetFactory:
    _train_map = {
        "RT": DatasetController.get_roadtracer_train_wrapper,
        "Mass": DatasetController.get_massachusetts_train_wrapper,
        "Drive": DatasetController.get_drive_train_wrapper,
        "Nuclei": DatasetController.get_cell_nuclei_train_wrapper,
    }

    _test_map = {
        "RT": DatasetController.get_roadtracer_test_wrapper,
        "Mass": DatasetController.get_massachusetts_test_wrapper,
        "Drive": DatasetController.get_drive_test_wrapper,
        "Nuclei": DatasetController.get_cell_nuclei_test_wrapper,
    }

    @staticmethod
    def get_train_loader(
        name: str,
        dataset_path: str,
        batch_size: int,
        add_channel: bool,
        num_workers: int,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        if name not in DatasetFactory._train_map:
            raise ValueError(f"Unsupported dataset: {name}")

        loader_fn = DatasetFactory._train_map[name]
        return loader_fn(
            dataset_path=dataset_path,
            batch_size=batch_size,
            add_channel=add_channel,
            num_workers=num_workers,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

    @staticmethod
    def get_test_loader(
        name: str,
        dataset_path: str,
        batch_size: int,
        add_channel: bool,
        num_workers: int,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        if name not in DatasetFactory._test_map:
            raise ValueError(f"Unsupported dataset: {name}")

        loader_fn = DatasetFactory._test_map[name]
        return loader_fn(
            dataset_path=dataset_path,
            batch_size=batch_size,
            add_channel=add_channel,
            num_workers=num_workers,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
