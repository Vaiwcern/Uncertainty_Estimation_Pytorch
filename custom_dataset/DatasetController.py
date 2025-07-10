import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_dataset.CustomDatasetPT import *

class DatasetController:
    @staticmethod
    def get_roadtracer_train_wrapper(dataset_path, batch_size, add_channel, buffer_size=None):
        print(f"📂 Loading RoadTracer train dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        train_dataset = RTDatasetPT(
            dataset_dir=dataset_path,
            train=True,
            add_channel=add_channel,
            normalize=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return train_loader

    @staticmethod
    def get_roadtracer_test_wrapper(dataset_path, batch_size, add_channel):
        print(f"📂 Loading RoadTracer test dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        test_dataset = RTDatasetPT(
            dataset_dir=dataset_path,
            train=False,
            add_channel=add_channel,
            normalize=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return test_loader

    @staticmethod
    def get_massachusetts_train_wrapper(dataset_path, batch_size, add_channel, buffer_size=None):
        print(f"📂 Loading Massachusetts train dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        train_dataset = MassachusettsDatasetPT(
            dataset_dir=dataset_path,
            split='train',
            add_channel=add_channel,
            normalize=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return train_loader

    @staticmethod
    def get_massachusetts_test_wrapper(dataset_path, batch_size, add_channel):
        print(f"📂 Loading Massachusetts test dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        test_dataset = MassachusettsDatasetPT(
            dataset_dir=dataset_path,
            split='test',
            add_channel=add_channel,
            normalize=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return test_loader

    @staticmethod
    def get_drive_train_wrapper(dataset_path, batch_size, add_channel, buffer_size=None):
        print(f"📂 Loading Drive train dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        train_dataset = DRIVEDatasetPT(
            dataset_dir=dataset_path,
            train=True,
            add_channel=add_channel,
            normalize=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return train_loader

    @staticmethod
    def get_drive_test_wrapper(dataset_path, batch_size, add_channel):
        print(f"📂 Loading Drive test dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        test_dataset = DRIVEDatasetPT(
            dataset_dir=dataset_path,
            train=False,
            add_channel=add_channel,
            normalize=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return test_loader

    @staticmethod
    def get_cell_nuclei_train_wrapper(dataset_path, batch_size, add_channel, buffer_size=None):
        print(f"📂 Loading Cell Nuclei train dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        train_dataset = CellNucleiDatasetPT(
            dataset_dir=dataset_path,
            train=True,
            add_channel=add_channel,
            normalize=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return train_loader

    @staticmethod
    def get_cell_nuclei_test_wrapper(dataset_path, batch_size, add_channel):
        print(f"📂 Loading Cell Nuclei test dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        test_dataset = CellNucleiDatasetPT(
            dataset_dir=dataset_path,
            train=False,
            add_channel=add_channel,
            normalize=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return test_loader
