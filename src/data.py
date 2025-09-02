import torchvision
import torchvision.transforms as T
import torch


def get_cifar10(batch_size=128, num_workers=2, pin_memory=False):
    transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),  # CIFAR-10 normalization
        ]
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),  # CIFAR-10 normalization
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def get_cifar100(batch_size=128, num_workers=2, pin_memory=False):
    transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),  # CIFAR-100 normalization
        ]
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),  # CIFAR-100 normalization
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def get_dataset(dataset_name, batch_size=128, num_workers=2, pin_memory=False):
    """Get dataset by name"""
    if dataset_name == "cifar10":
        return get_cifar10(batch_size, num_workers, pin_memory)
    elif dataset_name == "cifar100":
        return get_cifar100(batch_size, num_workers, pin_memory)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
