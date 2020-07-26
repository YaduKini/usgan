import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader

# Directory containing the data.
directory = '/home/yadu/BMC/PMSD_praktikum/PMSD/generatorinput'
root = '/home/yadu/BMC/PMSD_praktikum/PMSD/sampleinput'

def get_ultrasound_data(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)

    # Create the dataloader.
    dataloader = DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader


def get_generator_input(params):
    """
    data to be loaded into the generator
    """

    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    # Create the dataset.
    dataset = dset.ImageFolder(root=directory, transform=transform)

    # Create the dataloader.
    geninput_dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=params['bsize'],
                                             shuffle=True)

    return geninput_dataloader