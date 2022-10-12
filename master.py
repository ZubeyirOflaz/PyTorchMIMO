from utils.dataloaders import create_train_dataloader, create_test_dataloader
from torchvision import datasets, transforms
import torch
import time
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
train_load = create_train_dataloader(train_dataset,4,3)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

test_dataset = datasets.MNIST("../data", train=False, transform=transform)
test_load = create_test_dataloader(test_dataset,4,3)
test_test = next(iter(test_load))