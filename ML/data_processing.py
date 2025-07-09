import torch as t
from torch.utils.data import Dataset, DataLoader, random_split

from ML.model import MLParams

class ConcData(Dataset):
    
    def __init__(self, names, sample_ratio = 1.0):
        super().__init__()

        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")


        coords = []
        concs = []

        for name in names:
            coord = t.load("./Data/Torch/" + name + "_torchcoord.pt")
            coords.append(coord)

            conc = t.load("./Data/Torch/" + name + "_torchconc.pt")
            concs.append(conc)
        
        self.concs = t.cat(concs, 0).to(self.device)
        self.coords = t.cat(coords, 0).to(self.device)

        k = int(len(self.concs) * sample_ratio)

        indices = t.randperm(self.concs.size(0))[:k]
        
        self.concs = self.concs[indices]
        self.coords = self.coords[indices]


    def __getitem__(self, index):
        return self.coords[index], self.concs[index]
    
    def __len__(self):
        return len(self.coords)


def get_loaders(names, p: MLParams, sample_ratio):

    data = ConcData(names, sample_ratio)

    size_train = int(p.params["train_test_split"] * len(data))
    size_test = len(data) - size_train

    train_dataset, test_dataset = random_split(data, [size_train, size_test])

    if p["batch_size"] < 0:
        batch_size_train = size_train
        batch_size_test = size_test
    else:
        batch_size_train = p["batch_size"]
        batch_size_test = p["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size_train)
    test_loader = DataLoader(test_dataset, batch_size_test)

    return train_loader, test_loader
