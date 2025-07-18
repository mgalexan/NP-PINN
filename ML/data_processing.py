import torch as t
from torch.utils.data import Dataset, DataLoader, random_split

from ML.model import MLParams
from Util.evaluate_function import evaluate_env
from Environment.env_class import ParamSpace
from dolfinx.fem import Function

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


class PDatata(Dataset):
    def __init__(self, P_i: Function, env: ParamSpace, sample_ratio = 1.0):
        super().__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")


        P_mat = evaluate_env(P_i, env.geometry)[0]

        P_mat = t.from_numpy(P_mat).reshape(-1, 1).to(self.device)

        xvals = t.linspace(0, env.geometry.width, env.geometry.shape_x)
        yvals = t.linspace(0, env.geometry.height, env.geometry.shape_y)

        coords = t.meshgrid(xvals, yvals, indexing= "ij")
        coords = t.stack(coords, dim= -1).reshape(-1,2).to(self.device)

        k = int(len(coords) * sample_ratio)

        indices = t.randperm(coords.size(0))[:k]
        
        self.data = P_mat[indices].float()
        self.coords = coords[indices].float()
    
    def __getitem__(self, index):
        return self.coords[index], self.data[index] 
    
    def __len__(self):
        return len(self.coords)
        




def get_loaders(input, p: MLParams, sample_ratio = 1.0, data_type: str = "concentration"):

    if data_type == "concentration":
        data = ConcData(input, sample_ratio)
    
    elif data_type == "pressure":
        data = PDatata(input[0], input[1], sample_ratio)


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
