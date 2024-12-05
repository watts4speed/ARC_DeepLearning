import torch
from torch.utils.data import Dataset, DataLoader
import utils


# PyTorch Dataset Framework: Custom processing of data (incl. Augmentations, Padding)
class ARCDataset(Dataset):
    def __init__(self, X, y, stage="train", aug=[True, True, True]):
        self.stage = stage

        if self.stage == "train":
            self.X, self.y = utils.preprocess_matrix(X, y, aug)
        else:
            self.X = utils.get_final_matrix(X, self.stage)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        inp = self.X[idx]
        inp = torch.tensor(inp, dtype=torch.float32)

        if self.stage == "train":
            outp = self.y[idx]
            outp = torch.tensor(outp, dtype=torch.float32)
            return inp, outp
        else:
            return inp

# Defining loaders function to ease data preparation
def data_load(X_train, y_train, stage="train", aug=[False, False, False], batch_size=1, shuffle=False):
    # Define what augmentations are applied, batch size, and if shuffling is desired

    data_set = ARCDataset(X_train, y_train, stage=stage, aug=aug)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    del data_set

    return data_loader
