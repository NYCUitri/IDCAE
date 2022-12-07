from torch.utils.data import Dataset

class MelDataLoader(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature, label, nm_label = self.dataset[idx]
        return feature, label, nm_label