import torch
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, indexed_paths, transform=None):
        self.indexed_paths = indexed_paths
        self.transform = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()

    def __len__(self):
        return len(self.indexed_paths)

    def __getitem__(self, index):
        i, path = self.indexed_paths[index]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return i, img
        except Exception as e:
            print(f"Fehler bei Bild {i} ({path}): {e}")
            return i, None

def safe_collate(batch):
    batch = [item for item in batch if item[1] is not None]
    if not batch:
        return None, None
    indices, images = zip(*batch)
    return torch.tensor(indices), torch.stack(images)
