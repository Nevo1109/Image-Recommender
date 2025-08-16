import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, io
import torchvision.transforms.v2 as transforms


def unzip(batch):
    """
    Because the bytes have different sizes the batch cannot be returned as a tensor. 
    
    This will return it as a tuple with correct dtypes for hdf5 file.
    """
    
    valid_items = []
    
    for item in batch:
        id_, jpeg, dim = item
        # only if it could be loaded
        if jpeg is not None and dim is not None:
            valid_items.append((id_, jpeg, dim))
    
    if not valid_items:
        # return empty list for invalid batch
        return np.array([], dtype="uint32"), [], np.array([], dtype="uint16").reshape(0, 2)
    
    # return remaining items
    ids = np.array([item[0] for item in valid_items], dtype="uint32")
    jpeg_bytes = [item[1] for item in valid_items]
    dimensions = np.array([item[2] for item in valid_items], dtype="uint16")
    
    return ids, jpeg_bytes, dimensions


def create_fixed_sized_images_dataloader(indexed_paths, target_pixels, jpeg_quality=80, batch_size=16, num_workers=6, persistent_workers=True, prefetch_factor=2):
    dataset = JPEGDataset(indexed_paths, target_pixels, jpeg_quality)
    return DataLoader(
        dataset, 
        batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers, 
        prefetch_factor=prefetch_factor,
        collate_fn=unzip  
    )

class JPEGDataset(Dataset):
    def __init__(self, indexed_paths, target_pixels=65536, quality=80):
        self.indexed_paths = list(indexed_paths)
        self.target_pixels = target_pixels
        self.quality = quality

    def __len__(self):
        return len(self.indexed_paths)

    def __getitem__(self, index):
        i, path = self.indexed_paths[index]
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Image could not be loaded")
            h, w = img.shape[:2]

            scale_factor = (self.target_pixels / (w * h)) ** 0.5
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            _, jpeg_bytes = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            return i, jpeg_bytes.flatten(), (new_w, new_h)
        except Exception as e:
            print(f"Fehler bei Bild {i} ({path}): {e}")
            return i, None, None

class ImageDataset(Dataset):
    def __init__(self, indexed_paths, transform=None):
        self.indexed_paths = indexed_paths
        self.transform = transforms or models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms() 

    def __len__(self):
        return len(self.indexed_paths)

    def __getitem__(self, index):
        i, path = self.indexed_paths[index]
        try:
            img = io.decode_image(path, mode=io.ImageReadMode.RGB)
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
