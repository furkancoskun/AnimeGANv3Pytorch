from torch.utils.data.dataset import Dataset
from torch import Tensor
from random import randint
import numpy as np
import os
import cv2

class ImageDataSetTest(Dataset):
    def __init__(self, img_dir: str, img_size=(256, 256)) -> None:
        self.img_dir = img_dir
        self.img_size = img_size

        self.test_data = []
        self.test_data_length = 0

        self.init_data()
    
    def init_data(self):
        dataset_dir = "./dataset"
        self.test_data_dir = f"{dataset_dir}/{self.img_dir}"
        self.test_data = os.listdir(self.test_data_dir)
        self.test_data_length = len(self.test_data)
    
    def load_photo(self, path: str) -> np.ndarray:
        image = cv2.imread(os.path.join(self.test_data_dir, path))
        h, w, _ = image.shape
        image = cv2.resize(image, self.img_size)
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image1 = image1.transpose((2, 0, 1))
        image1 = image1 / 127.5 - 1.0
        return image1, (w,h)
    
    def __len__(self):
        return self.test_data_length
    
    def __getitem__(self, index) -> dict[str, Tensor]:
        photo, orig_image_size = self.load_photo(self.test_data[index])
        return {
            "photo": photo,
            "im_name": self.test_data[index],
            "orig_size": orig_image_size,
        }