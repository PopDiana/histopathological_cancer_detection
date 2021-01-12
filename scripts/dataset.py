from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils import *


# create a class for dataset properties
class CustomDataset(Dataset):
    def __init__(self, data, directory='./', transform=None):
        super().__init__()
        self.data = data.values
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name, label = self.data[index]
        image_path = os.path.join(self.directory, image_name + '.tif')
        image = cv2.imread(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
