import csv
import cv2
import numpy as np
import torch as th
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import cm
cuda0 = th.device('cuda:0')

transform_train = transforms.Compose([

    transforms.Resize(156),
    transforms.ColorJitter(),
    transforms.RandomErasing(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_valid = transforms.Compose([
    transforms.Resize(156),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
th.manual_seed(0)


class ImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def read_image(img_path: str):

    img = cv2.imread(img_path)
    img = img / 255
    img = np.transpose(img, [2, 0, 1])
    return img

def pseudo_label(path_to_unlabled_images_root_dir: str, model_to_use: str):

    net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 100)
    net.load_state_dict(th.load(model_to_use))
    net.to(cuda0)
    X = []
    y = []
    names = []
    net.eval()

    print("Loading unlabled data...")
    for i in range(0, 26445):

        img = read_image(path_to_unlabled_images_root_dir + str(i) + '.jpeg')

        X.append(th.tensor(img, dtype=th.float32))
        y.append(th.tensor(0))
        names.append(str(i) + '.jpeg')


    val_dataset = ImageDataset(X, y, transform_valid)
    val_loader = DataLoader(val_dataset, batch_size=64)

    cnt = 0

    final_x = []
    final_y = []
    totals = 0
    for data in val_loader:
        images, labels = data

        images = images.to(cuda0)

        # calculate outputs by running images through the network
        outputs = net(images)
        maximum_prob, indices = th.max(th.nn.Softmax(dim=1)(outputs.data), 1)

        maximum_prob = maximum_prob.cpu().numpy()
        indices = indices.cpu().numpy()
        images.cpu()

        i = 0
        for p in maximum_prob:
            if p > 0.9:
                final_x.append(X[cnt])
                final_y.append(th.tensor(indices[i], dtype=th.int32))
                totals = totals + 1
            i = i + 1
            cnt = cnt + 1
    
    net.cpu()
    del net
    print(cnt, totals)
    return final_x, final_y

def get_unlabeled_dataset(dataset_path: str):

    X = []
    y = []
    names = []

    for i in range(0, 5000):

        img = read_image(dataset_path + str(i) + '.jpeg')

        X.append(th.tensor(img, dtype=th.float32))
        y.append(th.tensor(0))
        names.append(str(i) + '.jpeg')


    val_dataset = ImageDataset(X, y, transform_valid)
    val_loader = DataLoader(val_dataset, batch_size=1)

    return val_loader, names


def get_labeled_dataset(
    file_path: str,
    batch_size: int,
    task: int = 1,
    pseudo_label_bool: bool = False,
    path_to_unlabled_images_root_dir = './task1/train_data/images/unlabeled/',
    model_to_use = ''
):

    X = []
    y = []
    cnt = 1
    print("Loading dataset...")
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:

            if cnt > 1:
                img = read_image('./' + row[0])

                X.append(img)
                aux_y = np.zeros([100])
                aux_y[(int(row[1]))] = 1
                y.append(int(row[1]))
            cnt = cnt + 1
    
    X = th.tensor(np.array(X), dtype=th.float32)
    y = th.tensor(np.array(y), dtype=th.long)

    my_dataset = TensorDataset(X, y)

    if task == 1:
        train_set, val_set = random_split(my_dataset, [21555, 2000])
    else:
        train_set, val_set = random_split(my_dataset, [46000, 4000]) 


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    X_train, y_train = [], []
    X_val, y_val = [], []

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        X_train.append(inputs[0])
        y_train.append(labels[0])

    if pseudo_label_bool == True and task == 1:
        final_x, final_y = pseudo_label(path_to_unlabled_images_root_dir, model_to_use)

        X_train = X_train + final_x
        y_train = y_train + final_y

    v = np.zeros([100])
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        X_val.append(inputs[0])
        y_val.append(labels[0])
        v[labels[0]] = v[labels[0]] + 1
    
    
    train_dataset = ImageDataset(X_train, y_train, transform_train)
    test_dataset = ImageDataset(X_val, y_val, transform_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader
