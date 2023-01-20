import torch as th
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import torch.nn as nn
from data_reader import get_unlabeled_dataset
import csv
from argparse import ArgumentParser

cuda0 = th.device("cuda:0" if th.cuda.is_available() else "cpu")
print(th.__version__)

if __name__ == "__main__":

    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--dataset_path', type=str, required=True)
    arg_parser.add_argument('--model_path', type=str, required=True)
    arg_parser.add_argument('--submit_path', type=str, required=True)
    args = arg_parser.parse_args()


    net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 100)
    net.load_state_dict(th.load(args.model_path))
    net = net.to(cuda0)
    net.eval()

    test_loader, names = get_unlabeled_dataset(args.dataset_path)

    with open(args.submit_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'label'])
        i = 0
        for data in test_loader:
            images, labels = data

            images = images.to(cuda0)

            # calculate outputs by running images through the network
            outputs1 = net(images)

            outputs = outputs1
            # the class with the highest energy is what we choose as prediction
            _, predicted = th.max(outputs.data, 1)

            writer.writerow([names[i], predicted.cpu().numpy()[0]])
            i = i + 1
