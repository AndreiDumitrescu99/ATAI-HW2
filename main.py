import torch as th
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_reader import get_labeled_dataset
from argparse import ArgumentParser


cuda0 = th.device("cuda:0" if th.cuda.is_available() else "cpu")

def train_loop(
    dataset_root_path: str,
    save_path: str,
    save_path_acc: str,
    task: int = 1,
    epochs: int = 30,
    batch_size: int = 72,
    lr: float = 1e-5,
    pseudolabel: bool = False,
    path_to_unlabled_data: str = './task1/train_data/images/unlabeled/',
    path_to_model_to_use_for_pseudo_label: str = '',
    start_from_pretrained_model: str = None
):

    net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 100)

    if start_from_pretrained_model != None:
        print("Loading pretrained model...")
        net.load_state_dict(th.load('C:\\Users\\andre\\Desktop\\Master - 2\\ATAI\\HW2\\models\\last_try_task2_2.pt'))

    net = net.to(cuda0)

    train_loader, test_loader = get_labeled_dataset(dataset_root_path, batch_size, task, pseudolabel, path_to_unlabled_data, path_to_model_to_use_for_pseudo_label)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    
    minimum_val_loss = 1e6
    maximum_acc = 0.0

    train_loss = []
    validation_loss = []

    train_acc = []
    validation_acc = []

    print("Starting training...")
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        cnt = 0
        net.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(cuda0)
            labels = labels.type(th.LongTensor)
            labels = labels.to(cuda0)

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.item()

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            cnt = cnt + 1
        
        print("~" *  50)
        print("Epoch: ", epoch + 1, "Training Loss: ", running_loss / cnt)
        train_loss.append(running_loss / cnt)

        net.eval()

        running_loss = 0.0
        cnt = 0
        with th.no_grad():
            for i, data in enumerate(test_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(cuda0)
                labels = labels.type(th.LongTensor)
                labels = labels.to(cuda0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # print statistics
                running_loss += loss.item()

                cnt = cnt + 1

        
        print("Epoch: ", epoch + 1, "Validation Loss: ", running_loss / cnt)
        validation_loss.append(running_loss / cnt)

        correct = 0
        total = 0
        for data in train_loader:
            images, labels = data

            images = images.to(cuda0)
            labels = labels.type(th.LongTensor)
            labels = labels.to(cuda0)

            # calculate outputs by running images through the network
            outputs = net(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = th.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        print("Epoch: ", epoch + 1, "Train Accuracy: ", 100 * correct / total)
        train_acc.append(100 * correct / total)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with th.no_grad():
            for data in test_loader:
                images, labels = data

                images = images.to(cuda0)
                labels = labels.type(th.LongTensor)
                labels = labels.to(cuda0)

                # calculate outputs by running images through the network
                outputs = net(images)

                # the class with the highest energy is what we choose as prediction
                _, predicted = th.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print("Validation Accuracy: ", 100 * correct / total)
        validation_acc.append(100 * correct / total)

        if (running_loss / cnt < minimum_val_loss):
            print("Saving model...")
            th.save(net.state_dict(), save_path)
            minimum_val_loss = running_loss / cnt

        if (correct / total > maximum_acc):
            print("Saving model acc...")
            th.save(net.state_dict(), save_path_acc)
            maximum_acc = correct / total

    return train_loss, validation_loss, train_acc, validation_acc

if __name__ == "__main__":

    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--task', type=int, choices=[1, 2], required=True)
    arg_parser.add_argument('--dataset_path', type=str, required=True)
    arg_parser.add_argument('--path_to_save_best_val_loss', type=str, required=True)
    arg_parser.add_argument('--path_to_save_best_val_acc', type=str, required=True)
    arg_parser.add_argument('--epochs', type=int, required=False, default=15)
    arg_parser.add_argument('--batch_size', type=int, required=False, default=72)
    arg_parser.add_argument('--learning_rate', type=float, required=False, default=3e-5)
    arg_parser.add_argument('--pseudolabel', type=bool, required=False, default=False)
    arg_parser.add_argument('--path_to_unlabled_data', type=str, required=False, default='./task1/train_data/images/unlabeled/')
    arg_parser.add_argument('--path_to_model_to_use_for_pseudo_label', type=str, required=False, default='')
    arg_parser.add_argument('--start_from_pretrained_model', type=str, required=False, default=None)

    args = arg_parser.parse_args()

    train_loop(
        dataset_root_path=args.dataset_path,
        save_path=args.path_to_save_best_val_loss,
        save_path_acc=args.path_to_save_best_val_acc,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        pseudolabel=args.pseudolabel,
        path_to_unlabled_data=args.path_to_unlabled_data,
        path_to_model_to_use_for_pseudo_label=args.path_to_model_to_use_for_pseudo_label,
        start_from_pretrained_model=args.start_from_pretrained_model
    )