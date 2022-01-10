from gcommand_dataset import GCommandLoader
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 37 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 30)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def netTrain(net, trainloader, optimizer, criterion, epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        mini_batches = 200
        if i % mini_batches == mini_batches - 1:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / mini_batches:.3f}')
            running_loss = 0.0

    print('Finished Training')


def netTest(net, loader, isValidation=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if isValidation:
        print(f'Accuracy of the network on ~ 6,800 validation images: {100 * correct // total} %')
    else:
        print(f'Accuracy of the network on 30,000 train images: {100 * correct // total} %')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = Net()
    # # load previously trained model
    # PATH='./'
    # net.load_state_dict(torch.load(PATH))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    batch_size = 10
    epochs = 20
    workers = 16

    dataset = GCommandLoader('./data/train')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=None)

    dataset = GCommandLoader('./data/valid')

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=None,
        num_workers=workers, pin_memory=True, sampler=None)

    # for k, (input, label) in enumerate(train_loader):
    #     print(input.size(), len(label))
    for e in range(epochs):
        netTrain(net, train_loader, optimizer, criterion, e)

    # # save our trained model
    # PATH = './cifar_net.pth'
    # torch.save(net.state_dict(), PATH)

        netTest(net, train_loader)
        netTest(net, valid_loader, True)
        print("on epoch No. " + str(e))


if __name__ == '__main__':
    main()
