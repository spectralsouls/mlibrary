import functools, itertools
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# input size: 227x227x3

def sequence(inp, layers): 
    return functools.reduce(lambda x, f: f(x), layers, inp)

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(227), 
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])])


# Download the data here: http://cs231n.stanford.edu/tiny-imagenet-200.zip
# Expects the data to be in the same directory

train_data = datasets.ImageFolder("./tiny-imagenet-200/train", transform=transform)
test_data = datasets.ImageFolder("./tiny-imagenet-200/test")

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layers= [ 
            nn.Conv2d(3, 96, 11, 4), 
            nn.BatchNorm2d(96),
            F.relu,
            lambda x: F.max_pool2d(x, 3, 2), 
            nn.Conv2d(96, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            F.relu,
            lambda x: F.max_pool2d(x, 3, 2),
            nn.Conv2d(256, 384, 3, padding=1),
            F.relu,
            nn.Conv2d(384, 384, 3, padding=1),
            F.relu,
            nn.Conv2d(384, 256, 3, padding=1),
            F.relu,
            lambda x: F.max_pool2d(x, 3, 2),
            lambda x: F.dropout(x, 0.5),
            lambda x: torch.flatten(x, start_dim=1),
            nn.Linear(6*6*256, 4096),
            F.relu,
            lambda x: F.dropout(x, 0.5),
            nn.Linear(4096, 4096),
            F.relu,
            lambda x: F.dropout(x, 0.5),
            nn.Linear(4096, 1000),
            #lambda x: F.softmax(x, dim=1) cross entropy in pytorch uses softmax internally
        ]
    def __call__(self, inp): return sequence(inp, self.layers)

    def get_params(self):
        params = [layer.parameters() for layer in self.layers if isinstance(layer, nn.Module)]
        p = itertools.chain(*params)
        return p
    
    def to_device(self, device='cpu'):
        for layer in self.layers:
            layer.to(device) if isinstance(layer, nn.Module) else layer

if __name__ == "__main__":
    train_ImageNet = DataLoader(train_data, batch_size=128, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # mps only works on MacOS


model = AlexNet()
model.to_device(device)
optimizer = optim.SGD(model.get_params(), lr=0.01, momentum=0.9, weight_decay=0.0005)

def train_step():
    model.train()
    for x_train, y_train in train_ImageNet:
        x_train, y_train = x_train.to(device), y_train.to(device)
        y_pred = model(x_train)
        loss = F.cross_entropy(y_pred, y_train)
        loss_value = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_value
    

# NOTE: takes a long time to train (~ 10 minutes per step on a Macbook Air M3)
for i in range(20):
   loss = train_step()
   print(loss)

