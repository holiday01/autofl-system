# ============================================================================
#  AutoFL benchmark expansion — ADVERSARIAL MUTANT A2: mnist_two_optimizers
#  Parent   : pytorch/mnist_main.py (sha256 30a3359d1911d2d859dd090ce20be7ed132a33f508dc389f40366010b3e9ecc8); the parent is a dev-set script,
#             so this mutant leaks nothing about the holdout set and cannot exist in any training corpus.
#  Change   : the network is split into a `features` group (conv1, conv2 -> SGD) and a `head` group (fc1, fc2 -> Adam); both optimizers are stepped every batch and each has its own StepLR stepped every epoch
#  Stresses : I1 (the harness owns the single optimizer; the converter must not construct or step either optimizer inside train_step)
#  Correct conversion : a single loss is returned; no `.step()` / `.zero_grad()` in train_step; the two-optimizer split is documented (merged optimizer or param groups in a build_optimizer hook if the contract allows)
#  Plausible-but-wrong: keeping `opt_f.step(); opt_h.step()` inside train_step, or constructing a second optimizer in the module
#  Notes    : the features/head split is expressed by parameter selection (conv1+conv2 vs fc1+fc2) rather than by restructuring `Net`, to stay within the 15-line budget
#  Diff vs parent: 14 changed lines (+9 / -5), unified diff with zero context below;
#  the header block you are reading is NOT part of the diff (the body after it is parent + diff).
#  --- diff ---
#  --- pytorch/mnist_main.py
#  +++ benchmarks/expansion/adversarial/pytorch/mnist_two_optimizers.py
#  @@ -37,0 +38 @@
#  +    opt_f, opt_h = optimizer
#  @@ -40 +41,2 @@
#  -        optimizer.zero_grad()
#  +        opt_f.zero_grad()
#  +        opt_h.zero_grad()
#  @@ -44 +46,2 @@
#  -        optimizer.step()
#  +        opt_f.step()
#  +        opt_h.step()
#  @@ -128 +131,2 @@
#  -    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
#  +    optimizer = (optim.SGD(list(model.conv1.parameters()) + list(model.conv2.parameters()), lr=0.01, momentum=0.9),   # features
#  +                 optim.Adam(list(model.fc1.parameters()) + list(model.fc2.parameters()), lr=1e-3))                    # head
#  @@ -130 +134 @@
#  -    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#  +    scheduler = [StepLR(opt, step_size=1, gamma=args.gamma) for opt in optimizer]
#  @@ -134 +138 @@
#  -        scheduler.step()
#  +        for s in scheduler: s.step()
#  --- end diff ---
# ============================================================================
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    opt_f, opt_h = optimizer
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt_f.zero_grad()
        opt_h.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        opt_f.step()
        opt_h.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = (optim.SGD(list(model.conv1.parameters()) + list(model.conv2.parameters()), lr=0.01, momentum=0.9),   # features
                 optim.Adam(list(model.fc1.parameters()) + list(model.fc2.parameters()), lr=1e-3))                    # head

    scheduler = [StepLR(opt, step_size=1, gamma=args.gamma) for opt in optimizer]
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        for s in scheduler: s.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
