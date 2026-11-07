import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import argparse

import litenet as QUANWNS
import util as U

import numpy as np
import random
import os

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def ParseArgs():
    parser = argparse.ArgumentParser(description='BWNs Pytorch Example.')
    parser.add_argument('--quan-mode', type=str, default="BINARY", metavar='Q',
                        help='QUAN mode(default: FP32)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='batch size for training(default: 50)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs', type=int, default=180, metavar='N',
                        help='number of epoch to train(default: 100)')
    parser.add_argument('--lr-epochs', type=int, default=50, metavar='N',
                        help='number of epochs to decay learning rate(default: 15)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-file-prefix', type=str, default="mnist", metavar='P',
                        help='')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main():
    args = ParseArgs()
    print("args: ", args)
    if args.cuda:
        setup_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("kwargs: ", kwargs)

    logf = open('./MnistNet2_BNN_KD.log', 'w')

    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    ###################################################################
    ##             Load Train Dataset                                ##
    ###################################################################
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    ###################################################################
    ##             Load Test Dataset                                ##
    ###################################################################
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    teacher_model = QUANWNS.Airbench94_mnist(class_num=10)
    # Your teacher model
    teacher_model.load_state_dict(torch.load('models/Airbench94_mnist.pth'))

    print(teacher_model)

    model = QUANWNS.BWN_BM3_FOBNN_scale2_v3()

    print(model)

    if args.cuda:
        teacher_model.cuda()
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def distillation(y, labels, teacher_scores, T, alpha):
        return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1)) * (T * T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

    best_acc = 0.0
    for epoch_index in range(1, args.epochs + 1):
        lr = adjust_learning_rate(learning_rate, optimizer, epoch_index, args.lr_epochs)
        train(args, epoch_index, train_loader, model, teacher_model, optimizer, loss_fn=distillation, lr=lr, logf=logf)
        train_evaluate(args, model, train_loader, logf)
        acc = test(args, model, test_loader, logf)
        if acc > best_acc:
            best_acc = acc
            print('Saving model ...')
            torch.save(model.state_dict(), './MnistNet2_BNN_KD.pth')
            print('*** DONE! ***')

    logf.write("best_acc: " + str(best_acc) + '\n')
    logf.close()


def train(args, epoch_index, train_loader, model, teacher_model, optimizer, loss_fn=None, lr=None, logf=None):
    model.train()
    teacher_model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        optimizer.zero_grad()

        output = model(data)

        with torch.inference_mode():
            teacher_output = teacher_model(data)

        loss = loss_fn(output, target, teacher_output, T=11, alpha=0.3)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            logss = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}@{:.6f}'.format(
                epoch_index, batch_idx * len(data), len(train_loader.dataset),
                             100. * batch_idx / len(train_loader), loss.item(), lr)
            print(logss)
            logf.write(logss + '\n')
            logf.flush()


def train_evaluate(args, model, train_loader, logf=None):
    model.eval()
    train_loss = 0
    correct = 0

    with torch.inference_mode():
        for data, target in train_loader:
            if args.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(data)
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()

    logss = '\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        train_loss / len(train_loader.dataset), correct.item(),
        len(train_loader.dataset),
        100. * correct.item() / len(train_loader.dataset))
    print(logss)
    logf.write(logss + '\n')
    logf.flush()


def test(args, model, test_loader, logf=None):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.inference_mode():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()

    acc = 100. * correct.item() / len(test_loader.dataset)

    logss = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss / len(test_loader.dataset), correct.item(),
        len(test_loader.dataset), acc)
    print(logss)
    logf.write(logss + '\n')
    logf.flush()

    return acc


def adjust_learning_rate(learning_rate, optimizer, epoch_index, lr_epoch):
    # lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr


if __name__ == '__main__':
    main()
