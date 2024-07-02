import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject.enum import *
# TODO: add device


class QuantNet(nn.Module):
    def __init__(self, input_dim, width, weight_quantizer):
        super(QuantNet, self).__init__()
        self.width = width
        self.fc1 = qnn.QuantLinear(input_dim, self.width, weight_quant=weight_quantizer)
        self.fc2 = qnn.QuantLinear(self.width, self.width, weight_quant=weight_quantizer)
        self.fc3 = qnn.QuantLinear(self.width, self.width, weight_quant=weight_quantizer)
        self.fc4 = qnn.QuantLinear(self.width, self.width, weight_quant=weight_quantizer)
        self.fc5 = qnn.QuantLinear(self.width, 10, weight_quant=weight_quantizer)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


def train(model, optimizer, criterion, train_loader, test_loader, epochs, baseline):
    train_losses = []
    test_losses = []
    l0_norms = []
    l1_norms = []
    l2_norms = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Calculate average train loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Calculate test loss
        test_loss = test(model, criterion, test_loader)
        test_losses.append(test_loss)

        # Calculate norms
        if baseline:
            l0_norm = sum(p.norm(0).item() for p in model.parameters())
            l1_norm = sum(p.norm(1).item() for p in model.parameters())
            l2_norm = sum(p.norm(2).item() for p in model.parameters())
            l0_norms.append(l0_norm)
            l1_norms.append(l1_norm)
            l2_norms.append(l2_norm)
        else:
            l0_norm = 0
            l1_norm = 0
            l2_norm = 0
            for name, module in model.named_modules():
                if hasattr(module, 'quant_weight'):
                    quant_weight = module.quant_weight().tensor
                    l0_norm += quant_weight.norm(0).item()
                    l1_norm += quant_weight.norm(1).item()
                    l2_norm += quant_weight.norm(2).item()
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}')
        print(f'L0 Norm: {l0_norm:.6f}, L1 Norm: {l1_norm:.6f}, L2 Norm: {l2_norm:.6f}')

    return train_losses, test_losses, l0_norms, l1_norms, l2_norms


def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss


def build_dataloaders(batch_size=128, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def run_experiment(BITS, ROUND, EPOCHS=10, BASELINE=False):
    class WeigthQuant(Int8WeightPerTensorFloat):
        quant_type = QuantType.INT  # integer quantization
        bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
        scaling_impl_type = ScalingImplType.STATS  # scale based on statistics
        scaling_stats_op = StatsOp.MAX  # scale statistics is the absmax value
        restrict_scaling_type = RestrictValueType.FP  # scale factor is a floating point value
        scaling_per_output_channel = False  # scale is per tensor
        signed = True  # quantization range is signed
        narrow_range = True  # quantization range is [-127,127] rather than [-128, 127]
        zero_point_impl = ZeroZeroPoint
        bit_width = BITS
        if ROUND == "stochastic":
            float_to_int_impl_type = FloatToIntImplType.STOCHASTIC_ROUND
        else:
            float_to_int_impl_type = FloatToIntImplType.ROUND

    if BASELINE:
        quantizer = None
    else:
        quantizer = WeigthQuant

    trainloader, testloader = build_dataloaders()

    model = QuantNet(input_dim=32*32*3, width=64, weight_quantizer=quantizer)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, test_losses, l0_norms, l1_norms, l2_norms = train(model=model, optimizer=optimizer, 
                                                                    criterion=criterion, train_loader=trainloader, 
                                                                    test_loader=testloader, epochs=EPOCHS, baseline=BASELINE)
    
    return model, train_losses, test_losses, l0_norms, l1_norms, l2_norms
