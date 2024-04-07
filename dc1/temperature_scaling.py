import torch
from torch import optim
import torch.nn as nn


class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits):
        return torch.softmax(logits / self.temperature, dim=1)

    def set_temperature(self, validation_loader, model, device):
        self.to(device)
        model.eval()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list).detach()
        labels = torch.cat(labels_list).detach()

        # Optimize the temperature
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def nll():
            loss = nn.CrossEntropyLoss()
            loss_val = loss(self(logits), labels)
            loss_val.backward()
            return loss_val

        optimizer.step(nll)
